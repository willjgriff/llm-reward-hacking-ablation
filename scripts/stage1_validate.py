#!/usr/bin/env python
"""Stage 1: validate trajectories.jsonl against the schema and print a summary.

Schema violations fail (exit 1). Chain-of-thought / token integrity checks are warnings.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.schema import Trajectory  # noqa: E402


def load(path: Path) -> tuple[list[Trajectory], list[str]]:
    records, errors = [], []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(Trajectory.model_validate_json(line))
            except ValidationError as e:
                tid = json.loads(line).get("trajectory_id", "?")
                errors.append(f"line {lineno} ({tid}):\n{e}")
    return records, errors


def integrity_warnings(records: list[Trajectory], tokenizer) -> list[str]:
    warnings = []
    for r in records:
        for t in r.turns:
            tag = f"{r.trajectory_id} turn {t.turn_index}"
            if t.error:
                continue
            if not t.reasoning and t.finish_reason != "length":
                warnings.append(f"{tag}: empty reasoning (finish_reason={t.finish_reason})")
            if t.rendered_completion is not None:
                has_close = "</think>" in t.rendered_completion
                if bool(t.reasoning) and not has_close and t.finish_reason != "length":
                    warnings.append(f"{tag}: reasoning parsed but no </think> in rendered_completion")
                if not t.reasoning and has_close and t.finish_reason != "length":
                    warnings.append(f"{tag}: </think> present but reasoning empty (parser may have stripped it)")
            if t.completion_token_ids is None or t.prompt_token_ids is None:
                warnings.append(f"{tag}: missing token ids (return_token_ids not honoured?)")
            elif t.usage.output_tokens is not None and len(t.completion_token_ids) != t.usage.output_tokens:
                warnings.append(f"{tag}: len(completion_token_ids)={len(t.completion_token_ids)} != output_tokens={t.usage.output_tokens}")
            if tokenizer is not None and t.rendered_prompt is not None and t.prompt_token_ids is not None:
                reencoded = tokenizer.encode(t.rendered_prompt, add_special_tokens=False)
                if reencoded != t.prompt_token_ids:
                    warnings.append(f"{tag}: re-encoding rendered_prompt does not reproduce prompt_token_ids ({len(reencoded)} vs {len(t.prompt_token_ids)} tokens)")
    return warnings


def summarize(records: list[Trajectory]) -> None:
    print("\n== status ==")
    for status, n in sorted(Counter(r.status for r in records).items()):
        print(f"  {status:10s} {n}")

    groups: dict[tuple[str, str], list[Trajectory]] = defaultdict(list)
    for r in records:
        groups[(r.variant, r.agent_type)].append(r)
    print("\n== per (variant, agent_type) ==")
    print(f"  {'variant':12s} {'agent':8s} {'n':>3s} {'scored':>6s} {'pass':>6s} {'hack':>6s} {'turns':>6s} {'out_tok':>8s} {'errors':>6s}")
    for (variant, agent), rs in sorted(groups.items()):
        scored = [r for r in rs if r.score is not None]
        passed = sum(1 for r in scored if r.score.value == "C")
        hacks = [r.programmatic_hack for r in rs if r.programmatic_hack is not None]
        hack_rate = f"{sum(hacks) / len(hacks):.2f}" if hacks else "n/a"
        pass_rate = f"{passed / len(scored):.2f}" if scored else "n/a"
        mean_turns = sum(r.n_turns for r in rs) / len(rs)
        mean_out = sum(r.token_counts.output_tokens for r in rs) / len(rs)
        errors = sum(1 for r in rs if r.status == "error")
        print(f"  {variant:12s} {agent:8s} {len(rs):3d} {len(scored):6d} {pass_rate:>6s} {hack_rate:>6s} {mean_turns:6.1f} {mean_out:8.0f} {errors:6d}")


def print_example(r: Trajectory, full: bool) -> None:
    data = r.model_dump()
    if not full:
        for t in data["turns"]:
            for key in ("prompt_token_ids", "completion_token_ids"):
                ids = t[key]
                if ids is not None and len(ids) > 12:
                    t[key] = ids[:12] + [f"... ({len(ids)} total)"]
    print("\n== example trajectory ==")
    print(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="data/stage1/<run_id>; reads <run-dir>/trajectories.jsonl")
    parser.add_argument("--file", type=Path, help="explicit JSONL path")
    parser.add_argument("--example-id", help="trajectory_id to print (default: first completed impossible-variant record)")
    parser.add_argument("--full", action="store_true", help="print full token id arrays in the example")
    parser.add_argument("--no-tokenizer-check", action="store_true", help="skip the re-encode warning check")
    args = parser.parse_args()

    path = args.file or (args.run_dir / "trajectories.jsonl" if args.run_dir else None)
    if path is None:
        parser.error("--run-dir or --file required")

    records, errors = load(path)
    print(f"{path}: {len(records)} valid records, {len(errors)} schema errors")
    if errors:
        print("\n== schema errors ==")
        for e in errors:
            print(e)
    if not records:
        sys.exit(1)

    tokenizer = None
    if not args.no_tokenizer_check:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(records[0].model, revision=records[0].model_revision)

    warnings = integrity_warnings(records, tokenizer)
    summarize(records)
    print(f"\n== integrity warnings: {len(warnings)} ==")
    for w in warnings[:50]:
        print(f"  WARN {w}")
    if len(warnings) > 50:
        print(f"  ... {len(warnings) - 50} more")

    if args.example_id:
        example = next((r for r in records if r.trajectory_id == args.example_id), None)
        if example is None:
            parser.error(f"trajectory_id not found: {args.example_id}")
    else:
        example = next(
            (r for r in records if r.status == "completed" and r.programmatic_hack is not None),
            records[0],
        )
    print_example(example, args.full)

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
