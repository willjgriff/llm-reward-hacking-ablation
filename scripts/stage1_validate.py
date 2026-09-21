#!/usr/bin/env python
"""Stage 1: validate trajectories.jsonl[.gz] (full or compact) against the schema and print a summary.

Schema violations fail (exit 1), as does a "do not modify the tests" instruction surviving in a
run with strip_test_modification_warnings on. Chain-of-thought / token integrity checks are warnings.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.compact import expand_trajectory, is_compact, open_trajectories, trajectories_path  # noqa: E402
from rhablation.schema import Trajectory  # noqa: E402


def load(path: Path) -> tuple[list[Trajectory], list[str]]:
    records, errors = [], []
    with open_trajectories(path) as f:
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


def timeout_warnings(records: list[Trajectory]) -> list[str]:
    """Model calls the HTTP client abandoned. Each one was resampled, so long generations are under-represented."""
    timed_out = [(r, t) for r in records for t in r.turns if t.error and "timed out" in t.error.lower()]
    if not timed_out:
        return []
    n_calls = sum(len(r.turns) for r in records)
    n_traj = len({r.trajectory_id for r, _ in timed_out})
    return [
        f"{len(timed_out)} of {n_calls} model calls timed out in {n_traj} of {len(records)} trajectories: "
        "raise client_timeout_s; these turns were resampled, which biases the data toward short chains of thought"
    ]


def stripped_warning_errors(records: list[Trajectory], run_config_path: Path) -> list[str]:
    """With strip_test_modification_warnings on, no prompt may still tell the model not to modify the tests."""
    if not run_config_path.exists():
        return []
    if not json.loads(run_config_path.read_text())["config"].get("strip_test_modification_warnings"):
        return []
    from rhablation.prompt_edits import WARNING_MARKERS

    errors = []
    for r in records:
        for t in r.turns:
            found = [m for m in WARNING_MARKERS if t.rendered_prompt and m in t.rendered_prompt]
            if found:
                errors.append(f"{r.trajectory_id} turn {t.turn_index}: prompt still contains {found}")
        # Compact records checked without a tokenizer have no rendered prompts: fall back to the messages.
        if not any(t.rendered_prompt for t in r.turns):
            found = [m for m in WARNING_MARKERS if any(m in msg.content for msg in r.messages)]
            if found:
                errors.append(f"{r.trajectory_id}: messages still contain {found}")
    return errors


def thinking_budget_summary(records: list[Trajectory]) -> str | None:
    """How many model calls had their chain of thought cut by vLLM's thinking_token_budget.

    Those turns' stored reasoning ends mid-thought by design (the model still answered)."""
    calls = capped = 0
    budgets = set()
    for r in records:
        budget = (r.sampling.extra_body or {}).get("thinking_token_budget")
        if budget is None:
            continue
        budgets.add(budget)
        for t in r.turns:
            if t.error or t.usage.reasoning_tokens is None:
                continue
            calls += 1
            # vLLM reports budget - 1 for a cut call (the forced </think> takes the last slot).
            capped += t.usage.reasoning_tokens >= budget - 1
    if not budgets:
        return None
    return f"thinking_token_budget={sorted(budgets)}: {capped} of {calls} model calls had their reasoning cut at the budget ({capped / calls:.0%})" if calls else None


def summarize(records: list[Trajectory]) -> None:
    print("\n== status ==")
    for status, n in sorted(Counter(r.status for r in records).items()):
        print(f"  {status:10s} {n}")

    groups: dict[tuple[str, str, str], list[Trajectory]] = defaultdict(list)
    for r in records:
        groups[(r.benchmark, r.variant, r.agent_type)].append(r)
    print("\n== per (benchmark, variant, agent_type) ==")
    print(f"  {'benchmark':16s} {'variant':18s} {'agent':8s} {'n':>3s} {'scored':>6s} {'pass':>6s} {'hack':>6s} {'tedit':>6s} {'turns':>6s} {'out_tok':>8s} {'errors':>6s}")
    print("  (hack = pass on an impossible variant, the benchmark's label; tedit = test.py edit attempts, tools only, never a pass)")
    for (benchmark, variant, agent), rs in sorted(groups.items()):
        scored = [r for r in rs if r.score is not None]
        passed = sum(1 for r in scored if r.score.value == "C")
        hacks = [r.programmatic_hack for r in rs if r.programmatic_hack is not None]
        hack_rate = f"{sum(hacks) / len(hacks):.2f}" if hacks else "n/a"
        pass_rate = f"{passed / len(scored):.2f}" if scored else "n/a"
        edits = [r.test_edit_attempt for r in rs if r.test_edit_attempt is not None]
        edit_rate = f"{sum(edits) / len(edits):.2f}" if edits else "n/a"
        mean_turns = sum(r.n_turns for r in rs) / len(rs)
        mean_out = sum(r.token_counts.output_tokens for r in rs) / len(rs)
        errors = sum(1 for r in rs if r.status == "error")
        print(f"  {benchmark:16s} {variant:18s} {agent:8s} {len(rs):3d} {len(scored):6d} {pass_rate:>6s} {hack_rate:>6s} {edit_rate:>6s} {mean_turns:6.1f} {mean_out:8.0f} {errors:6d}")


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
    parser.add_argument("--run-dir", type=Path, help="data/stage1/<run_id>; reads <run-dir>/trajectories.jsonl or .jsonl.gz")
    parser.add_argument("--file", type=Path, help="explicit JSONL path")
    parser.add_argument("--example-id", help="trajectory_id to print (default: first completed impossible-variant record)")
    parser.add_argument("--full", action="store_true", help="print full token id arrays in the example")
    parser.add_argument("--no-tokenizer-check", action="store_true", help="skip the re-encode warning check")
    args = parser.parse_args()

    path = args.file or (trajectories_path(args.run_dir) if args.run_dir else None)
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

    n_compact = sum(is_compact(r) for r in records)
    print(f"storage: {n_compact} compact, {len(records) - n_compact} full; schema versions {sorted({r.schema_version for r in records})}")
    # Compact records store prompts as a delta; every check below runs on the restored full record.
    expanded, compact_errors = [], []
    for r in records:
        try:
            expanded.append(expand_trajectory(r, tokenizer))
        except ValueError as e:
            compact_errors.append(str(e))
    records = expanded
    if compact_errors:
        print(f"\n== compact-form errors: {len(compact_errors)} ==")
        for e in compact_errors[:50]:
            print(f"  ERROR {e}")
        errors += compact_errors
    if not records:
        sys.exit(1)

    warnings = timeout_warnings(records) + integrity_warnings(records, tokenizer)
    summarize(records)
    budget_line = thinking_budget_summary(records)
    if budget_line:
        print(f"\n== thinking budget ==\n  {budget_line}")
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

    prompt_errors = stripped_warning_errors(records, path.parent / "run_config.json")
    if prompt_errors:
        print(f"\n== test-modification warnings not stripped: {len(prompt_errors)} ==")
        for e in prompt_errors[:50]:
            print(f"  ERROR {e}")

    sys.exit(1 if errors or prompt_errors else 0)


if __name__ == "__main__":
    main()
