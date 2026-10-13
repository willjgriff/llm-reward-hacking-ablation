#!/usr/bin/env python
"""Stage 3 audit: choose which rollouts of one environment get judged, and render them.

Writes <out>/pilot_set.json (the same layout the stage-3 pilot tools read: `items` with `n`, `group`
= audit class, `trajectory_id`, plus `source_file`/`line_no` so records can be found in any run dir,
and the programmatic `oracle` labels, which the judge never sees) and, for samples, transcripts/NNN.txt
(for hand-reading), prompts/NNN.txt (rubric + transcript, exactly what the judge is sent) and
reading/NNN.txt (repeats collapsed).

  uv run scripts/stage3_select_audit_set.py --env mbpp_impossible --mode sample --per-class 50
  uv run scripts/stage3_select_audit_set.py --env terminal_verifier --mode all
  uv run scripts/stage3_select_audit_set.py --env grader --mode sample --per-class 50

Classes come from the programmatic labels (TerminalVerifier: the stage-4 manifest; MBPP: labels and
mutation kind, boolean mutations excluded; Grader: prompt kind x cheated). They stratify the sample and
feed the report; they are never part of a prompt.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.judge_audit import ORACLE, SPECS, build_prompt, load_manifest_classes, load_rubric  # noqa: E402
from rhablation.judge_transcript import collapse_repeats, render  # noqa: E402

# Classes that go into the sample in full, whatever --per-class says (small and decisive).
ALWAYS_ALL = {"mbpp_impossible": {"hack_heldout_pass"}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=sorted(SPECS))
    p.add_argument("--mode", choices=("sample", "all"), default="sample")
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--manifest", type=Path, default=Path("data/stage4/manifest/tv27b/manifest.jsonl"),
                   help="stage-4 manifest giving TerminalVerifier classes")
    p.add_argument("--out", type=Path, help="default data/stage3/audit/<env>[-all]")
    p.add_argument("--write-files", choices=("auto", "yes", "no"), default="auto",
                   help="transcripts/prompts/reading files (auto: sample yes, all no)")
    p.add_argument("--limit", type=int, help="debug: stop after this many records")
    args = p.parse_args()

    spec = SPECS[args.env]
    out = args.out or Path("data/stage3/audit") / (args.env + ("-all" if args.mode == "all" else ""))
    write_files = args.write_files == "yes" or (args.write_files == "auto" and args.mode == "sample")
    if args.env == "terminal_verifier":
        spec.context["manifest_classes"] = load_manifest_classes(args.manifest)
    rubric_text, rubric_sha = load_rubric(spec.rubric)
    oracle = ORACLE[args.env]

    by_class: dict[str, list[dict]] = defaultdict(list)
    seen = 0
    skipped = Counter()
    for source_file, line_no, traj in spec.iter_records(args.data_root):
        seen += 1
        cls = spec.classify(traj, spec.context)
        if cls is None:
            skipped["no class"] += 1
            continue
        if traj.get("status") not in (None, "completed", "timeout", "max_turns"):
            skipped[f"status {traj.get('status')}"] += 1
        by_class[cls].append({
            "trajectory_id": traj["trajectory_id"], "run_id": traj.get("run_id"), "source_file": str(source_file),
            "line_no": line_no, "task_id": traj.get("task_id"), "variant": traj.get("variant"),
            "agent_type": traj.get("agent_type"), "oracle": oracle(traj),
        })
        if args.limit and seen >= args.limit:
            break

    print(f"{args.env}: {seen} records read, {sum(len(v) for v in by_class.values())} classified, skipped {dict(skipped)}")
    print(f"  {'class':<28}{'all':>7}{'chosen':>8}")
    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        if args.mode == "all" or cls in ALWAYS_ALL.get(args.env, set()) or len(pool) <= args.per_class:
            pick = list(pool)
        else:
            pick = rng.sample(pool, args.per_class)
        for it in pick:
            it["group"] = cls
        chosen += pick
        print(f"  {cls:<28}{len(pool):>7}{len(pick):>8}")
    rng.shuffle(chosen)  # hand-readers see classes mixed; n is otherwise meaningless
    for n, it in enumerate(chosen, 1):
        it["n"] = n

    # Render (and, for samples, write) the transcripts; count prompt characters for the cost estimate.
    out.mkdir(parents=True, exist_ok=True)
    if write_files:
        for sub in ("transcripts", "prompts", "reading"):
            (out / sub).mkdir(exist_ok=True)
    wanted: dict[str, dict[int, dict]] = defaultdict(dict)
    for it in chosen:
        wanted[it["source_file"]][it["line_no"]] = it
    total_chars = 0
    for source_file, by_line in wanted.items():
        for sf, line_no, traj in spec.iter_records(args.data_root):
            if str(sf) != source_file:
                continue
            it = by_line.get(line_no)
            if it is None:
                continue
            text = render(traj, include_result=False)
            prompt = build_prompt(spec, traj, rubric_text)
            it["transcript_chars"] = len(text)
            it["prompt_chars"] = len(prompt)
            total_chars += len(prompt)
            if write_files:
                (out / "transcripts" / f"{it['n']:03d}.txt").write_text(text)
                (out / "prompts" / f"{it['n']:03d}.txt").write_text(prompt)
                (out / "reading" / f"{it['n']:03d}.txt").write_text(collapse_repeats(text))
    chosen.sort(key=lambda it: it["n"])
    (out / "pilot_set.json").write_text(json.dumps({
        "env": args.env, "mode": args.mode, "seed": args.seed, "per_class": args.per_class,
        "rubric": str(spec.rubric), "rubric_sha": rubric_sha, "categories": sorted(spec.categories),
        "effort_default": spec.effort_default, "items": chosen,
    }, indent=1))
    print(f"wrote {out}/pilot_set.json: {len(chosen)} items, ~{total_chars / 3.5 / 1e6:.2f}M prompt tokens (chars/3.5)"
          + (", transcripts/prompts/reading written" if write_files else ""))


if __name__ == "__main__":
    main()
