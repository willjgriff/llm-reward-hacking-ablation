#!/usr/bin/env python
"""Stage 4, step 1: build the manifest of trajectories for the activation cache (CPU, no model).

Scans the sweep run directories named in the config, assigns each trajectory a class, splits task IDs
into train/eval, fixes the processing order (balanced round-robin so a subset_per_class subset is
complete early) and checks the token layout of every trajectory.

  uv run scripts/stage4_build_manifest.py --config configs/stage4_terminal_verifier.yaml [--out data/stage4/manifest/<manifest_id>] [--limit N]

Writes manifest.jsonl, splits.json and manifest_summary.md.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.stage4_config import load_stage4_config  # noqa: E402
from rhablation.stage4_manifest import build_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default data/stage4/manifest/<manifest_id>")
    ap.add_argument("--limit", type=int, help="stop after this many kept trajectories (smoke test)")
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root that data paths are relative to")
    args = ap.parse_args()

    cfg = load_stage4_config(args.config)
    out = args.out or Path("data/stage4/manifest") / cfg.manifest_id
    out.mkdir(parents=True, exist_ok=True)

    rows, report = build_rows(cfg, args.root, limit=args.limit)
    if not rows:
        print("no trajectories kept", file=sys.stderr)
        return 1

    with (out / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")
    tasks_by_split = defaultdict(set)
    for r in rows:
        tasks_by_split[r.split].add(r.task_id)
    (out / "splits.json").write_text(
        json.dumps(
            {
                "eval_fraction": cfg.split.eval_fraction,
                "seed": cfg.split.seed,
                "n_tasks": len(tasks_by_split["train"]) + len(tasks_by_split["eval"]),
                "eval_task_ids": sorted(tasks_by_split["eval"]),
                "train_task_ids": sorted(tasks_by_split["train"]),
            },
            indent=1,
        )
    )

    # Summary
    counts = Counter((r.cls, r.split) for r in rows)
    classes = sorted({r.cls for r in rows}, key=lambda c: (c not in cfg.order.priority_classes, c))
    tok_total = sum(r.seq_len for r in rows)
    subset = [r for r in rows if r.in_subset]
    lines = [
        f"# Stage 4 manifest `{cfg.manifest_id}`",
        "",
        f"Records scanned: {report.n_records}; kept: {report.n_kept}; excluded: {report.excluded or {}}",
        f"Model: `{cfg.model}` @ `{cfg.model_revision}`; eval fraction {cfg.split.eval_fraction} (seed {cfg.split.seed}); "
        f"tasks: {len(tasks_by_split['train'])} train / {len(tasks_by_split['eval'])} eval",
        "",
        "| class | train | eval | total | in subset | calls (mean) | seq_len p50 | seq_len max | cot tokens p50 | answer tokens p50 | calls w/o </think> | calls with 2+ </think> |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in classes:
        members = [r for r in rows if r.cls == c]
        seq = sorted(r.seq_len for r in members)
        cot = sorted(r.n_cot_tokens for r in members)
        ans = sorted(r.n_answer_tokens for r in members)
        lines.append(
            f"| {c} | {counts[(c, 'train')]} | {counts[(c, 'eval')]} | {len(members)} | {sum(r.in_subset for r in members)} | "
            f"{sum(r.n_calls for r in members) / len(members):.2f} | {seq[len(seq) // 2]} | {seq[-1]} | {cot[len(cot) // 2]} | "
            f"{ans[len(ans) // 2]} | {sum(r.n_calls_without_think_close for r in members)} | {sum(r.n_calls_multi_think_close for r in members)} |"
        )
    lines += [
        "",
        f"Forward tokens: {tok_total / 1e6:.1f}M total; subset ({len(subset)} trajectories): {sum(r.seq_len for r in subset) / 1e6:.1f}M. "
        f"At 5k tok/s: {tok_total / 5e3 / 3600:.1f} h total, {sum(r.seq_len for r in subset) / 5e3 / 60:.0f} min for the subset.",
        f"Order: round-robin over {cfg.order.priority_classes} then the rest; subset_per_class={cfg.order.subset_per_class}; "
        f"the subset is complete after order index {max(r.order for r in subset) if subset else -1}.",
    ]
    if report.prefix_errors:
        lines += ["", "## Prefix property violations (excluded)"] + [f"- {e}" for e in report.prefix_errors[:50]]
    if report.layout_errors:
        lines += ["", "## Token layout errors (excluded)"] + [f"- {e}" for e in report.layout_errors[:50]]
    (out / "manifest_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/manifest.jsonl ({len(rows)} rows), splits.json, manifest_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
