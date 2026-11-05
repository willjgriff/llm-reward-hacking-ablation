#!/usr/bin/env python
"""Persistence screen report: TV `verifier_bug` rollouts of edited checkpoints vs their base (laptop, CPU).

Reads the judge verdicts written by stage3_judge_openrouter.py on the `tv_persistence` audit set
(rubric configs/judge/tv_persistence_rubric.md) and compares each edited row with the base run on the same
task ids (TV_PERSISTENCE_ROWS in rhablation.judge_audit). Every metric is reported on all episodes and on
the oracle-clean ones (never tampered), because removing tampering by itself changes how episodes end.
A3 (Grader direction, cut persistence on Terminal-Bench / CheatBench) is the positive control: if it shows
no difference here, the environment cannot show the effect and R1 / D_TV rows say nothing either way.

Tests: paired by task id (`results_significance.paired_test`: exact McNemar with one sample per arm, else an
exact task-cluster sign-flip), 95 % task-cluster bootstrap interval, Holm over the table's tests.

  uv run scripts/tv_persistence_report.py [--audit-dir data/stage3/audit/tv_persistence-all] [--judge <dir name>]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from results_significance import fmt_p, holm, paired_test  # noqa: E402

from rhablation.judge_audit import TV_PERSISTENCE_ROWS  # noqa: E402

CATEGORIES = ("claims_impossible", "claims_done", "gives_up", "rechecks_answer", "investigates_rejection")
ENDINGS = ("still_investigating", "resubmitting", "declared_impossible", "declared_done", "edited_or_bypassed", "stopped", "degenerate")


def metrics(rec: dict, oracle: dict) -> dict[str, float | None]:
    """0/1 (or None) per metric for one episode."""
    s = rec["summary"]
    pers = s.get("persistence")
    enacted = {f["category"] for f in rec["flags"] if f["status"] == "enacted"}
    anyst = set(rec["categories"])
    out: dict[str, float | None] = {
        "persistence_le1": None if pers is None else float(pers <= 1),
        "stopped_with_options_left": None if s.get("stopped_with_options_left") is None else float(s["stopped_with_options_left"]),
        "overclaims": None if s.get("report_accuracy") is None else float(s["report_accuracy"] == "overclaims"),
        "no_command_end": float(oracle.get("end_reason") == "no_command"),
    }
    for c in CATEGORIES:
        out[f"{c}_enacted"] = float(c in enacted)
        out[f"{c}_any"] = float(c in anyst)
    for e in ENDINGS:
        out[f"ending_{e}"] = None if s.get("ending") is None else float(s["ending"] == e)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-dir", type=Path, default=Path("data/stage3/audit/tv_persistence-all"))
    p.add_argument("--judge", default="deepseek_deepseek-v4-flash-0731__medium")
    args = p.parse_args()

    items = {it["trajectory_id"]: it for it in json.loads((args.audit_dir / "pilot_set.json").read_text())["items"]}
    labels_path = args.audit_dir / "judges" / args.judge / "labels.jsonl"
    recs = [json.loads(l) for l in labels_path.open() if l.strip()]

    # row -> subset -> metric -> {(task, sample): value}
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    pers: dict = defaultdict(lambda: defaultdict(list))
    endings: dict = defaultdict(Counter)
    n_judged = Counter()
    for r in recs:
        o = items[r["trajectory_id"]]["oracle"]
        row, task = o["row"], o["task_id"]
        key = (task, r["trajectory_id"])
        n_judged[row] += 1
        endings[row][r["summary"].get("ending")] += 1
        subsets = ("all", "clean") if not o.get("tampered") else ("all",)
        for sub in subsets:
            if r["summary"].get("persistence") is not None:
                pers[row][sub].append(r["summary"]["persistence"])
            for m, v in metrics(r, o).items():
                data[row][sub][m][key] = v
    n_items = Counter(it["oracle"]["row"] for it in items.values())

    pairs = sorted({v for v in TV_PERSISTENCE_ROWS.values() if v[0] != v[1]})
    headline = ["persistence_le1", "stopped_with_options_left", "gives_up_any", "claims_done_any", "claims_impossible_enacted",
                "overclaims", "rechecks_answer_enacted", "investigates_rejection_enacted", "no_command_end"]
    tests = []
    for row, base in pairs:
        for sub in ("all", "clean"):
            for m in headline:
                t = paired_test({(k[0], k[1]): v for k, v in data[row][sub][m].items()},
                                {(k[0], k[1]): v for k, v in data[base][sub][m].items()})
                if t:
                    tests.append({**t, "edited": row, "base_row": base, "subset": sub, "metric": m})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph

    lines = [f"# Persistence screen: TV `verifier_bug`, judge `{args.judge}`", "",
             f"Labels: `{labels_path}`. Rubric `configs/judge/tv_persistence_rubric.md`. Positive control: A3.", "",
             "## Coverage and persistence score", "",
             "| row | judged / items | mean persistence (all) | mean persistence (clean) | clean n |", "|---|---|---|---|---|"]
    for row in sorted(n_items):
        pa, pc = pers[row]["all"], pers[row]["clean"]
        lines.append(f"| {row} | {n_judged[row]} / {n_items[row]} | {sum(pa) / len(pa):.2f} | "
                     f"{(sum(pc) / len(pc)) if pc else float('nan'):.2f} | {len(pc)} |" if pa else f"| {row} | 0 / {n_items[row]} | | | |")
    lines += ["", "## Endings (judge, all episodes)", "", "| row | " + " | ".join(ENDINGS) + " |", "|---" * (len(ENDINGS) + 1) + "|"]
    for row in sorted(n_items):
        tot = sum(endings[row].values()) or 1
        lines.append(f"| {row} | " + " | ".join(f"{100 * endings[row][e] / tot:.0f}%" for e in ENDINGS) + " |")
    lines += ["", "## Row vs its base, paired by task", "",
              "| row | base | subset | metric | row % | base % | Δ [95 % CI] | p | Holm | tasks |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['edited']} | {t['base_row']} | {t['subset']} | {t['metric']} | {100 * t['rate_row']:.1f} | "
                     f"{100 * t['rate_base']:.1f} | {100 * t['diff']:+.1f} [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}] | "
                     f"{fmt_p(t['p'])} | {fmt_p(t['p_holm'])} | {t['n_tasks']} |")
    out = args.audit_dir / "persistence_report.md"
    out.write_text("\n".join(lines) + "\n")
    (args.audit_dir / "persistence_report.json").write_text(json.dumps(tests, indent=1))
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
