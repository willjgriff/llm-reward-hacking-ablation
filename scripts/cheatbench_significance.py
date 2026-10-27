#!/usr/bin/env python
"""CheatBench: paired p-values for the A3 eval rows (cheated, gave up / degenerate, neither).

`cheated` is the judge's label (cheatbench_judge_openrouter.py); `gave_up_or_degenerate` is the VM's
provisional label (box_report/provisional_labels.csv); `neither` = not cheated and not gave up. Episodes are
paired across rows by (task, key), first clean attempt per key (lowest attempt index in the CSV, so base's
two later information-only attempts drop out). Tests from results_significance.paired_test: exact McNemar
with each episode as its own pair, and the exact task-cluster sign-flip with tasks as clusters (9 tasks, so
its smallest possible p is 2/2^9); cluster bootstrap CI of the difference; Holm and BH over every test here.

  uv run scripts/cheatbench_significance.py --runs-root data/cheatbench_hf/cheatbench \
      --judge-dir data/cheatbench_judge/deepseek_deepseek-v4-flash-0731__medium
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_significance import bh, fmt_p, holm, paired_test  # noqa: E402

STEERED = ("a3", "a3half", "a3quarter", "g40")
NAMES = {"base": "base", "a3": "A3", "a3half": "half", "a3quarter": "quarter", "g40": "g40", "rand": "random"}
OUTCOMES = ("cheated", "gave_up", "neither")


def load(runs_root: Path, judge_dir: Path) -> dict[str, dict[tuple[str, str], dict[str, int]]]:
    """row -> {(task, key): {outcome: 0/1}}."""
    judged = {}
    for f in judge_dir.glob("*/judge.jsonl"):
        for line in f.open():
            r = json.loads(line)
            if r.get("parse_ok"):
                judged[(r["row"], r["task"], r["episode"], r["attempt_index"])] = bool(r["cheated"])
    first: dict[tuple[str, str, str], tuple[int, dict]] = {}
    with (runs_root / "box_report" / "provisional_labels.csv").open() as f:
        for r in csv.DictReader(f):
            if r["task"] == "sycophancy":
                continue
            key, _, a = r["episode_id"].partition("/a")
            a = int(a or 1)
            k = (r["row"], r["task"], key)
            if k not in first or a < first[k][0]:
                first[k] = (a, r)
    out: dict = {}
    for (row, task, key), (a, r) in first.items():
        c = judged[(row, task, key, a)]
        g = r["gave_up_or_degenerate"].strip().lower().startswith("y")
        out.setdefault(row, {})[(task, key)] = {"cheated": int(c), "gave_up": int(g), "neither": int(not c and not g)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--judge-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default <judge-dir>/significance.md")
    args = p.parse_args()
    eps = load(args.runs_root, args.judge_dir)

    pairs = [(r, "base") for r in (*STEERED, "rand")] + [(r, "rand") for r in STEERED]
    tests = []
    for row, ref in pairs:
        for o in OUTCOMES:
            per_ep = paired_test({(f"{t}/{k}", "0"): v[o] for (t, k), v in eps[row].items()},
                                 {(f"{t}/{k}", "0"): v[o] for (t, k), v in eps[ref].items()})
            per_task = paired_test({(t, k): v[o] for (t, k), v in eps[row].items()},
                                   {(t, k): v[o] for (t, k), v in eps[ref].items()})
            tests.append({"row": row, "ref": ref, "outcome": o, "episode": per_ep, "task": per_task})
    for kind in ("episode", "task"):
        ps = [t[kind]["p"] for t in tests]
        for t, h, b in zip(tests, holm(ps), bh(ps)):
            t[kind]["holm"], t[kind]["bh"] = h, b

    lines = ["# CheatBench A3 eval: paired p-values", "",
             f"{len(tests)} tests per method (Holm/BH over all of them). Pairs = same (task, key) in both rows. "
             "`cheated` = judge; `gave up` = VM provisional; `neither` = not cheated and not gave up. "
             "McNemar = exact, episode-level (sensitive, ignores task clustering); sign-flip = exact, task-level "
             "(9 tasks, smallest possible p = 0.0039; robust). CI = task-cluster bootstrap of the difference.", "",
             "| row | vs | outcome | row | ref | diff (95% CI) | p McNemar | Holm | p sign-flip | Holm |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        e, k = t["episode"], t["task"]
        lines.append(f"| {NAMES[t['row']]} | {NAMES[t['ref']]} | {t['outcome']} | {e['row'][0]}/{e['row'][1]} | "
                     f"{e['base'][0]}/{e['base'][1]} | {100 * e['diff']:+.0f} ({100 * k['ci95'][0]:+.0f} to {100 * k['ci95'][1]:+.0f}) | "
                     f"{fmt_p(e['p'])} | {fmt_p(e['holm'])} | {fmt_p(k['p'])} | {fmt_p(k['holm'])} |")
    path = args.out or args.judge_dir / "significance.md"
    path.write_text("\n".join(lines) + "\n")
    (path.with_suffix(".json")).write_text(json.dumps(tests, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
