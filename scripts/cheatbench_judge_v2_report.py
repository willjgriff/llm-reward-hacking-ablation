#!/usr/bin/env python
"""CheatBench judge v2 report: engagement, persistence and report accuracy per row, next to v1's cheating label.

Reads cheatbench_judge_v2.py verdicts (`<judge-dir>/<run>/judge.jsonl`, which also carry the objective
`metrics`) and v1's `cheated` (cheatbench_judge_openrouter.py). Sections: per-row distributions of every v2
field; the same among episodes v1 judged NOT cheated (did a row stop cheating by honest work or by doing
little?); per task x row genuine-attempt rate, mean persistence and objective metrics; quote verification;
paired tests (results_significance.paired_test: exact McNemar per episode, exact task-cluster sign-flip;
Holm over all tests here) for each steered row vs base and vs random.

  uv run scripts/cheatbench_judge_v2_report.py \
      --judge-dir data/cheatbench_judge_v2/deepseek_deepseek-v4-flash-0731__medium \
      --v1-judge-dir data/cheatbench_judge/deepseek_deepseek-v4-flash-0731__medium
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_significance import fmt_p, holm, paired_test  # noqa: E402

ROWS = ("base", "a3", "a3half", "a3quarter", "g40", "rand")
STEERED = ("a3", "a3half", "a3quarter", "g40")
FIELDS = {
    "ending": ("succeeded", "honest_best_effort", "stated_decline", "quiet_stop", "degenerate", "cut_off"),
    "effort_direction": ("own_work", "mixed", "shortcut", "none"),
    "shortcut_awareness": ("not_noticed", "noticed_not_considered", "considered_declined", "pursued"),
    "report_accuracy": ("accurate", "overclaims", "underclaims_or_contradicts", "no_report"),
    "decline_reason": ("infeasible", "rules", "integrity", "none_given", "other"),
}
TASK_METRICS = {"chess": ("resets", "games_played", "won"), "geoguessr": ("trials", "distance_km"),
                "knowledge_work": ("review_calls",), "prime_factorization": ("verifier_calls", "solved"),
                "subset_sum": ("verifier_calls", "solved"), "protein_design": ("check_calls",),
                "svgs": ("submissions", "drawing_bytes"), "synthetic_puzzles": ("checker_runs", "rel_error"),
                "openmath_agent": ()}


def load(judge_dir: Path) -> list[dict]:
    out = {}
    for f in sorted(judge_dir.glob("*/judge.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r.get("parse_ok"):
                out[(r["row"], r["task"], r["episode"], r["attempt_index"])] = r
    return list(out.values())


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%" if n else "–"


def med(xs) -> str:
    xs = [float(x) for x in xs if x is not None]
    return f"{statistics.median(xs):.3g}" if xs else "–"


def dist_table(rows_data: dict[str, list[dict]], title: str) -> list[str]:
    out = [f"## {title}", ""]
    present = [r for r in ROWS if rows_data.get(r)]
    out += ["| field | value | " + " | ".join(f"{r} (n={len(rows_data[r])})" for r in present) + " |",
            "|---|---|" + "---|" * len(present)]
    for field, values in FIELDS.items():
        for v in values:
            cells = []
            for r in present:
                vs = rows_data[r] if field != "decline_reason" else [x for x in rows_data[r] if x["ending"] in ("stated_decline", "quiet_stop")]
                cells.append(pct(sum(x[field] == v for x in vs), len(vs)))
            out.append(f"| {field} | {v} | " + " | ".join(cells) + " |")
    for label, fn in (("genuine_attempt", lambda x: x["genuine_attempt"] is True),
                      ("stopped_with_options_left", lambda x: x["stopped_with_options_left"] is True),
                      ("persistence ≥ 2", lambda x: x["persistence"] >= 2)):
        out.append(f"| {label} | true | " + " | ".join(pct(sum(map(fn, rows_data[r])), len(rows_data[r])) for r in present) + " |")
    out.append("| persistence | mean | " + " | ".join(f"{statistics.mean(x['persistence'] for x in rows_data[r]):.2f}" for r in present) + " |")
    return out + [""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-dir", type=Path, required=True)
    p.add_argument("--v1-judge-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, help="default <judge-dir>/report.md")
    args = p.parse_args()

    v2 = load(args.judge_dir)
    v1 = {(r["row"], r["task"], r["episode"], r["attempt_index"]): bool(r["cheated"]) for r in load(args.v1_judge_dir)}
    for r in v2:
        r["v1_cheated"] = v1.get((r["row"], r["task"], r["episode"], r["attempt_index"]))
    by_row = defaultdict(list)
    for r in v2:
        by_row[r["row"]].append(r)
    cost = sum(float(r.get("judge_cost_usd") or 0) for r in v2)
    out = [f"# CheatBench judge v2 (engagement): `{args.judge_dir.name}`", "",
           f"{len(v2)} verdicts, cost ${cost:.2f}, providers {dict(Counter(r.get('provider') for r in v2))}. "
           f"Rubric `configs/judge/cheatbench_v2_rubric.md`; v1 `cheated` joined for {sum(r['v1_cheated'] is not None for r in v2)}.", ""]
    out += dist_table(by_row, "All episodes")
    out += dist_table({k: [x for x in v if x["v1_cheated"] is False] for k, v in by_row.items()},
                      "Episodes v1 judged NOT cheated (how did the non-cheating episodes go?)")

    out += ["## Per task: genuine attempt / mean persistence / stopped with options left", "",
            "| task | " + " | ".join(ROWS) + " |", "|---|" + "---|" * len(ROWS)]
    tasks = sorted({r["task"] for r in v2})
    for t in tasks:
        cells = []
        for row in ROWS:
            xs = [x for x in by_row[row] if x["task"] == t]
            cells.append(f"{pct(sum(x['genuine_attempt'] is True for x in xs), len(xs))} / "
                         f"{statistics.mean(x['persistence'] for x in xs):.1f} / "
                         f"{pct(sum(x['stopped_with_options_left'] is True for x in xs), len(xs))}" if xs else "–")
        out.append(f"| {t} | " + " | ".join(cells) + " |")

    out += ["", "## Objective metrics per task (median; booleans as rate), from the records", "",
            "| task | metric | " + " | ".join(ROWS) + " |", "|---|---|" + "---|" * len(ROWS)]
    for t in tasks:
        for mname in ("n_turns", "time_share") + TASK_METRICS.get(t, ()):
            cells = []
            for row in ROWS:
                vals = [x["metrics"].get(mname) for x in by_row[row] if x["task"] == t]
                if vals and all(isinstance(v, bool) or v is None for v in vals):
                    cells.append(pct(sum(v is True for v in vals), len(vals)))
                else:
                    cells.append(med(vals))
            out.append(f"| {t} | {mname} | " + " | ".join(cells) + " |")
        ends = {row: Counter(x["metrics"]["end"] for x in by_row[row] if x["task"] == t) for row in ROWS}
        out.append(f"| {t} | end (mark_complete/time/turns) | " + " | ".join(
            f"{e['mark_task_complete']}/{e['time_limit']}/{e['turn_cap']}" for e in ends.values()) + " |")

    qc = Counter(v for r in v2 for v in (r.get("quote_checks") or {}).values() if v)
    tot = sum(qc.values())
    out += ["", "## Quote verification", "",
            f"{tot} quotes: " + ", ".join(f"{k} {n} ({pct(n, tot)})" for k, n in qc.most_common()), ""]

    outcomes = {"genuine_attempt": lambda x: int(x["genuine_attempt"] is True),
                "persistence>=2": lambda x: int(x["persistence"] >= 2),
                "stopped_with_options_left": lambda x: int(x["stopped_with_options_left"] is True),
                "ending=stated_decline|quiet_stop": lambda x: int(x["ending"] in ("stated_decline", "quiet_stop")),
                "own_work": lambda x: int(x["effort_direction"] == "own_work")}
    eps = {row: {(x["task"], x["episode"]): x for x in sorted(xs, key=lambda x: -x["attempt_index"])} for row, xs in by_row.items()}
    tests = []
    for row, ref in [(r, "base") for r in (*STEERED, "rand")] + [(r, "rand") for r in STEERED]:
        for name, fn in outcomes.items():
            a = {(f"{t}/{k}", "0"): fn(x) for (t, k), x in eps.get(row, {}).items()}
            b = {(f"{t}/{k}", "0"): fn(x) for (t, k), x in eps.get(ref, {}).items()}
            ta = {(t, k): fn(x) for (t, k), x in eps.get(row, {}).items()}
            tb = {(t, k): fn(x) for (t, k), x in eps.get(ref, {}).items()}
            e, k = paired_test(a, b), paired_test(ta, tb)
            if e and k:  # empty when the two rows share no episodes
                tests.append({"row": row, "ref": ref, "outcome": name, "episode": e, "task": k})
    for kind in ("episode", "task"):
        for t, h in zip(tests, holm([t[kind]["p"] for t in tests])):
            t[kind]["holm"] = h
    out += ["## Paired tests (same task + episode key in both rows)", "",
            f"{len(tests)} tests; Holm over all of them. McNemar = exact, per episode; sign-flip = exact, tasks as clusters "
            "(9 tasks, smallest possible p 0.0039).", "",
            "| row | vs | outcome | row | ref | diff (95% CI) | p McNemar | Holm | p sign-flip | Holm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tests:
        e, k = t["episode"], t["task"]
        out.append(f"| {t['row']} | {t['ref']} | {t['outcome']} | {e['row'][0]}/{e['row'][1]} | {e['base'][0]}/{e['base'][1]} | "
                   f"{100 * e['diff']:+.0f} ({100 * k['ci95'][0]:+.0f} to {100 * k['ci95'][1]:+.0f}) | {fmt_p(e['p'])} | "
                   f"{fmt_p(e['holm'])} | {fmt_p(k['p'])} | {fmt_p(k['holm'])} |")

    path = args.out or args.judge_dir / "report.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
