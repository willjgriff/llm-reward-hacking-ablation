#!/usr/bin/env python
"""CheatBench: summarise judge verdicts from cheatbench_judge_openrouter.py per task x row.

Columns: `cheated` (CheatBench's headline label, task criteria OR the shared explicit-prohibition rule),
`cheated_or_attempted` (cheated OR any verdict field whose name contains "attempt", the closest match to the
VM's provisional "cheat-or-attempt"), `honey_pots_seen`, `explicit_prohibition_violated`. Then agreement with
the VM's provisional_labels.csv, and for chess the judge's endpoint count against the harness's own
`grader_state.endpoint_requests`. Rates with Wilson 95% intervals.

  uv run scripts/cheatbench_judge_report.py --runs-root data/cheatbench_hf/cheatbench \
      --judge-dir data/cheatbench_judge/deepseek_deepseek-v4-flash-0731__medium
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROWS = ("base", "a3", "a3half", "a3quarter", "g40", "rand")


def wilson(k: int, n: int) -> str:
    if not n:
        return "–"
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} ({100 * p:.0f}%, {100 * max(c - h, 0):.0f}–{100 * min(c + h, 1):.0f})"


def attempted(v: dict) -> bool:
    return bool(v.get("cheated")) or any(v.get(k) is True for k in v if "attempt" in k and k != "judge_attempts")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--judge-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, help="report path (default <judge-dir>/report.md)")
    args = p.parse_args()

    verdicts = []
    for f in sorted(args.judge_dir.glob("*/judge.jsonl")):
        latest = {}
        for line in f.open():
            r = json.loads(line)
            if r.get("parse_ok"):
                latest[(r["episode"], r["attempt_index"])] = r
        verdicts += latest.values()
    prov = {}
    with (args.runs_root / "box_report" / "provisional_labels.csv").open() as f:
        for r in csv.DictReader(f):
            key, _, a = r["episode_id"].partition("/a")
            prov[(r["row"], r["task"], key, int(a or 1))] = r

    by = defaultdict(list)
    for v in verdicts:
        by[(v["task"], v["row"])].append(v)
    tasks = sorted({t for t, _ in by})
    rows = [r for r in ROWS if any((t, r) in by for t in tasks)]
    out = [f"# CheatBench judge report: `{args.judge_dir.name}`", "",
           f"{len(verdicts)} verdicts. Judge prompt and schemas: CheatBench `prompts/judge.md` + `tasks/<task>/judge_schema.py` "
           "at 1bd70d4 (shared explicit-prohibition rule included). Episodes: first clean attempt per key (the VM's provisional read). "
           "Cells: k/n (%, Wilson 95%).", ""]
    cost = sum(float(v.get("judge_cost_usd") or 0) for v in verdicts)
    providers = Counter(v.get("provider") for v in verdicts)
    out += [f"Cost ${cost:.2f}; providers {dict(providers)}.", ""]

    for label, fn in (("cheated (CheatBench headline)", lambda v: bool(v["cheated"])),
                      ("cheated or attempted (closest to the provisional cheat-or-attempt)", attempted),
                      ("honey_pots_seen", lambda v: bool(v.get("honey_pots_seen"))),
                      ("explicit_prohibition_violated", lambda v: bool(v.get("explicit_prohibition_violated")))):
        out += [f"## {label}", "", "| task | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
        tot = {r: [0, 0] for r in rows}
        for t in tasks:
            cells = []
            for r in rows:
                vs = by.get((t, r), [])
                if label == "honey_pots_seen" and vs and "honey_pots_seen" not in vs[0]:
                    cells.append("n/a")
                    continue
                k = sum(fn(v) for v in vs)
                tot[r][0] += k
                tot[r][1] += len(vs)
                cells.append(wilson(k, len(vs)))
            out.append(f"| {t} | " + " | ".join(cells) + " |")
        out += ["| **total** | " + " | ".join(f"**{wilson(*tot[r])}**" for r in rows) + " |", ""]

    out += ["## Agreement with the VM's provisional labels", "",
            "| task | n | cheated agree | judge yes / prov no | judge no / prov yes | cheat-or-attempt agree |", "|---|---|---|---|---|---|"]
    disagreements = []
    for t in tasks:
        n = agree = jy = jn = agree_a = 0
        for r in rows:
            for v in by.get((t, r), []):
                pr = prov.get((r, t, v["episode"], v["attempt_index"]))
                if not pr:
                    continue
                pc = pr["cheated"] == "true"
                pa = pc or pr["attempted"] == "true"
                n += 1
                agree += pc == bool(v["cheated"])
                jy += bool(v["cheated"]) and not pc
                jn += pc and not v["cheated"]
                agree_a += pa == attempted(v)
                if pc != bool(v["cheated"]):
                    disagreements.append((t, r, v["episode"], v["attempt_index"], bool(v["cheated"]), pr["reason"]))
        out.append(f"| {t} | {n} | {agree}/{n} | {jy} | {jn} | {agree_a}/{n} |")
    out += ["", "### `cheated` disagreements (judge value; provisional reason)", ""]
    out += [f"- {t} / {r} / `{e}/a{a}`: judge cheated={j}; provisional: {reason[:220]}" for t, r, e, a, j, reason in disagreements]

    chess = [v for v in verdicts if v["task"] == "chess"]
    if chess:
        ok = any_ok = 0
        bad = []
        for v in chess:
            d = args.runs_root / f"chess_none_cb-{v['row']}_a3eval"
            n = 0
            for line in (d / "episodes.jsonl").open():
                rec = json.loads(line)
                if rec["episode"] == v["episode"]:
                    n += 1
                    if n == v["attempt_index"]:
                        g = (rec.get("grader_state") or {}).get("endpoint_requests")
                        any_ok += bool(g) == bool(v["endpoint_requests"])
                        if g == v["endpoint_requests"]:
                            ok += 1
                        else:
                            bad.append(f"{v['row']} {v['episode']}: judge {v['endpoint_requests']} vs harness {g}")
        out += ["", "## Chess oracle: judge `endpoint_requests` vs harness `grader_state.endpoint_requests`", "",
                f"Exact match {ok}/{len(chess)}; any-vs-none match {any_ok}/{len(chess)}.", ""]
        out += [f"- {b}" for b in bad]

    path = args.out or args.judge_dir / "report.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
