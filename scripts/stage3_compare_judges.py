#!/usr/bin/env python
"""Stage 3: compare scripted judges (<pilot-dir>/judges/<model>/labels.jsonl) with the gold labels
(<pilot-dir>/labels.jsonl), at transcript level. The three statuses are never merged: a transcript's
label is its strongest flag status (enacted > attempted > considered > none), and agreement is
reported both on that four-way label and on "acted" (enacted or attempted) vs not.

  uv run scripts/stage3_compare_judges.py --pilot-dir data/stage3/pilot1 [--show-disagreements]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LEVELS = ("none", "considered", "attempted", "enacted")


def load(path: Path) -> dict[int, dict]:
    return {r["n"]: r for line in path.open() if (r := json.loads(line))}


def level(record: dict) -> str:
    return record["strongest_status"] or "none"


def acted(record: dict) -> bool:
    return level(record) in ("attempted", "enacted")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot-dir", type=Path, required=True)
    p.add_argument("--show-disagreements", action="store_true")
    args = p.parse_args()

    gold = load(args.pilot_dir / "labels.jsonl")
    groups = defaultdict(list)
    for n, r in sorted(gold.items()):
        groups[r["group"]].append(n)

    print(f"GOLD ({len(gold)} transcripts): strongest status per group")
    print(f"  {'group':<26}{'n':>3}" + "".join(f"{lv:>12}" for lv in LEVELS))
    for g, ns in groups.items():
        c = Counter(level(gold[n]) for n in ns)
        print(f"  {g:<26}{len(ns):>3}" + "".join(f"{c[lv]:>12}" for lv in LEVELS))

    for judge_dir in sorted((args.pilot_dir / "judges").glob("*")):
        if not (judge_dir / "labels.jsonl").exists():
            continue
        pred = load(judge_dir / "labels.jsonl")
        both = sorted(set(gold) & set(pred))
        failures = Counter()
        if (judge_dir / "failures.jsonl").exists():
            failures = Counter(json.loads(line)["n"] for line in (judge_dir / "failures.jsonl").open())
        never = sorted(n for n in failures if n not in pred)
        print(f"\nJUDGE {judge_dir.name}: {len(pred)} judged, {len(never)} never parsed {never or ''}")
        if not both:
            continue
        exact = sum(level(gold[n]) == level(pred[n]) for n in both)
        same_acted = sum(acted(gold[n]) == acted(pred[n]) for n in both)
        miss = [n for n in both if acted(gold[n]) and not acted(pred[n])]
        false_alarm = [n for n in both if not acted(gold[n]) and acted(pred[n])]
        print(f"  four-way status agreement: {exact}/{len(both)}; acted-vs-not agreement: {same_acted}/{len(both)}")
        print(f"  gold acted, judge did not: {miss}; judge acted, gold did not: {false_alarm}")
        print("  confusion (rows gold, columns judge):")
        print(f"  {'':<12}" + "".join(f"{lv:>12}" for lv in LEVELS))
        conf = Counter((level(gold[n]), level(pred[n])) for n in both)
        for g in LEVELS:
            print(f"  {g:<12}" + "".join(f"{conf[(g, j)]:>12}" for j in LEVELS))
        kept = sum(len(pred[n]["flags"]) for n in both)
        dropped = sum(len(pred[n].get("discarded_flags", [])) for n in both)
        print(f"  flags kept {kept}, discarded by quote/field check {dropped} ({dropped / max(kept + dropped, 1):.0%})")
        cost = [pred[n]["cost_usd"] for n in both if pred[n].get("cost_usd") is not None]
        wall = [pred[n]["wall_s"] for n in both if pred[n].get("wall_s") is not None]
        tok_in = [sum(v for k, v in (pred[n].get("usage") or {}).items() if k.endswith("input_tokens") and isinstance(v, int)) for n in both]
        tok_out = [(pred[n].get("usage") or {}).get("output_tokens", 0) for n in both]
        print(f"  per transcript: input tokens mean {sum(tok_in) / len(both):,.0f} (max {max(tok_in):,}), output mean {sum(tok_out) / len(both):,.0f}, "
              f"wall mean {sum(wall) / max(len(wall), 1):.0f}s; API-equivalent cost total ${sum(cost):.2f} (mean ${sum(cost) / max(len(cost), 1):.3f})")
        if args.show_disagreements:
            for n in both:
                if level(gold[n]) != level(pred[n]):
                    print(f"\n  n={n} [{gold[n]['group']}] gold={level(gold[n])} judge={level(pred[n])}")
                    for f in pred[n]["flags"]:
                        print(f"     judge: step {f['step']} {f['channel']} {f['category']} {f['status']}: {f['quote'][:110]!r}")
                    for f in gold[n]["flags"]:
                        print(f"     gold:  step {f['step']} {f['channel']} {f['category']} {f['status']}: {f['quote'][:110]!r}")
                    print(f"     judge rationale: {pred[n]['rationale'][:400]}")


if __name__ == "__main__":
    main()
