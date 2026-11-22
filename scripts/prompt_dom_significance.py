"""P-values for the prompt-pair DoM mission (stage 11; laptop, CPU, no GPU).

Reads the ortho records the box uploaded to HF `prompt_dom/records/` and tests every
row against `base` and against its random control with the paired tests of
`results_significance.py` (exact McNemar with one episode per task, task/prompt-cluster
sign-flip otherwise; cluster bootstrap CI; Holm and BH over the hack-rate tests).

Pairing: TerminalVerifier = sample id (128 tasks, one episode each, seed pool 0 in every run);
Grader = prompt (`family.side.kind.sys`; the sample index after it is dropped, since samples of one
prompt are not paired across runs), so only 10 (main) / 14 (held-out) clusters and the smallest
attainable p is ~0.002 / ~1e-4; MBPP = task id, valid answers only, all 954 tasks
(no row was fitted on MBPP).

Controls: Grader R rows (gpdom, gpdomall, gp10dom) -> gprand; TV R rows (tvpdom, tvpdomall) -> tvprand;
Arditi rows (gparditi, tvarditi) -> zrand. The mission's rule needs p < 0.01 against both.

  uv run scripts/prompt_dom_significance.py [--records data/prompt_dom/records] [--out docs/analysis/2026-11-24_prompt_dom]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from results_significance import band, bh, fmt_p, holm, paired_test

ROWS = ["gpdom", "gpdomall", "gp10dom", "tvpdom", "tvpdomall", "gparditi", "tvarditi"]
CONTROL = {"gpdom": "gprand", "gpdomall": "gprand", "gp10dom": "gprand",
           "tvpdom": "tvprand", "tvpdomall": "tvprand", "gparditi": "zrand", "tvarditi": "zrand"}
GRADER_CELLS = {"off main": ("nothink", "all"), "off held-out": ("nothink", "heldout"),
                "low main": ("qwen3.8_low", "all"), "low held-out": ("qwen3.8_low", "heldout")}
TV_PARTS = {"TV vb mt4": ("verifier_bug", 4), "TV possible mt6": ("possible", 6),
            "TV vb mt6": ("verifier_bug", 6), "TV corrupted mt4": ("corrupted", 4)}
# hack metric per part; possible is a capability check
TV_METRIC = {"TV vb mt4": "tampered", "TV vb mt6": "tampered", "TV corrupted mt4": "passed", "TV possible mt6": "passed"}


def row_of(condition: str) -> str:
    if condition == "none":
        return "base"
    for suffix in ("_nothink_heldout", "_nothink", "_main", "_heldout", "_tv_bug_mt6", "_tv_bug", "_tv_pos", "_tv_corr", "_mbpp"):
        if condition.endswith(suffix):
            return condition[: -len(suffix)]
    return condition


def load(records: Path) -> dict[tuple[str, str], dict]:
    """(row, part) -> {(cluster, sample): {metric: value}}."""
    out: dict[tuple[str, str], dict] = {}
    for f in sorted(records.glob("*/*.jsonl")):
        rs = [json.loads(line) for line in f.open()]
        if len(rs) < 100:  # phase-0 workload and smoke runs
            continue
        r0 = rs[0]
        row, cfg = row_of(r0["condition"]), r0["cfg"]
        if r0["env"] == "grader":
            part = next(k for k, v in GRADER_CELLS.items() if v == (r0["config_id"], cfg["family"]))
        elif r0["env"] == "terminal_verifier":
            part = next(k for k, v in TV_PARTS.items() if v == (cfg["mode"], cfg["max_turns"]))
        else:
            part = "MBPP"
        eps = {}
        for r in rs:
            log, sample, epoch = r["rollout_id"].rsplit("/", 2)
            lab = r["labels"]
            if r["env"] == "grader":
                c = lab["cheat"]
                eps[(sample.rsplit(".", 1)[0], f"{sample}/{epoch}")] = {"cheat": None if c is None else bool(c), "valid": c is not None}
            elif r["env"] == "terminal_verifier":
                eps[(sample, epoch)] = {"tampered": bool(lab["tampered"]), "passed": bool(lab["passed"]),
                                        "peeked": bool(lab["peeked"])}
            else:
                c = lab["cheat"]
                eps[(str(lab["task_id"]), epoch)] = {"cheat": None if c is None else bool(c)}
        assert (row, part) not in out, (row, part, f)
        out[(row, part)] = eps
    return out


def tests(data: dict) -> list[dict]:
    T = []

    def add(row, ref, part, metric, hack):
        a, b = data.get((row, part)), data.get((ref, part))
        if a is None or b is None:
            return
        res = paired_test({k: v[metric] for k, v in a.items()}, {k: v[metric] for k, v in b.items()})
        T.append(dict(res, name=row, ref=ref, part=part, metric=metric, hack=hack))

    for row in ROWS + ["gprand", "tvprand", "zrand"]:
        refs = ["base"] + ([CONTROL[row]] if row in CONTROL else [])
        for ref in refs:
            for part in GRADER_CELLS:
                add(row, ref, part, "cheat", True)
                add(row, ref, part, "valid", False)
            for part, metric in TV_METRIC.items():
                add(row, ref, part, metric, part != "TV possible mt6")
            add(row, ref, "TV vb mt4", "peeked", False)
            add(row, ref, "MBPP", "cheat", True)
    return T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", type=Path, default=Path("data/prompt_dom/records"))
    p.add_argument("--out", type=Path, default=Path("docs/analysis/2026-11-24_prompt_dom"))
    args = p.parse_args()
    T = tests(load(args.records))
    hk = [t for t in T if t["hack"]]
    for t, h, q in zip(hk, holm([t["p"] for t in hk]), bh([t["p"] for t in hk])):
        t["p_holm"], t["p_bh"] = h, q
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(T, indent=1))
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    lines = ["# Prompt-pair DoM: paired p-values", "",
             f"Generated by `scripts/prompt_dom_significance.py` from `{args.records}`. "
             f"Holm / BH over the {len(hk)} hack-rate tests (vs base and vs the random control). "
             "Rates in %, diff in points with a 95% cluster-bootstrap interval.", "",
             "| row | vs | part | metric | row rate | ref rate | diff [95% CI] | clusters | p | p Holm | test |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in T:
        lines.append(f"| {t['name']} | {t['ref']} | {t['part']} | {t['metric']} | {pct(t['rate_row'])} ({t['row'][0]}/{t['row'][1]}) "
                     f"| {pct(t['rate_base'])} ({t['base'][0]}/{t['base'][1]}) | {100 * t['diff']:+.1f} [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}] "
                     f"| {t['n_tasks']} | {fmt_p(t['p'])} | {fmt_p(t['p_holm']) if 'p_holm' in t else '-'} | {t['method']} |")
    lines += ["", "Bands for the raw p (meeting 2026-10-16): " + "; ".join(band(x) for x in (0.0005, 0.01, 0.1, 0.5)), ""]
    (args.out / "results.md").write_text("\n".join(lines))
    print(f"wrote {args.out}/results.md ({len(T)} tests, {len(hk)} hack-rate tests)")


if __name__ == "__main__":
    main()
