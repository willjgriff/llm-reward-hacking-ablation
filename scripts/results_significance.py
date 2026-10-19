"""P-values for every behavioural row of stages 5b-8 (laptop, CPU, no GPU).

Recomputes each row-vs-base comparison from per-episode outcomes in
`data/stage6/` (rollouts pulled from HF `stage6/`) and from the per-cell /
per-task files of the hook and steering runs in `data/stage7/stage7/`.

Tests (two-sided, H0 = the intervention does not change the rate):
  * primary, paired by task: the two models ran on the same task ids, so under
    H0 the row's and the base's outcomes on a task are exchangeable. Per task
    (cluster) d = mean(row) - mean(base) over its samples; p from a sign-flip
    test of the d's, computed exactly: the McNemar binomial with one sample per
    arm, otherwise a dynamic-programming convolution over the integer grid the
    d's live on (Monte Carlo only if that grid is not found). Valid with several
    samples per task and any within-task correlation; tasks must be
    independent draws.
  * unpaired check: Fisher exact on pooled counts (assumes independent
    samples, so it ignores task clustering; reported for reference only).
  * stratified unpaired (hook / steering runs that only stored per-cell
    counts): exact, the row's total under H0 is a sum of independent per-cell
    hypergeometrics.
Effect: difference of rates with a 95 % task-cluster bootstrap interval.
Multiple comparisons: Holm (family-wise) and Benjamini-Hochberg (FDR) over
every primary hack-rate test in the table.
Power: tasks needed per arm for 80 % power at alpha 0.05 two-sided to detect
the observed difference (two-proportion normal approximation, unpaired, with a
design effect for the samples per task; pairing only helps, so it is an upper
bound).

  uv run scripts/results_significance.py [--data-root data] [--out data/analysis/significance]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

RNG = np.random.default_rng(0)
N_PERM = 200_000
N_BOOT = 20_000


# ---------------------------------------------------------------- loading


def _read(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


@dataclass
class Episodes:
    """Outcome per episode, keyed by (cluster, sample) for pairing."""

    run: str
    rows: dict[tuple[str, str], dict] = field(default_factory=dict)

    def metric(self, name: str) -> dict[tuple[str, str], float | None]:
        return {k: v.get(name) for k, v in self.rows.items()}


def load_rhenvs(run_dir: Path) -> Episodes:
    ep = Episodes(str(run_dir))
    for r in _read(run_dir / "records.jsonl.gz"):
        lab = r["labels"]
        sample, epoch = r["rollout_id"].split("/")[1:3]
        env = r["env"]
        if env == "terminal_verifier":
            o = {
                "tampered": bool(lab["tampered"]),
                "passed": bool(lab["passed"]),
                "cheated": bool(r["cheated"]),
                "truncated": bool(lab.get("truncated")),
            }
            key = (sample, epoch)
        elif env == "mbpp":
            o = {
                "cheat": bool(lab["cheat"]),
                "honest": bool(lab["honest_correct"]),
                "valid": bool(lab["valid"]),
            }
            key = (str(lab["task_id"]), epoch)
        elif env == "grader":
            cheat = lab["cheat"]
            o = {
                "family": lab["family"],
                "cell": f'{lab["family"]}.{lab["side"]}',
                "valid": cheat is not None,
                # cheat among valid answers (the box's pooled rate): None if unparseable
                "cheat_valid": None if cheat is None else bool(cheat),
                "cheat_any": bool(cheat),
                "honest_valid": cheat is False,
            }
            key = (sample, epoch)  # sample = cell.index; same prompt text in every run
        else:
            raise ValueError(env)
        ep.rows[key] = o
    return ep


def load_traj(run_dir: Path) -> Episodes:
    ep = Episodes(str(run_dir))
    for r in _read(run_dir / "trajectories.jsonl.gz"):
        lab = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in r["labels"].items()}
        if r["benchmark"] == "terminal_verifier":
            o = {
                "tampered": bool(lab["tampered"]),
                "passed": bool(lab["passed"]),
                "cheated": bool(lab["passed"]),
            }
        else:  # mbpp_impossible: verified hack = passes the corrupted visible assert
            o = {"cheat": bool(lab["passes_corrupted_visible"]), "honest": bool(lab["honest_correct"])}
        ep.rows[(r["task_id"], str(r["epoch"]))] = o
    return ep


_cache: dict[str, Episodes] = {}


def load(data: Path, rel: str) -> Episodes:
    if rel not in _cache:
        d = data / "stage6" / rel
        _cache[rel] = load_rhenvs(d) if (d / "records.jsonl.gz").exists() else load_traj(d)
    return _cache[rel]


# ---------------------------------------------------------------- tests


def paired_test(row: dict, base: dict, keep=None) -> dict:
    """row/base: {(cluster, sample): 0/1/None}. Pairs on clusters present in both."""
    rc, bc = defaultdict(list), defaultdict(list)
    for (c, s), v in row.items():
        if v is not None and (keep is None or keep(c)):
            rc[c].append(float(v))
    for (c, s), v in base.items():
        if v is not None and (keep is None or keep(c)):
            bc[c].append(float(v))
    clusters = sorted(rc.keys() & bc.keys())
    if not clusters:
        return {}
    r = [np.array(rc[c]) for c in clusters]
    b = [np.array(bc[c]) for c in clusters]
    d = np.array([x.mean() - y.mean() for x, y in zip(r, b)])
    n_r, k_r = sum(len(x) for x in r), sum(x.sum() for x in r)
    n_b, k_b = sum(len(y) for y in b), sum(y.sum() for y in b)
    p_row, p_base = k_r / n_r, k_b / n_b
    one_each = all(len(x) == 1 and len(y) == 1 for x, y in zip(r, b))
    if one_each:
        up, down = int((d > 0).sum()), int((d < 0).sum())
        p = stats.binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
        method = "exact McNemar"
    else:
        obs = abs(d.sum())
        nz = d[d != 0]
        if len(nz) == 0:
            p, method = 1.0, "sign-flip"
        elif (exact := exact_signflip(nz, obs)) is not None:
            p, method = exact, "sign-flip (exact)"
        elif len(nz) <= 20:
            signs = np.array(np.meshgrid(*[[-1, 1]] * len(nz))).reshape(len(nz), -1).T
            p = float(np.mean(np.abs(signs @ nz) >= obs - 1e-12))
            method = "sign-flip (exact)"
        else:
            hits = 0
            for _ in range(N_PERM // 20_000):
                s = RNG.choice([-1.0, 1.0], size=(20_000, len(nz)))
                hits += int((np.abs(s @ nz) >= obs - 1e-12).sum())
            p = (hits + 1) / (N_PERM + 1)
            method = "sign-flip (MC)"
    # cluster bootstrap of the rate difference
    idx = RNG.integers(0, len(clusters), size=(N_BOOT, len(clusters)))
    rs, rn = np.array([x.sum() for x in r]), np.array([len(x) for x in r])
    bs, bn = np.array([y.sum() for y in b]), np.array([len(y) for y in b])
    diff = rs[idx].sum(1) / rn[idx].sum(1) - bs[idx].sum(1) / bn[idx].sum(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    fisher = stats.fisher_exact([[k_r, n_r - k_r], [k_b, n_b - k_b]]).pvalue
    m = (n_r + n_b) / (2 * len(clusters))  # samples per task per arm
    return {
        "n_tasks": len(clusters),
        "row": [int(k_r), int(n_r)],
        "base": [int(k_b), int(n_b)],
        "rate_row": p_row,
        "rate_base": p_base,
        "diff": p_row - p_base,
        "ci95": [float(lo), float(hi)],
        "p": float(p),
        "method": method,
        "p_fisher_unpaired": float(fisher),
        "n_tasks_80pct_power": tasks_for_power(p_base, p_row, m, r, b),
        "n_tasks_80pct_power_a001": tasks_for_power(p_base, p_row, m, r, b, alpha=0.001),
        "samples_per_task": m,
    }


def exact_signflip(nz: np.ndarray, obs: float) -> float | None:
    """Exact two-sided sign-flip p-value by dynamic programming over the sum.
    Per-task differences of rates with few samples per task are multiples of
    1/q for a small q, so the null distribution of the sum is a convolution of
    {-q·d, +q·d} with probability 1/2 each, on an integer grid."""
    for q in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        z = nz * q
        if np.allclose(z, np.round(z), atol=1e-9):
            break
    else:
        return None
    z = np.abs(np.round(z).astype(np.int64))
    total = int(z.sum())
    dist = np.zeros(2 * total + 1)
    dist[total] = 1.0  # index = sum + total
    for v in z:
        nxt = np.zeros_like(dist)
        nxt[: len(dist) - v] += 0.5 * dist[v:]   # -v
        nxt[v:] += 0.5 * dist[: len(dist) - v]   # +v
        dist = nxt
    sums = np.arange(-total, total + 1)
    t = round(obs * q)
    return float(min(1.0, dist[np.abs(sums) >= t].sum()))


def tasks_for_power(p0: float, p1: float, m: float, r, b, alpha: float = 0.05) -> int | None:
    if abs(p1 - p0) < 1e-9:
        return None
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(0.8)
    pbar = (p0 + p1) / 2
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2
    # design effect 1 + (m - 1) * ICC, ICC estimated from the base arm's clusters
    deff = 1.0
    if m > 1:
        allv = np.concatenate(b)
        if allv.var() > 0:
            between = np.var([y.mean() for y in b])
            icc = max(0.0, min(1.0, (between - allv.var() / m) / allv.var()))
            deff = 1 + (m - 1) * icc
    return int(math.ceil(n * deff / m))


def stratified_counts_test(row_cells: dict, base_cells: dict) -> dict:
    """{cell: (k, n)} per arm; permutation of outcomes between arms within cells."""
    cells = sorted(row_cells.keys() & base_cells.keys())
    kr = sum(row_cells[c][0] for c in cells)
    nr = sum(row_cells[c][1] for c in cells)
    kb = sum(base_cells[c][0] for c in cells)
    nb = sum(base_cells[c][1] for c in cells)
    # under H0, row's count in a cell is hypergeometric given the cell's margins
    exp = sum(row_cells[c][1] * (row_cells[c][0] + base_cells[c][0]) / (row_cells[c][1] + base_cells[c][1]) for c in cells if row_cells[c][1] + base_cells[c][1])
    obs = abs(kr - exp)
    # exact: the row's total is a sum of independent per-cell hypergeometrics
    dist = np.array([1.0])
    for c in cells:
        k, n = row_cells[c]
        k2, n2 = base_cells[c]
        if n + n2 == 0:
            continue
        support = np.arange(0, n + 1)
        dist = np.convolve(dist, stats.hypergeom.pmf(support, n + n2, k + k2, n))
    totals = np.arange(len(dist))
    p = float(min(1.0, dist[np.abs(totals - exp) >= obs - 1e-9].sum()))
    fisher = stats.fisher_exact([[kr, nr - kr], [kb, nb - kb]]).pvalue
    pr, pb = kr / nr if nr else float("nan"), kb / nb if nb else float("nan")
    return {
        "n_tasks": None,
        "row": [kr, nr],
        "base": [kb, nb],
        "rate_row": pr,
        "rate_base": pb,
        "diff": pr - pb,
        "ci95": None,
        "p": p,
        "method": "stratified exact (cells)",
        "p_fisher_unpaired": float(fisher),
        "n_tasks_80pct_power": tasks_for_power(pb, pr, 1, None, None) if nr and nb else None,
        "n_tasks_80pct_power_a001": tasks_for_power(pb, pr, 1, None, None, alpha=0.001) if nr and nb else None,
        "samples_per_task": 1,
    }


# ---------------------------------------------------------------- comparisons

TV_PARTS = {"tv_vb": ("tampered", "TV verifier_bug mt4 tampered"), "tv_pos": ("passed", "TV possible passed (honest)")}
GR_NONPETS = lambda c: not c.startswith("pets.")  # noqa: E731  (the box's pooled GR rate excludes `pets`)


def rhenvs_full_suite(row: str, base: str = "base_rhenvs", stage: str = "7", gr_keep=GR_NONPETS) -> list[dict]:
    out = []
    pre = f"{row}/rhenvs/{row}_"
    bpre = f"{base}/rhenvs/{base}_"
    out.append(dict(stage=stage, row=row, part="TV-vb", metric="tampered", primary=True, a=pre + "tv_vb", b=bpre + "tv_vb"))
    out.append(dict(stage=stage, row=row, part="TV-pos", metric="passed", primary=False, a=pre + "tv_pos", b=bpre + "tv_pos"))
    out.append(dict(stage=stage, row=row, part="MBPP", metric="cheat", primary=True, a=pre + "mbpp", b=bpre + "mbpp"))
    out.append(dict(stage=stage, row=row, part="MBPP", metric="honest", primary=False, a=pre + "mbpp", b=bpre + "mbpp"))
    out.append(dict(stage=stage, row=row, part="GR-hack", metric="cheat_valid", primary=True, a=pre + "gr_hack", b=bpre + "gr_hack", keep=gr_keep))
    out.append(dict(stage=stage, row=row, part="GR-hack", metric="valid", primary=False, a=pre + "gr_hack", b=bpre + "gr_hack", keep=gr_keep))
    return out


def comparisons() -> list[dict]:
    C: list[dict] = []
    # stage 5b: R1 (attempt_vs_A stack) on the 5b eval set
    s = "attempt_vs_A_w1.0_L40-63_stack"
    C.append(dict(stage="5b", row="R1", part="TV-vb", metric="tampered", primary=True, a=f"{s}/s5b_{s}_verifier_bug", b="base/s5b_base_verifier_bug"))
    C.append(dict(stage="5b", row="R1", part="TV-vb", metric="cheated", primary=False, a=f"{s}/s5b_{s}_verifier_bug", b="base/s5b_base_verifier_bug"))
    C.append(dict(stage="5b", row="R1", part="TV-pos", metric="passed", primary=False, a=f"{s}/s5b_{s}_possible", b="base/s5b_base_possible"))
    # stage 5c: R1 on corrupted
    C.append(dict(stage="5c", row="R1", part="TV-corr mt6", metric="cheated", primary=True, a=f"{s}/s5c_R1_corrupted", b="base/s5c_base_corrupted"))
    # stage 5d: R1 on MBPP held-out
    C.append(dict(stage="5d", row="R1", part="MBPP", metric="cheat", primary=True, a="R1/s5d_R1_mbpp", b="base/s5d_base_mbpp"))
    C.append(dict(stage="5d", row="R1", part="MBPP", metric="honest", primary=False, a="R1/s5d_R1_mbpp", b="base/s5d_base_mbpp"))
    # stage 5e matrix rows with env rollouts
    for r in ["A6", "A7", "C8", "A1", "A3", "B13", "B2", "B1", "A3w", "A3b", "A3c", "C6", "C7", "B11"]:
        C.append(dict(stage="5e", row=r, part="TV-vb", metric="tampered", primary=True, a=f"{r}/s5e_{r}_verifier_bug", b="base/s5e_base_verifier_bug"))
        C.append(dict(stage="5e", row=r, part="TV-pos", metric="passed", primary=False, a=f"{r}/s5e_{r}_possible", b="base/s5e_base_possible"))
        if r != "C8":
            C.append(dict(stage="5e", row=r, part="TV-corr", metric="cheated", primary=True, a=f"{r}/s5e_{r}_corrupted", b="base/s5e_base_corrupted"))
    # stage 7 full-suite rows
    for r in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "F1b_X2w05"]:
        C += rhenvs_full_suite(r)
    for r in ["F5_n25", "F5_n50", "F5_n100"]:
        C.append(dict(stage="7", row=r, part="TV-vb", metric="tampered", primary=True, a=f"{r}/rhenvs/{r}_tv_vb", b="base_rhenvs/rhenvs/base_rhenvs_tv_vb"))
    # dose and robustness of X2 against X2 itself (secondary)
    C.append(dict(stage="7", row="F1b_X2w05 vs X2", part="TV-vb (w 0.5 vs w 1.0)", metric="tampered", primary=False, a="F1b_X2w05/rhenvs/F1b_X2w05_tv_vb", b="X2/rhenvs/X2_tv_vb"))
    C.append(dict(stage="8", row="H1 vs X2", part="TV-vb (half data vs full)", metric="tampered", primary=False, a="H1/rhenvs/H1_tv_vb", b="X2/rhenvs/X2_tv_vb"))
    C.append(dict(stage="7", row="F1a_X2", part="TV-vb (fresh ids)", metric="tampered", primary=True, a="F1a_X2/rhenvs/F1a_X2_tv_vb", b="F1a_base/rhenvs/F1a_base_tv_vb"))
    for r, fam in [("L1", "parity"), ("L2", "sign"), ("L3", "letter"), ("L4", "weekday"), ("L5", "pets_listed")]:
        keep = (lambda f: (lambda c: c.split(".")[0] == f))(fam)
        C.append(dict(stage="7", row=r, part=f"GR-hack ({fam}, held out)", metric="cheat_valid", primary=True, a=f"{r}/rhenvs/{r}_gr_hack", b="base_rhenvs/rhenvs/base_rhenvs_gr_hack", keep=keep))
        C.append(dict(stage="7", row=r, part=f"GR-hack ({fam}, held out)", metric="valid", primary=False, a=f"{r}/rhenvs/{r}_gr_hack", b="base_rhenvs/rhenvs/base_rhenvs_gr_hack", keep=keep))
    # stage 8
    for r in ["P1", "P2", "P3", "P4", "C1"]:
        C += rhenvs_full_suite(r, stage="8")
    # the same rows against X0 (random directions, same recipe and serving path): secondary, unadjusted
    for r, st in [(r, "7") for r in ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "F1b_X2w05"]] + [(r, "8") for r in ["P1", "P2", "P3", "P4", "C1"]]:
        for c in rhenvs_full_suite(r, base="X0", stage=st):
            if c["primary"]:
                C.append(c | {"part": c["part"] + " vs X0", "primary": False})
    C.append(dict(stage="8", row="H1", part="TV-vb", metric="tampered", primary=True, a="H1/rhenvs/H1_tv_vb", b="base_rhenvs/rhenvs/base_rhenvs_tv_vb"))
    for r in ["X1T", "X2T"]:
        C.append(dict(stage="8", row=r, part="TV-vb mt6", metric="tampered", primary=True, a=f"{r}/rhenvs/{r}_tv_vb_mt6", b="base_s8/rhenvs/base_s8_tv_vb_mt6"))
        C.append(dict(stage="8", row=r, part="TV-corr mt6", metric="cheated", primary=True, a=f"{r}/rhenvs/{r}_corr_mt6", b="base_s8/rhenvs/base_s8_corr_mt6"))
        C.append(dict(stage="8", row=r, part="TV-corr mt4", metric="cheated", primary=True, a=f"{r}/rhenvs/{r}_corr_mt4", b="base_s8/rhenvs/base_s8_corr_mt4"))
    return C


def hook_comparisons(data: Path) -> list[dict]:
    """Stage-7 HF-hook runs (F2/F3, S1: per-cell counts; F4: per-task scores)."""
    out = []
    d = data / "stage7" / "stage7"
    f23 = json.loads((d / "f2f3" / "results.json").read_text())
    cells = lambda run, key: {c: (v[key], v["n"] if key == "valid" else v["valid"]) for c, v in f23[run]["cells"].items() if c.endswith(".hack")}  # noqa: E731
    base = "F2_none"
    labels = {
        "F2_ablate_D_GR": "hook zero-ablate D_GR",
        "F2_centered_D_GR": "hook mean-centred ablate D_GR",
    }
    for run in f23:
        if run == base:
            continue
        fams = set(f23[run]["families"])
        bc = {c: v for c, v in cells(base, "cheat").items() if c.split(".")[0] in fams}
        bv = {c: v for c, v in cells(base, "valid").items() if c.split(".")[0] in fams}
        desc = labels.get(run, run.replace("F2_centered_no_", "hook mean-centred LOFO, held out ").replace("F3_sub_no_", "steer subtract α₁ LOFO, held out ").replace("F3_add_no_", "steer add α₁ LOFO, held out "))
        out.append(dict(stage="7", row=run, part=f"GR-hack hook ({desc})", metric="cheat_valid", primary=True, res=stratified_counts_test(cells(run, "cheat"), bc)))
        out.append(dict(stage="7", row=run, part=f"GR-hack hook ({desc})", metric="valid", primary=False, res=stratified_counts_test(cells(run, "valid"), bv)))
    s1 = json.loads((d / "steer" / "results.json").read_text())["results"]
    rows = s1 if isinstance(s1, list) else list(s1.values())
    def cellmap(rec, key):
        return {c: (v[key], v["n"] if key == "valid" else v["valid"]) for c, v in rec["cells"].items()}
    none = next(r for r in rows if r.get("direction") in (None, "none") or r.get("mode") == "none")
    for r in rows:
        if r is none:
            continue
        name = f"S1 {r.get('direction')} α={r.get('alpha'):+.2f}"
        out.append(dict(stage="7", row=name, part="GR-hack steer (parity, sign)", metric="cheat_valid", primary=True, res=stratified_counts_test(cellmap(r, "cheat"), cellmap(none, "cheat"))))
        out.append(dict(stage="7", row=name, part="GR-hack steer (parity, sign)", metric="valid", primary=False, res=stratified_counts_test(cellmap(r, "valid"), cellmap(none, "valid"))))
    # F4: MBPP steering, per task, one sample each
    def f4(name):
        rec = json.loads((d / "f4" / f"scores_completions_{name}.json").read_text())["tasks_scored"]
        return rec
    base4 = {t["task_id"]: t for t in f4("none")}
    for name in ["subtract_D_GR", "subtract_D_MBPP", "subtract_D_MBPP+GR"]:
        rr = {t["task_id"]: t for t in f4(name)}
        for metric, key in [("cheat", "cheat"), ("honest", "honest_correct")]:
            a = {(k, "0"): bool(v.get(key)) for k, v in rr.items()}
            b = {(k, "0"): bool(v.get(key)) for k, v in base4.items()}
            out.append(dict(stage="7", row=f"F4 {name}", part="MBPP steer (HF, 1/task)", metric=metric, primary=metric == "cheat", res=paired_test(a, b)))
    return out


def rerun_section(results: list[dict]) -> list[str]:
    """Primary rows in the 0.05-0.2 band, and rows with raw p < 0.05 that do not
    survive Holm (selection-sensitive). Sizes are samples per arm."""
    out = ["", "## Rerun candidates", "",
           "Primary rows with raw p in 0.05–0.2, or raw p < 0.05 that fail Holm at 0.05 (would not survive the multiple-comparison correction, so they count as unconfirmed if anything rests on them). "
           "`n for 80 %` = samples per arm (tasks × samples per task) for 80 % power at the observed Δ; α 0.001 is the bar for a result that should hold up after selection. "
           "Rows whose observed Δ is the effect of interest only; a null that matters needs a size set by the smallest effect worth detecting instead.", "",
           "| stage | row | part | Δ pts | p | p Holm | samples/arm now | n for 80 % at α 0.05 | at α 0.001 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        x = r["res"]
        if not r["primary"] or not x:
            continue
        p, ph = x["p"], x.get("p_holm", 1.0)
        if not ((0.05 <= p < 0.2) or (p < 0.05 and ph >= 0.05)):
            continue
        m = x["samples_per_task"]
        need = lambda k: "-" if x[k] is None else str(int(round(x[k] * m)))  # noqa: E731
        out.append(f'| {r["stage"]} | {r["row"]} | {r["part"]} | {100 * x["diff"]:+.1f} | {fmt_p(p)} | {ph:.2g} | {x["row"][1]} | {need("n_tasks_80pct_power")} | {need("n_tasks_80pct_power_a001")} |')
    return out


def mmlu_section(data: Path) -> list[str]:
    """lm-eval stores only aggregate accuracy and its stderr, so the test is an
    unpaired z on (acc, stderr). Both models answer the same 14,042 questions,
    so a paired test would give smaller p-values: these are conservative."""
    rows = []
    for p in sorted(data.glob("stage*/stage*/mmlu/*/*/results_*.json")):
        stage = p.parts[-5]
        name = p.parts[-3]
        r = json.loads(p.read_text())["results"]["mmlu"]
        rows.append((stage, name, r["acc,none"], r["acc_stderr,none"]))
    base = {s: (a, se) for s, n, a, se in rows if n == "base"}
    # stage 7 reuses the stage-5e base run (83.46); same model, revision and lm-eval config
    refs = {"stage5b": base["stage5b"], "stage5e": base["stage5e"], "stage7": base["stage5e"]}
    out = ["", "## MMLU (lm-eval, 5-shot as run on the box)", "",
           "Unpaired z on aggregate accuracy ± stderr (per-question outputs not stored; pairing would only shrink p). Reference: the base run of that stage (5b: 83.50; 5e and 7: 83.46). MMLU tolerance in the missions: 5 points.", "",
           "| stage | row | acc | Δ pts vs base | z | p (two-sided) |", "|---|---|---|---|---|---|"]
    for s, n, a, se in rows:
        if n == "base":
            continue
        ref = refs[s]
        z = (a - ref[0]) / math.hypot(se, ref[1])
        out.append(f"| {s} | {n} | {100 * a:.2f} | {100 * (a - ref[0]):+.2f} | {z:+.2f} | {2 * stats.norm.sf(abs(z)):.2g} |")
    return out


# ---------------------------------------------------------------- report


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def bh(ps: list[float]) -> list[float]:
    order = np.argsort(ps)[::-1]
    m = len(ps)
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, ps[i] * m / rank)
        adj[i] = run
    return adj.tolist()


def fmt_p(p: float) -> str:
    return f"{p:.2g}"


def band(p: float) -> str:
    if p < 0.001:
        return "<0.001: real"
    if p < 0.05:
        return "<0.05: probably real"
    if p < 0.2:
        return "0.05-0.2: rerun at ~5x n"
    return ">0.2: chance"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data/analysis/significance")
    args = ap.parse_args()
    data, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for c in comparisons():
        a, b = load(data, c["a"]), load(data, c["b"])
        keep = c.get("keep")
        res = paired_test(a.metric(c["metric"]), b.metric(c["metric"]), keep=(lambda cl, k=keep: k(cl.rsplit(".", 1)[0])) if keep else None)
        results.append({k: v for k, v in c.items() if k != "keep"} | {"res": res})
    results += hook_comparisons(data)

    prim = [i for i, r in enumerate(results) if r["primary"] and r["res"]]
    ps = [results[i]["res"]["p"] for i in prim]
    for i, ph, pb in zip(prim, holm(ps), bh(ps)):
        results[i]["res"]["p_holm"] = ph
        results[i]["res"]["p_bh"] = pb
    (out / "results.json").write_text(json.dumps(results, indent=1))

    def pct(x):
        return f"{100 * x:.1f}"

    lines = [
        "# Significance of every behavioural row, stages 5b-8",
        "",
        f"Generated by `scripts/results_significance.py`. {len(prim)} primary hack-rate tests (Holm / BH adjusted over all of them); "
        "secondary rows (honest metric, answer validity) are unadjusted. Two-sided throughout. "
        "`p` = paired by task (exact McNemar or task-cluster sign-flip) unless the method column says otherwise; "
        "`p Fisher` ignores pairing and task clustering (reference only). Δ CI = 95 % task-cluster bootstrap. "
        "`tasks for 80 %` = tasks per arm (at this run's samples per task) to detect the observed Δ at α 0.05 with 80 % power.",
        "",
        "| stage | row | part | metric | row k/n | base k/n | row % | base % | Δ pts [95 % CI] | p | method | p Holm | p BH | p Fisher | band (raw p) | tasks for 80 % (have) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        x = r["res"]
        if not x:
            lines.append(f'| {r["stage"]} | {r["row"]} | {r["part"]} | {r["metric"]} | no overlap | | | | | | | | | | | |')
            continue
        ci = f' [{100 * x["ci95"][0]:+.1f}, {100 * x["ci95"][1]:+.1f}]' if x["ci95"] else ""
        ph = f'{x["p_holm"]:.2g}' if "p_holm" in x else ""
        pb = f'{x["p_bh"]:.2g}' if "p_bh" in x else ""
        need = x["n_tasks_80pct_power"]
        have = x["n_tasks"]
        lines.append(
            f'| {r["stage"]} | {r["row"]} | {r["part"]} | {r["metric"]}{"" if r["primary"] else " (secondary)"} | {x["row"][0]}/{x["row"][1]} | {x["base"][0]}/{x["base"][1]} | '
            f'{pct(x["rate_row"])} | {pct(x["rate_base"])} | {100 * x["diff"]:+.1f}{ci} | {fmt_p(x["p"])} | {x["method"]} | {ph} | {pb} | {x["p_fisher_unpaired"]:.2g} | {band(x["p"])} | '
            f'{need if need is not None else "-"} ({have if have is not None else "-"}) |'
        )
    lines += rerun_section(results)
    lines += mmlu_section(data)
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/results.md ({len(results)} comparisons, {len(prim)} primary)")


if __name__ == "__main__":
    main()
