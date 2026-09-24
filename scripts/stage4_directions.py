#!/usr/bin/env python
"""Stage 4, step 3: directions, held-out probes and direction comparisons from the activation cache (CPU).

For every contrast in the config and every aggregation: per-layer difference-of-means direction fitted on
train-split tasks; held-out (eval-split tasks) AUROC of the projection onto it; logistic-regression probe on
a layer subset; covariate-only baseline; projections of the other classes; cross-contrast transfer; split-half
noise floor; sample-size curve. Directions are written as safetensors with a JSON sidecar.

  uv run scripts/stage4_directions.py --config configs/stage4_terminal_verifier.yaml --cache-dir data/stage4/cache/tv27b \
      --out data/stage4/directions/tv27b-full [--subset-per-class 400] [--aggregations ...] [--contrasts ...] [--no-probes]

Runs on a partial cache (only rows listed in index.jsonl are used) and on the laptop from a downloaded cache.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.stage4_analysis import (  # noqa: E402
    auroc, cosine, diff_of_means, eval_probe, fit_probe, projection_auc, projection_scores, sample_efficiency, split_half_cosine, unit,
)
from rhablation.stage4_cache import CacheReader  # noqa: E402
from rhablation.stage4_config import DERIVED_AGGREGATIONS, ContrastSpec, load_stage4_config  # noqa: E402
from rhablation.stage4_manifest import ManifestRow, read_manifest  # noqa: E402


def select_rows(rows: list[ManifestRow], done: set[int], subset_per_class: int | None) -> list[ManifestRow]:
    rows = [r for r in rows if r.order in done]
    if subset_per_class is None:
        return rows
    taken: dict[str, int] = defaultdict(int)
    out = []
    for r in rows:  # rows are in processing order, so "first N per class" is the early-usable subset
        if taken[r.cls] < subset_per_class:
            taken[r.cls] += 1
            out.append(r)
    return out


def contrast_mask(rows: list[ManifestRow], spec: ContrastSpec, side: str) -> np.ndarray:
    classes = set(spec.positive if side == "pos" else spec.negative)
    mt = spec.positive_max_turns if side == "pos" else spec.negative_max_turns
    return np.array([(r.cls in classes) and (mt is None or r.max_turns == mt) for r in rows])


def length_matched(idx_pos: np.ndarray, idx_neg: np.ndarray, seq_len: np.ndarray, seed: int, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Subsample the larger class within seq_len bins so both classes have the same length histogram."""
    rng = np.random.default_rng(seed)
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return idx_pos, idx_neg
    edges = np.quantile(seq_len[np.concatenate([idx_pos, idx_neg])], np.linspace(0, 1, bins + 1))
    keep_pos, keep_neg = [], []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        inb = lambda idx: idx[(seq_len[idx] >= lo) & ((seq_len[idx] < hi) if b < bins - 1 else (seq_len[idx] <= hi))]
        p, n = inb(idx_pos), inb(idx_neg)
        k = min(len(p), len(n))
        if k == 0:
            continue
        keep_pos.append(rng.choice(p, size=k, replace=False))
        keep_neg.append(rng.choice(n, size=k, replace=False))
    if not keep_pos:
        return idx_pos[:0], idx_neg[:0]
    return np.concatenate(keep_pos), np.concatenate(keep_neg)


def layer_matrix(reader: CacheReader, agg: str, layer: int, orders: np.ndarray, rows: list[ManifestRow]) -> np.ndarray:
    if agg == "completion_mean":
        cot = reader.layer("cot_mean", layer, orders)
        ans = reader.layer("answer_mean", layer, orders)
        n_cot = np.array([r.n_cot_tokens for r in rows], dtype=np.float32)[:, None]
        n_ans = np.array([r.n_answer_tokens for r in rows], dtype=np.float32)[:, None]
        tot = np.maximum(n_cot + n_ans, 1.0)
        return (cot * n_cot + ans * n_ans) / tot
    return reader.layer(agg, layer, orders)


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset-per-class", type=int, help="use only the first N finished rows of each class (processing order)")
    ap.add_argument("--aggregations", nargs="+")
    ap.add_argument("--contrasts", nargs="+")
    ap.add_argument("--no-probes", action="store_true", help="skip logistic-regression probes (projection AUROC only)")
    ap.add_argument("--no-curves", action="store_true", help="skip split-half and sample-size curves")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_stage4_config(args.config)
    an = cfg.analysis
    reader = CacheReader(args.cache_dir)
    rows_all = read_manifest(args.cache_dir / "manifest.jsonl")
    rows = select_rows(rows_all, reader.done_orders(), args.subset_per_class)
    if not rows:
        print("no finished rows in the cache", file=sys.stderr)
        return 1
    aggs = args.aggregations or an.aggregations
    for a in aggs:
        needed = ["cot_mean", "answer_mean"] if a in DERIVED_AGGREGATIONS else [a]
        for n in needed:
            if n not in reader.aggregations:
                print(f"aggregation {a} needs {n}, which is not in the cache ({reader.aggregations})", file=sys.stderr)
                return 1
    contrasts = {k: v for k, v in an.contrasts.items() if not args.contrasts or k in args.contrasts}
    L1 = reader.n_layers_plus1
    orders = np.array([r.order for r in rows])
    split = np.array([r.split for r in rows])
    cls = np.array([r.cls for r in rows])
    seq_len = np.array([r.seq_len for r in rows], dtype=np.float32)
    task = np.array([r.task_id for r in rows])
    classes = sorted(set(cls))
    class_counts = {c: {"train": int(((cls == c) & (split == "train")).sum()), "eval": int(((cls == c) & (split == "eval")).sum())} for c in classes}
    print(f"{len(rows)} rows ({len(reader.done_orders())} finished in cache, {len(rows_all)} in manifest); classes: {class_counts}")

    # Index sets per contrast
    sets: dict[str, dict[str, np.ndarray]] = {}
    for name, spec in contrasts.items():
        pos, neg = contrast_mask(rows, spec, "pos"), contrast_mask(rows, spec, "neg")
        s = {
            "train_pos": np.where(pos & (split == "train"))[0], "train_neg": np.where(neg & (split == "train"))[0],
            "eval_pos": np.where(pos & (split == "eval"))[0], "eval_neg": np.where(neg & (split == "eval"))[0],
        }
        s["eval_pos_lm"], s["eval_neg_lm"] = length_matched(s["eval_pos"], s["eval_neg"], seq_len, an.seed)
        sets[name] = s
        print(f"  {name}: train {len(s['train_pos'])}/{len(s['train_neg'])}, eval {len(s['eval_pos'])}/{len(s['eval_neg'])}, length-matched eval {len(s['eval_pos_lm'])}/{len(s['eval_neg_lm'])}")
        if min(len(s["train_pos"]), len(s["train_neg"]), len(s["eval_pos"]), len(s["eval_neg"])) < 2:
            print(f"  WARNING: contrast {name} has too few rows on some side; skipped")
    contrasts = {k: v for k, v in contrasts.items() if min(len(sets[k][s]) for s in ("train_pos", "train_neg", "eval_pos", "eval_neg")) >= 2}

    # Storage for directions: [contrast][agg] -> (L1, hidden) arrays
    H = reader.hidden
    dirs = {c: {a: {"direction": np.zeros((L1, H), np.float32), "mean_pos": np.zeros((L1, H), np.float32), "mean_neg": np.zeros((L1, H), np.float32)} for a in aggs} for c in contrasts}
    res: dict = {
        "cache_dir": str(args.cache_dir), "manifest_id": cfg.manifest_id, "model": reader.config.get("model"), "model_revision": reader.config.get("model_revision"),
        "layer_index_convention": reader.config.get("layer_index_convention"), "n_layers_plus1": L1, "hidden": H,
        "n_rows_used": len(rows), "n_rows_finished_in_cache": len(reader.done_orders()), "subset_per_class": args.subset_per_class,
        "class_counts": class_counts, "aggregations": aggs, "winsorization_quantile": an.winsorization_quantile,
        "contrasts": {}, "covariate_baseline": {}, "cosine_pairs": {}, "aggregation_cosines": {},
    }
    for c, spec in contrasts.items():
        s = sets[c]
        res["contrasts"][c] = {
            "positive": spec.positive, "negative": spec.negative, "positive_max_turns": spec.positive_max_turns, "negative_max_turns": spec.negative_max_turns,
            "description": spec.description,
            "n_train_pos": len(s["train_pos"]), "n_train_neg": len(s["train_neg"]), "n_eval_pos": len(s["eval_pos"]), "n_eval_neg": len(s["eval_neg"]),
            "n_eval_length_matched": len(s["eval_pos_lm"]),
            "aggregations": {a: {"proj_auc": [], "proj_auc_length_matched": [], "direction_norm": [], "split_half_cos_mean": [], "split_half_cos_std": [],
                                 "sample_curve": {str(n): {"cos_mean": [], "cos_std": [], "auc_mean": [], "auc_std": []} for n in an.sample_sizes},
                                 "class_projection_mean": {k: [] for k in classes}, "transfer_auc": {}, "probe": {}} for a in aggs},
        }

    # Covariate-only baseline per contrast
    cov = np.array([[float(getattr(r, f) or 0) for f in an.covariates] for r in rows], dtype=np.float32)
    for c in contrasts:
        s = sets[c]
        tr = np.concatenate([s["train_pos"], s["train_neg"]]); ytr = np.r_[np.ones(len(s["train_pos"])), np.zeros(len(s["train_neg"]))]
        ev = np.concatenate([s["eval_pos"], s["eval_neg"]]); yev = np.r_[np.ones(len(s["eval_pos"])), np.zeros(len(s["eval_neg"]))]
        probe = fit_probe(cov[tr], ytr, C=an.probe_C, max_iter=an.probe_max_iter, seed=an.seed)
        res["covariate_baseline"][c] = {"features": an.covariates, **eval_probe(probe, cov[ev], yev)}

    # Main pass: per aggregation, per layer
    transfer_pairs = [tuple(p) for p in an.transfer_pairs if p[0] in contrasts and p[1] in contrasts]
    for a in aggs:
        for layer in range(L1):
            X = layer_matrix(reader, a, layer, orders, rows)
            layer_dirs = {}
            for c, spec in contrasts.items():
                s = sets[c]; R = res["contrasts"][c]["aggregations"][a]
                d, mp, mn = diff_of_means(X[s["train_pos"]], X[s["train_neg"]], an.winsorization_quantile)
                dirs[c][a]["direction"][layer], dirs[c][a]["mean_pos"][layer], dirs[c][a]["mean_neg"][layer] = d, mp, mn
                layer_dirs[c] = (d, mp, mn)
                ev = np.concatenate([s["eval_pos"], s["eval_neg"]]); yev = np.r_[np.ones(len(s["eval_pos"])), np.zeros(len(s["eval_neg"]))]
                R["proj_auc"].append(projection_auc(d, X[ev], yev))
                lm = np.concatenate([s["eval_pos_lm"], s["eval_neg_lm"]]); ylm = np.r_[np.ones(len(s["eval_pos_lm"])), np.zeros(len(s["eval_neg_lm"]))]
                R["proj_auc_length_matched"].append(projection_auc(d, X[lm], ylm) if len(lm) >= 4 else float("nan"))
                R["direction_norm"].append(float(np.linalg.norm(d)))
                # Where do the classes fall along this direction? Eval rows for the contrast's own classes, all rows otherwise.
                own = set(spec.positive) | set(spec.negative)
                for k in classes:
                    sel = np.where((cls == k) & ((split == "eval") if k in own else True))[0]
                    R["class_projection_mean"][k].append(float(projection_scores(d, mp, mn, X[sel]).mean()) if len(sel) else float("nan"))
                if not args.no_curves:
                    sh = split_half_cosine(X[s["train_pos"]], task[s["train_pos"]], X[s["train_neg"]], task[s["train_neg"]], an.bootstrap_reps, an.seed + layer, an.winsorization_quantile)
                    R["split_half_cos_mean"].append(float(np.nanmean(sh))); R["split_half_cos_std"].append(float(np.nanstd(sh)))
                    for n in an.sample_sizes:
                        se = sample_efficiency(X[s["train_pos"]], X[s["train_neg"]], d, X[ev], yev, n, an.sample_reps, an.seed + layer, an.winsorization_quantile)
                        for k2, v2 in se.items():
                            R["sample_curve"][str(n)][k2].append(v2)
            for src, dst in transfer_pairs:
                s = sets[dst]; ev = np.concatenate([s["eval_pos"], s["eval_neg"]]); yev = np.r_[np.ones(len(s["eval_pos"])), np.zeros(len(s["eval_neg"]))]
                res["contrasts"][src]["aggregations"][a]["transfer_auc"].setdefault(dst, []).append(projection_auc(layer_dirs[src][0], X[ev], yev))
            if layer % 8 == 0 or layer == L1 - 1:
                print(f"  [{a}] layer {layer}/{L1 - 1}: " + ", ".join(f"{c}={res['contrasts'][c]['aggregations'][a]['proj_auc'][-1]:.3f}" for c in contrasts) + f"  ({time.time() - t0:.0f}s)")
        # Best layer per contrast for this aggregation
        for c in contrasts:
            R = res["contrasts"][c]["aggregations"][a]
            pa = np.array(R["proj_auc"]); pa = np.where(np.isnan(pa), -1, pa)
            R["best_layer"] = int(np.argmax(pa)); R["best_proj_auc"] = float(pa.max())

    # Probes on a layer subset (+ best projection layer per contrast)
    if not args.no_probes:
        for a in aggs:
            layers = sorted({*range(0, L1, an.probe_layer_stride), L1 - 1, *(res["contrasts"][c]["aggregations"][a]["best_layer"] for c in contrasts)})
            for layer in layers:
                X = layer_matrix(reader, a, layer, orders, rows)
                for c in contrasts:
                    s = sets[c]
                    tr = np.concatenate([s["train_pos"], s["train_neg"]]); ytr = np.r_[np.ones(len(s["train_pos"])), np.zeros(len(s["train_neg"]))]
                    ev = np.concatenate([s["eval_pos"], s["eval_neg"]]); yev = np.r_[np.ones(len(s["eval_pos"])), np.zeros(len(s["eval_neg"]))]
                    probe = fit_probe(X[tr], ytr, C=an.probe_C, max_iter=an.probe_max_iter, seed=an.seed)
                    res["contrasts"][c]["aggregations"][a]["probe"][str(layer)] = eval_probe(probe, X[ev], yev)
                print(f"  probes [{a}] layer {layer} done ({time.time() - t0:.0f}s)")

    # Direction comparisons
    for p in an.cosine_pairs:
        c1, c2 = p
        if c1 in contrasts and c2 in contrasts:
            res["cosine_pairs"][f"{c1}|{c2}"] = {a: [cosine(dirs[c1][a]["direction"][l], dirs[c2][a]["direction"][l]) for l in range(L1)] for a in aggs}
    for c in contrasts:
        res["aggregation_cosines"][c] = {}
        for i, a1 in enumerate(aggs):
            for a2 in aggs[i + 1:]:
                res["aggregation_cosines"][c][f"{a1}|{a2}"] = [cosine(dirs[c][a1]["direction"][l], dirs[c][a2]["direction"][l]) for l in range(L1)]

    # Write directions
    out = args.out
    (out / "directions").mkdir(parents=True, exist_ok=True)
    for c, spec in contrasts.items():
        for a in aggs:
            d = dirs[c][a]
            u = np.stack([unit(d["direction"][l]) for l in range(L1)])
            save_file(
                {"direction": d["direction"], "direction_unit": u.astype(np.float32), "mean_pos": d["mean_pos"], "mean_neg": d["mean_neg"],
                 "norm": np.linalg.norm(d["direction"], axis=1).astype(np.float32)},
                str(out / "directions" / f"{c}__{a}.safetensors"),
            )
            meta = {
                "model": reader.config.get("model"), "model_revision": reader.config.get("model_revision"), "cache_dir": str(args.cache_dir),
                "manifest_id": cfg.manifest_id, "contrast": c, "positive": spec.positive, "negative": spec.negative,
                "positive_max_turns": spec.positive_max_turns, "negative_max_turns": spec.negative_max_turns,
                "n_pos": int(len(sets[c]["train_pos"])), "n_neg": int(len(sets[c]["train_neg"])), "fitted_on": "train split (by task id)",
                "split_seed": cfg.split.seed, "aggregation": a, "layer_index_convention": reader.config.get("layer_index_convention"),
                "winsorization_quantile": an.winsorization_quantile, "shape": [L1, H], "tensor_layout": "(layer index, hidden); direction = mean_pos - mean_neg",
                "best_layer_by_eval_proj_auc": res["contrasts"][c]["aggregations"][a]["best_layer"],
            }
            (out / "directions" / f"{c}__{a}.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "results.md").write_text(render_markdown(res, contrasts, aggs, an.sample_sizes))
    print(f"wrote {out}/results.md, results.json and {len(contrasts) * len(aggs)} direction files ({time.time() - t0:.0f}s)")
    return 0


def render_markdown(res: dict, contrasts: dict, aggs: list[str], sample_sizes: list[int]) -> str:
    L1 = res["n_layers_plus1"]
    lines = [
        f"# Stage 4 directions: `{res['manifest_id']}`",
        "",
        f"Model `{res['model']}` @ `{res['model_revision']}`; rows used {res['n_rows_used']} (subset_per_class={res['subset_per_class']}); "
        f"layer indices 0..{L1 - 1} ({res['layer_index_convention']}).",
        "",
        "Class counts (train/eval): " + ", ".join(f"{k} {v['train']}/{v['eval']}" for k, v in res["class_counts"].items()),
        "",
        "## Best layer per contrast and aggregation (eval-split, held-out tasks)",
        "",
        "| contrast | agg | best layer | proj AUROC | length-matched | probe AUROC | probe bal. acc. | covariate-only AUROC | split-half cos | dir. norm |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in contrasts:
        C = res["contrasts"][c]
        for a in aggs:
            R = C["aggregations"][a]; L = R["best_layer"]
            pr = R["probe"].get(str(L), {})
            sh = R["split_half_cos_mean"][L] if R["split_half_cos_mean"] else None
            lines.append(
                f"| {c} | {a} | {L} | {fmt(R['proj_auc'][L])} | {fmt(R['proj_auc_length_matched'][L])} | {fmt(pr.get('auc'))} | {fmt(pr.get('balanced_accuracy'))} | "
                f"{fmt(res['covariate_baseline'][c]['auc'])} | {fmt(sh)} | {R['direction_norm'][L]:.1f} |"
            )
    lines += ["", "Contrasts: " + "; ".join(f"`{c}` = {res['contrasts'][c]['positive']} vs {res['contrasts'][c]['negative']} (train {res['contrasts'][c]['n_train_pos']}/{res['contrasts'][c]['n_train_neg']}, eval {res['contrasts'][c]['n_eval_pos']}/{res['contrasts'][c]['n_eval_neg']})" for c in contrasts)]

    step = max(1, (L1 - 1) // 16)
    layer_cols = list(range(0, L1, step))
    if layer_cols[-1] != L1 - 1:
        layer_cols.append(L1 - 1)
    hdr = "| contrast | agg | " + " | ".join(f"L{l}" for l in layer_cols) + " |"
    sep = "|---|---|" + "---|" * len(layer_cols)
    lines += ["", "## Projection AUROC by layer (eval split)", "", hdr, sep]
    for c in contrasts:
        for a in aggs:
            R = res["contrasts"][c]["aggregations"][a]
            lines.append(f"| {c} | {a} | " + " | ".join(fmt(R["proj_auc"][l]) for l in layer_cols) + " |")

    if res["cosine_pairs"]:
        lines += ["", "## Cosine similarity between contrast directions, by layer", "", "| pair | agg | " + " | ".join(f"L{l}" for l in layer_cols) + " | split-half floor (each) |", sep + "---|"]
        for pair, per_agg in res["cosine_pairs"].items():
            c1, c2 = pair.split("|")
            for a in aggs:
                floor = " / ".join(fmt(res["contrasts"][c]["aggregations"][a]["split_half_cos_mean"][res["contrasts"][c]["aggregations"][a]["best_layer"]]) if res["contrasts"][c]["aggregations"][a]["split_half_cos_mean"] else "-" for c in (c1, c2))
                lines.append(f"| {c1} vs {c2} | {a} | " + " | ".join(fmt(per_agg[a][l]) for l in layer_cols) + f" | {floor} |")

    lines += ["", "## Cosine between aggregations of the same contrast (at that contrast's best layer per first aggregation)", ""]
    for c in contrasts:
        L = res["contrasts"][c]["aggregations"][aggs[0]]["best_layer"]
        lines.append(f"- {c} (layer {L}): " + ", ".join(f"{k} {fmt(v[L])}" for k, v in res["aggregation_cosines"][c].items()))

    any_transfer = any(res["contrasts"][c]["aggregations"][a]["transfer_auc"] for c in contrasts for a in aggs)
    if any_transfer:
        lines += ["", "## Transfer: direction from one contrast scored on the other's eval rows (AUROC at the source's best layer / best over layers)", "", "| source | target | agg | AUROC at source best layer | best transfer AUROC (layer) |", "|---|---|---|---|---|"]
        for c in contrasts:
            for a in aggs:
                R = res["contrasts"][c]["aggregations"][a]
                for dst, arr in R["transfer_auc"].items():
                    arr2 = np.array(arr, dtype=float); arr2 = np.where(np.isnan(arr2), -1, arr2)
                    lines.append(f"| {c} | {dst} | {a} | {fmt(arr[R['best_layer']])} | {fmt(float(arr2.max()))} (L{int(arr2.argmax())}) |")

    lines += ["", "## Where the classes fall along each direction (mean normalised projection at the best layer: 0 = negative mean, 1 = positive mean)", ""]
    classes = list(res["class_counts"])
    lines += ["| contrast | agg | " + " | ".join(classes) + " |", "|---|---|" + "---|" * len(classes)]
    for c in contrasts:
        for a in aggs:
            R = res["contrasts"][c]["aggregations"][a]; L = R["best_layer"]
            lines.append(f"| {c} | {a} | " + " | ".join(fmt(R["class_projection_mean"][k][L]) for k in classes) + " |")

    if sample_sizes and any(res["contrasts"][c]["aggregations"][a]["sample_curve"][str(sample_sizes[0])]["cos_mean"] for c in contrasts for a in aggs):
        lines += ["", "## Sample-size curve at the best layer (n per class; cosine to the full-train direction, eval projection AUROC; mean ± std over draws)", "",
                  "| contrast | agg | " + " | ".join(f"n={n}" for n in sample_sizes) + " | full |", "|---|---|" + "---|" * (len(sample_sizes) + 1)]
        for c in contrasts:
            for a in aggs:
                R = res["contrasts"][c]["aggregations"][a]; L = R["best_layer"]
                cells = []
                for n in sample_sizes:
                    S = R["sample_curve"][str(n)]
                    if S["cos_mean"]:
                        cells.append(f"cos {S['cos_mean'][L]:.2f}±{S['cos_std'][L]:.2f}, AUC {S['auc_mean'][L]:.3f}±{S['auc_std'][L]:.3f}")
                    else:
                        cells.append("-")
                lines.append(f"| {c} | {a} | " + " | ".join(cells) + f" | AUC {fmt(R['proj_auc'][L])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
