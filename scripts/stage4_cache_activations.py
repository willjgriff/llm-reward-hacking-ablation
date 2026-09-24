#!/usr/bin/env python
"""Stage 4, step 2: build the activation cache (GPU; one forward pass per trajectory).

Reads the manifest, replays every trajectory's final sequence (prompt + completion of the last call, which
contains the whole rollout verbatim) through the model, and stores per-layer aggregated residuals for each
aggregation (rhablation.stage4_masks) as memory-mapped arrays (rhablation.stage4_cache). Resumable: rows
already in <cache_dir>/index.jsonl are skipped. Processes the manifest in its order, in chunks, so the
balanced subset is complete early.

  python scripts/stage4_cache_activations.py --config configs/stage4_terminal_verifier.yaml \
      --manifest data/stage4/manifest/tv27b/manifest.jsonl --cache-dir data/stage4/cache/tv27b [--limit 10 --sanity-check] [--device cuda]

Runs as root on the box from the torch env (no model-written code is executed here; the weights are in root's
HF cache). --sanity-check compares the hooks with output_hidden_states, tests batch invariance and the
teacher-forced NLL of the stored completions before any chunk is written; its report goes to
<cache_dir>/sanity_check.json.
"""

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhablation.compact import open_trajectories  # noqa: E402
from rhablation.schema import Trajectory  # noqa: E402
from rhablation.stage4_cache import LAYER_INDEX_CONVENTION, CacheWriter, sha256_file  # noqa: E402
from rhablation.stage4_config import load_stage4_config  # noqa: E402
from rhablation.stage4_manifest import ManifestRow, final_sequence, read_manifest  # noqa: E402
from rhablation.stage4_masks import aggregation_weights, token_spans  # noqa: E402


def load_sequences(rows: list[ManifestRow], root: Path) -> dict[int, tuple[np.ndarray, list[tuple[int, int]]]]:
    """order -> (final sequence ids as int32, per-call (prompt_len, completion_len)); one pass per source file."""
    wanted: dict[str, dict[int, ManifestRow]] = defaultdict(dict)
    for r in rows:
        wanted[r.source_file][r.line_no] = r
    out = {}
    for source, by_line in wanted.items():
        with open_trajectories(root / source) as f:
            for line_no, line in enumerate(f, 1):
                r = by_line.get(line_no)
                if r is None:
                    continue
                rec = Trajectory.model_validate_json(line)
                if rec.trajectory_id != r.trajectory_id:
                    raise ValueError(f"{source}:{line_no}: {rec.trajectory_id} != manifest {r.trajectory_id}")
                ids, lengths = final_sequence(rec)
                if len(ids) != r.seq_len:
                    raise ValueError(f"{r.trajectory_id}: seq_len {len(ids)} != manifest {r.seq_len}")
                out[r.order] = (np.asarray(ids, dtype=np.int32), lengths)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="only the first N manifest rows (smoke test)")
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", help="override the config's model (e.g. a local path for tests)")
    ap.add_argument("--dtype", help="override the config's dtype")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--stop-after-chunks", type=int, help="exit after this many chunks (resume test)")
    ap.add_argument("--chunk-trajectories", type=int, help="override cache.chunk_trajectories")
    ap.add_argument("--no-nll-gate", action="store_true", help="report the teacher-forced NLL check but do not fail on it (random-weight tests)")
    args = ap.parse_args()

    import torch

    from rhablation.stage4_activations import (
        ResidualReducer, forward_reduce, hidden_states_reference, load_model, make_batches, pad_batch, teacher_forced_nll, weights_tensor,
    )

    t0 = time.time()
    cfg = load_stage4_config(args.config)
    model_id = args.model or cfg.model
    dtype = args.dtype or cfg.dtype
    rows_all = read_manifest(args.manifest)
    rows = rows_all[: args.limit] if args.limit else rows_all
    aggs = cfg.cache.aggregations
    print(f"manifest {args.manifest}: {len(rows_all)} rows, using {len(rows)}; aggregations {aggs}")

    print(f"loading {model_id} @ {cfg.model_revision} ({dtype}) on {args.device} ...", flush=True)
    parts = load_model(model_id, cfg.model_revision if not args.model else None, dtype, device_map=args.device, attn_implementation=cfg.cache.attn_implementation)
    L1 = parts.n_layers + 1
    print(f"model loaded in {time.time() - t0:.0f}s: {parts.n_layers} layers, hidden {parts.hidden}, dtype {parts.dtype}, device {parts.device}", flush=True)
    versions = {"torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import fla  # noqa: F401

        versions["flash_linear_attention"] = getattr(fla, "__version__", "present")
    except ImportError:
        versions["flash_linear_attention"] = None
    print(f"versions: {versions}")

    cache_config = {
        "model": model_id, "model_revision": cfg.model_revision, "dtype": dtype, "storage_dtype": cfg.cache.storage_dtype,
        "n_rows": len(rows_all), "n_layers_plus1": L1, "hidden": parts.hidden, "aggregations": aggs,
        "layer_index_convention": LAYER_INDEX_CONVENTION, "think_open_id": cfg.think_open_id, "think_close_id": cfg.think_close_id,
        "im_end_id": cfg.im_end_id, "manifest_sha256": sha256_file(args.manifest), "versions": versions, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if not (args.cache_dir / "manifest.jsonl").exists():
        shutil.copy(args.manifest, args.cache_dir / "manifest.jsonl")
    writer = CacheWriter(args.cache_dir, cache_config, len(rows_all), L1, parts.hidden)
    done = writer.done_orders()
    todo = [r for r in rows if r.order not in done]
    print(f"{len(done)} rows already cached, {len(todo)} to do", flush=True)
    if not todo:
        return 0

    print("loading token sequences ...", flush=True)
    seqs = load_sequences(todo, args.root)
    print(f"{len(seqs)} sequences, {sum(len(s[0]) for s in seqs.values()) / 1e6:.2f}M tokens ({time.time() - t0:.0f}s)", flush=True)

    def spans_and_weights(r: ManifestRow):
        ids, lengths = seqs[r.order]
        spans = token_spans(ids.tolist(), lengths, cfg.think_open_id, cfg.think_close_id, cfg.im_end_id, cfg.newline_id)
        return spans, aggregation_weights(spans, aggs)

    reducer = ResidualReducer(parts, len(aggs))
    rng = np.random.default_rng(0)

    if args.sanity_check:
        report = run_sanity_check(parts, reducer, todo, seqs, spans_and_weights, aggs, cfg, args)
        (args.cache_dir / "sanity_check.json").write_text(json.dumps(report, indent=1) + "\n")
        print("sanity check:", json.dumps(report, indent=1))
        if not report["all_passed"]:
            print("SANITY CHECK FAILED; nothing written", file=sys.stderr)
            return 2

    tokens_done, rows_done, chunks_done = 0, 0, 0
    t_start = time.time()
    chunk_size = args.chunk_trajectories or cfg.cache.chunk_trajectories
    nll_rate = 1.0 if args.limit else cfg.cache.nll_sample_rate
    for c0 in range(0, len(todo), chunk_size):
        chunk = todo[c0 : c0 + chunk_size]
        by_order = {r.order: r for r in chunk}
        batches = make_batches([(r.order, r.seq_len) for r in chunk], cfg.cache.max_tokens_per_batch, cfg.cache.max_batch_size)
        values = {a: np.zeros((len(chunk), L1, parts.hidden), np.float32) for a in aggs}
        pos_in_chunk = {r.order: i for i, r in enumerate(chunk)}
        entries = []
        t_chunk = time.time()
        for batch in batches:
            brows = [by_order[o] for o in batch]
            seq_list = [seqs[o][0].tolist() for o in batch]
            sw = [spans_and_weights(r) for r in brows]
            ids, mask = pad_batch(seq_list)
            W = weights_tensor([w for _, w in sw], aggs, ids.shape[1])
            buffer, max_abs, last_hidden = forward_reduce(parts, reducer, ids, mask, W)
            if float(max_abs.max()) > cfg.cache.max_abs_activation:
                print(f"ABORT: |activation| {float(max_abs.max()):.3g} exceeds max_abs_activation={cfg.cache.max_abs_activation} (storage {cfg.cache.storage_dtype}); switch storage_dtype to float32", file=sys.stderr)
                return 3
            nll_idx = [i for i in range(len(brows)) if rng.random() < nll_rate]
            nll = {}
            if nll_idx:
                res = teacher_forced_nll(parts, last_hidden, ids, [sw[i][0].completion_positions() if i in nll_idx else [] for i in range(len(brows))])
                nll = {i: res[i] for i in nll_idx}
            del last_hidden
            for i, r in enumerate(brows):
                spans = sw[i][0]
                for a_i, a in enumerate(aggs):
                    values[a][pos_in_chunk[r.order]] = buffer[:, i, a_i, :].numpy()
                entries.append({
                    "order": r.order, "trajectory_id": r.trajectory_id, "chunk": chunks_done, "seq_len": r.seq_len, "n_calls": spans.n_calls,
                    "n_cot_tokens": spans.n_cot_tokens(), "n_answer_tokens": spans.n_answer_tokens(),
                    "max_abs_by_layer": [round(float(x), 2) for x in max_abs.tolist()],
                    "nll": nll.get(i), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                tokens_done += r.seq_len
        writer.write_rows([r.order for r in chunk], values)
        writer.append_index(entries)
        rows_done += len(chunk)
        chunks_done += 1
        elapsed = time.time() - t_start
        tok_s = tokens_done / max(elapsed, 1e-6)
        remaining = sum(r.seq_len for r in todo[c0 + chunk_size :])
        progress = {
            "rows_done_this_run": rows_done, "rows_cached_total": len(done) + rows_done, "rows_in_manifest": len(rows_all), "chunks_done": chunks_done,
            "tokens_done": tokens_done, "elapsed_s": round(elapsed), "tokens_per_s": round(tok_s), "eta_s": round(remaining / max(tok_s, 1e-6)),
            "last_chunk_s": round(time.time() - t_chunk, 1), "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        writer.write_progress(progress)
        nlls = [e["nll"]["nll"] for e in entries if e.get("nll")]
        print(f"chunk {chunks_done}: {len(chunk)} rows in {time.time() - t_chunk:.0f}s; {tok_s:.0f} tok/s; cached {len(done) + rows_done}/{len(rows_all)}; "
              f"ETA {progress['eta_s'] / 60:.0f} min; max|h| {float(max_abs.max()):.0f}; nll mean {np.mean(nlls) if nlls else float('nan'):.3f} (n={len(nlls)})", flush=True)
        if args.stop_after_chunks and chunks_done >= args.stop_after_chunks:
            print("stopping after", chunks_done, "chunks (--stop-after-chunks)")
            break
    reducer.remove()
    print(f"done: {rows_done} rows, {tokens_done / 1e6:.2f}M tokens in {(time.time() - t_start) / 60:.1f} min")
    return 0


def run_sanity_check(parts, reducer, todo, seqs, spans_and_weights, aggs, cfg, args) -> dict:
    import torch

    from rhablation.stage4_activations import ResidualReducer, forward_reduce, hidden_states_reference, pad_batch, teacher_forced_nll, weights_tensor

    report: dict = {}
    by_len = sorted(todo, key=lambda r: r.seq_len)
    short, long_ = by_len[0], by_len[-1]
    L1 = parts.n_layers + 1

    # (a) hooks vs output_hidden_states on the shortest sequence, at three positions (one-hot weights)
    ids, mask = pad_batch([seqs[short.order][0].tolist()])
    T = ids.shape[1]
    positions = [0, T // 2, T - 1]
    W = torch.zeros((1, len(positions), T))
    for i, p in enumerate(positions):
        W[0, i, p] = 1.0
    # A second reducer with one-hot "aggregations" at those positions; the main reducer's hooks are inert
    # (no batch set) while it runs.
    reducer_tmp = ResidualReducer(parts, len(positions))
    buffer, _, _ = forward_reduce(parts, reducer_tmp, ids, mask, W)
    reducer_tmp.remove()
    ref = hidden_states_reference(parts, ids, mask)
    diffs = []
    for layer in range(L1):
        for i, p in enumerate(positions):
            a, b = buffer[layer, 0, i], ref[layer][0, p].cpu()
            diffs.append(float((a - b).abs().max() / (b.abs().max() + 1e-6)))
    report["hooks_vs_output_hidden_states"] = {"n_hidden_states": len(ref), "expected": L1, "max_rel_diff": max(diffs), "passed": len(ref) == L1 and max(diffs) < 1e-3}

    # (b) batch invariance: shortest alone vs shortest + longest padded together
    def reduce_rows(rows_):
        seq_list = [seqs[r.order][0].tolist() for r in rows_]
        sw = [spans_and_weights(r) for r in rows_]
        ids_, mask_ = pad_batch(seq_list)
        W_ = weights_tensor([w for _, w in sw], aggs, ids_.shape[1])
        buf, max_abs, last = forward_reduce(parts, reducer, ids_, mask_, W_)
        return buf, max_abs, last, ids_, sw

    alone, _, _, _, sw_alone = reduce_rows([short])

    # (a2) aggregations of the shortest sequence equal a naive mean over the reference hidden states
    spans_short = sw_alone[0][0]
    naive_pos = {"cot_mean": spans_short.cot_positions(), "answer_mean": spans_short.answer_positions(),
                 "prompt_last": spans_short.prompt_last_positions(), "completion_last": spans_short.completion_last_positions()}
    agg_diffs = {}
    for a_i, a in enumerate(aggs):
        pos = naive_pos[a]
        if not pos:
            continue
        d = []
        for layer in range(L1):
            naive = ref[layer][0, pos].mean(dim=0).cpu()
            d.append(float((alone[layer, 0, a_i] - naive).abs().max() / (naive.abs().max() + 1e-6)))
        agg_diffs[a] = max(d)
    report["aggregations_vs_naive"] = {"max_rel_diff": agg_diffs, "n_positions": {a: len(p) for a, p in naive_pos.items()}, "passed": all(v < 1e-3 for v in agg_diffs.values())}
    both, max_abs, last, ids_b, sw_b = reduce_rows([short, long_])
    rel = float((alone[:, 0] - both[:, 0]).abs().max() / (alone[:, 0].abs().max() + 1e-6))
    report["batch_invariance"] = {"short_len": short.seq_len, "long_len": long_.seq_len, "max_rel_diff": rel, "passed": rel < 1e-2}

    # (c) teacher-forced NLL of the stored completions, plus a negative control (another trajectory's final completion)
    positions_list = [sw_b[i][0].completion_positions() for i in range(2)]
    nll = teacher_forced_nll(parts, last, ids_b, positions_list)
    del last
    # control: prompt of `short` + final completion of `long_`
    sp_short, sp_long = sw_b[0][0], sw_b[1][0]
    p_len = sp_short.turns[-1].prompt_len
    comp_long = seqs[long_.order][0][sp_long.turns[-1].start : sp_long.turns[-1].end].tolist()
    ctrl = seqs[short.order][0][:p_len].tolist() + comp_long
    ids_c, mask_c = pad_batch([ctrl])
    Wc = torch.zeros((1, len(aggs), ids_c.shape[1]))
    _, _, last_c = forward_reduce(parts, reducer, ids_c, mask_c, Wc)
    ctrl_nll = teacher_forced_nll(parts, last_c, ids_c, [list(range(p_len, len(ctrl)))])[0]
    del last_c
    nll_ok = all(x["nll"] < 1.5 and x["top1"] >= 0.5 for x in nll) and ctrl_nll["nll"] > max(x["nll"] for x in nll) + 1.0
    report["teacher_forced_nll"] = {
        "short": nll[0], "long": nll[1], "negative_control": ctrl_nll, "criteria": "nll < 1.5 and top1 >= 0.5 on both; control nll > best + 1",
        "gate": not args.no_nll_gate, "passed": nll_ok or args.no_nll_gate, "criteria_met": nll_ok,
    }
    report["max_abs_activation"] = {"value": float(max_abs.max()), "limit": cfg.cache.max_abs_activation, "passed": float(max_abs.max()) < cfg.cache.max_abs_activation}
    report["all_passed"] = all(v["passed"] for k, v in report.items() if isinstance(v, dict) and "passed" in v)
    return report


if __name__ == "__main__":
    sys.exit(main())
