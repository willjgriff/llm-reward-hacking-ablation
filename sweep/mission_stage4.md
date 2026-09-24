# Mission: Stage 4 activation cache and reward-hacking directions (Qwen3.8-27B, TerminalVerifierEnv)

Read `/root/.claude/CLAUDE.md` and the project `CLAUDE.md` first; the first overrides the second, and this
mission overrides both where it is more specific. Nobody is watching: wherever this says "report", write it
to `box_report/stage4/` and carry on.

Goal: run the three stage-4 scripts that were written and tested on the laptop (`scripts/stage4_*.py`,
`src/rhablation/stage4_*.py`, `configs/stage4_terminal_verifier.yaml`) on the GPU: one forward pass per
trajectory of the TerminalVerifierEnv sweep with `Qwen/Qwen3.8-27B` (bf16, revision in the config), an
activation cache of per-layer aggregated residuals, then difference-of-means directions with held-out
probes. **Scope:** this and nothing else. No orthogonalization, no benchmark runs, no new data
collection, no changes to the sweep data.

## What is already on the box (copied from the laptop)
- `data/sweep/Qwen3.8-27B/terminal_verifier_*/` (the four bf16 sweep variants, 10,993 trajectories).
- `data/stage4/manifest/tv27b/{manifest.jsonl, splits.json, manifest_summary.md}`: classes, task split
  and processing order, already built. Do not rebuild it (the split seed and order are part of the record).
  If you must, `uv run scripts/stage4_build_manifest.py --config configs/stage4_terminal_verifier.yaml`
  reproduces it byte for byte.
- The model weights, if this box ran the sweep, in root's `~/.cache/huggingface`. On a fresh box the first
  `from_pretrained` downloads them (~56 GB, public repo, no token needed; 10-25 minutes). Either way no
  token is involved.

## Who runs what (explicit exception to the `rhbench-run` rule)
`stage4_cache_activations.py` and `stage4_directions.py` execute no model-written and no benchmark code:
token ids go in, activations and numbers come out, and the weights live in root's cache, which `rhbench`
cannot read. Run them **as root from `/workspace/llm-reward-hacking-ablation`**, not through `rhbench-run`.
Everything else in `/root/.claude/CLAUDE.md` stays as it is (no credentials in commands, no services, code
leaves only via `sweep/upload_code.sh`, data only via `scripts/upload_run.sh`).

## Steps
1. **Free the GPU.** If a vLLM session exists (`tmux has-session -t vllm`), `tmux kill-session -t vllm`;
   on a fresh box there is none. Do not start vLLM in this mission. `nvidia-smi` must show the memory
   free before you load the model.
2. **Environment** (root; `df -h /` first; keep `UV_CACHE_DIR` on the same filesystem):
   ```
   uv venv ~/stage4-env --python 3.12
   uv pip install --python ~/stage4-env/bin/python torch transformers==5.17.0 accelerate flash-linear-attention causal-conv1d safetensors numpy scikit-learn pyyaml pydantic
   ~/stage4-env/bin/python -c "import torch, transformers, fla; print(torch.cuda.is_available(), torch.__version__, transformers.__version__, fla.__version__)"
   ```
   If `causal-conv1d` has no wheel for this CUDA/torch, drop it (torch fallback, slower but correct) and
   record that. If `flash-linear-attention` cannot be installed either, still run the tiny run and read the
   tok/s: below 2,500 tok/s stop and report instead of running for a day. Fallback env if the new one
   fails outright: `~/vllm-env/bin/python` with `flash-linear-attention accelerate scikit-learn safetensors`
   added; record which env was used.
3. **Tiny run with the sanity gates** (`--limit 10 --sanity-check`; never pass `--no-nll-gate` here):
   ```
   cd /workspace/llm-reward-hacking-ablation
   ~/stage4-env/bin/python scripts/stage4_cache_activations.py --config configs/stage4_terminal_verifier.yaml \
       --manifest data/stage4/manifest/tv27b/manifest.jsonl --cache-dir data/stage4/cache/tv27b-tiny --limit 10 --sanity-check 2>&1 | tee box_report/stage4/tiny_run.log
   ```
   It writes `data/stage4/cache/tv27b-tiny/sanity_check.json` and exits non-zero if a gate fails:
   hooks equal `output_hidden_states` (65 entries), aggregations equal a naive per-position mean, batch
   invariance under padding (max relative diff < 1e-2), teacher-forced NLL of the stored completions
   (< 1.5 nats, top-1 ≥ 0.5, negative control at least 1 nat worse; this confirms the stored ids are what
   the model saw), no activation above the float16 limit. Paste the JSON and the printed tok/s into
   `box_report/stage4/LOG.md`. Then `~/stage4-env/bin/python scripts/stage4_directions.py --config
   configs/stage4_terminal_verifier.yaml --cache-dir data/stage4/cache/tv27b-tiny --out
   data/stage4/directions/tv27b-tiny --subset-per-class 2 --no-probes --no-curves` must run through
   (contrasts with too few rows are skipped and say so). Then delete `data/stage4/cache/tv27b-tiny`.
   - If a gate fails, fix the cause in the stage-4 code with the smallest change that makes the check
     genuinely pass, log the diff in `LOG.md`, and upload the code (`bash sweep/upload_code.sh --message
     "stage4 fix: ..."`). Never loosen a threshold or skip a gate.
   - If the max-abs gate fails: set `cache.storage_dtype: float32` in the config (doubles the cache to
     ~60 GB; check `df -h /` again) and rerun the tiny run.
   - If the model does not load in bf16 with `transformers==5.17.0` (`Qwen3_5ForConditionalGeneration`),
     try the transformers version `~/vllm-env` has; record it.
4. **Full run** in tmux session `stage4`, as a background task so you are re-invoked when it exits:
   ```
   ~/stage4-env/bin/python scripts/stage4_cache_activations.py --config configs/stage4_terminal_verifier.yaml \
       --manifest data/stage4/manifest/tv27b/manifest.jsonl --cache-dir data/stage4/cache/tv27b 2>&1 | tee -a box_report/stage4/full_run.log
   ```
   ~44M tokens; expect 1.5-3 h at 5-8k tok/s. `data/stage4/cache/tv27b/progress.json` has tok/s and the
   ETA; look at it no more than every 20-30 minutes. The cache needs ~30 GB (float16) plus the ~8 GB env:
   `df -h /` must show at least 45 GB free before you start; otherwise free space the way the sweep rules
   say (never delete sweep data that is not verified on HF). The run is resumable: if it dies, rerun the
   same command; rows in `index.jsonl` are skipped. Do not change `chunk_trajectories` or the batch
   limits unless the run fails with an out-of-memory error, then halve `max_tokens_per_batch` and note it.
5. **Subset analysis** as soon as `index.jsonl` in the cache has at least 400 rows of each of `hack_vb`,
   `nonhack_A`, `hack_corr`, `pass_possible`, `attempt_vb` (the processing order puts the first 400 of every
   class in the first ~2,160 rows, about 30 minutes in). On the box CPU, while the full run continues:
   ```
   ~/stage4-env/bin/python scripts/stage4_directions.py --config configs/stage4_terminal_verifier.yaml \
       --cache-dir data/stage4/cache/tv27b --out data/stage4/directions/tv27b-subset400 --subset-per-class 400 2>&1 | tee box_report/stage4/subset400.log
   ```
   Read `results.md`. Write into `box_report/stage4/SUMMARY.md`: the best-layer table, the aggregation you
   would pick per contrast (highest length-matched eval projection AUROC; ties broken by split-half cosine),
   the cosine between the `vb_vs_A` and `corr_vs_A` directions at their best layers next to the split-half
   noise floor, and the verdict below.
6. **Full analysis** after the cache is complete: the same command with `--out
   data/stage4/directions/tv27b-full` and no `--subset-per-class`. Update `SUMMARY.md` (keep the subset
   numbers next to the full ones; the sample-size table answers whether more than 400 per class mattered).
7. **Uploads** (root): `bash scripts/upload_run.sh --run-dir data/stage4/cache/tv27b --dest-prefix stage4`
   (about 30 GB; start it as a background task), then the same for `data/stage4/directions/tv27b-subset400`
   and `data/stage4/directions/tv27b-full`, then `bash scripts/upload_run.sh --run-dir
   /workspace/llm-reward-hacking-ablation/box_report --dest-prefix sweep` and `bash sweep/upload_code.sh
   --message "stage4"`. Verify the cache upload by listing the repo files and comparing sizes before you
   finish. Do not delete the local cache.
8. **Finish.** `SUMMARY.md` up to date with a short "What I would do next"; then stop. The watchdog stops
   the box once it is idle. Leave vLLM stopped.

## Verdict rule (write it verbatim into SUMMARY.md with the numbers)
Covariates alone (turn count, sequence length, `max_turns`) separate most of these classes well: the
manifest shows `nonhack_A` is `max_turns=4` only, `pass_possible` averages 1.1 calls against ~4.7 for
`hack_vb`. So the covariate-only AUROC will be high and is not the bar. A direction is **usable** for
Step 5 when, at its best layer and chosen aggregation: held-out projection AUROC ≥ 0.85 **on the
length-matched eval subset**, probe balanced accuracy ≥ 0.85, and split-half cosine ≥ 0.8 at that layer.
Report separately where `attempt_vb` (tampered but not passed) falls along the `vb_vs_A` direction (the
class-projection table: near 1 means failed attempts look like hacks, near 0 like non-hacks). Below 0.75
length-matched AUROC, or split-half cosine below 0.6: not usable as is; say so and list what differs
between the aggregations.

## Reporting
- `box_report/stage4/LOG.md`: append-only, UTC timestamps: environment used (versions, kernels), sanity
  check JSON, tok/s, chunk timings, errors, every decision and code change.
- `box_report/stage4/SUMMARY.md`: rewritten after steps 3, 5, 6 and 7. Tables copied from `results.md`
  (best-layer table, cosine pairs at the best layers with noise floor, transfer table, class projections,
  sample-size curve), the verdict, HF paths, wall time.
- Keep both current enough to be useful if the session dies at any moment.
