# Project: Orthogonalization against reward hacking

## Research question
Does orthogonalization (abliteration, as in heretic/obliteratus) remove reward hacking from production LLMs better than DPO, along two axes:
- **Tolerance to low-quality training data**: directions extracted from simple/toy reward hacks still remove complex/realistic ones.
- **Sample efficiency**: works with few examples.

General capability degradation must always be measured, and orthogonalization should be done in a way that minimises it.

## Key framing (do not deviate without asking)
- We remove reward hacking from **production models after all training** (e.g. Qwen/Qwen3.5-4B). We do NOT train a model to reward hack first.
- Orthogonalization and DPO remove **propensities, not capabilities**. Do not build experiments testing whether reward hacking can be fine-tuned back in.
- Off-policy data is expected to be fine for direction extraction. Only switch to on-policy if results show it's needed.
- Follow what heretic (or the chosen library) does for refusal by default: layer choice, difference-in-means, hyperparameter search, capability-preserving steps. Do not invent novel ablation techniques unless explicitly asked.
- Until the minimal viable version is finished, no extra degradation mitigations beyond what the library already does.
- Small models (4B–8B) first for iteration speed; scale up later.

## Minimal viable version
- 2–4 training datasets of paired rollouts (reward hack vs no reward hack), varying in size, realism, and complexity.
- 2–4 reward hacking evals, varying in realism, complexity, and egregiousness.
- A well-tuned orthogonalization recipe and an **equally well-tuned** DPO recipe. An under-tuned baseline invalidates the comparison.
- Output: eval results for each dataset × {orthogonalization, DPO}.

## Current phase: basic end-to-end orthogonalization setup
1. **Harness**: one small model running ImpossibleBench via vLLM, saving trajectories in a fixed schema.
2. **Rollouts at scale**: target roughly 10× the dataset size heretic/obliteratus use for refusal, reduced if direction computation would take more than about an hour. Report the hack rate.
3. **Labelling**: programmatic label where available, plus an LLM judge for reward hacking, plus a separate check of the CoT for whether the hack appears intentional or accidental.
4. **Activation extraction and direction**: difference in means over hack vs no-hack trajectories. Validate with a held-out linear probe before ablating.
5. **Ablation**: heretic/obliteratus adapted to use our direction. Check coherence and KL on benign prompts.
6. **Evals**: rerun the same reward hacking benchmark on held-out tasks; basic capability eval (MMLU via lm-eval-harness).
7. **Repeat** with other data/benchmarks; compare directions (e.g. cosine similarity per layer).

## Design decisions (living log: add each approved decision here)
- Train/eval split is **by task ID**. Tasks used for direction extraction never appear in evaluation.
- Capability evals must be unhackable, otherwise reduced reward hacking looks like capability loss.
- Trajectory schema: `src/rhablation/schema.py` (`SCHEMA_VERSION`), one JSON object per line in `data/stage1/<run_id>/trajectories.jsonl`. `rendered_text`/token IDs are stored **per assistant turn** because Qwen3.5's chat template drops `<think>` blocks from assistant turns before the latest user message, so the whole-trajectory string is not what the model saw. Later stages use stored token IDs directly (decode→re-encode is not identity).
- ImpossibleBench subset and variants used: LiveCodeBench (`fjzzq2002/impossible_livecodebench`, 103 tasks/split), splits `original`/`oneoff`/`conflicting`, scaffolds `minimal` (text submission loop, no tool calls) and `tools` (bash/python/text_editor via Inspect `basic_agent`). Paper-replication prompt (prompt D of the ImpossibleBench paper's prompt ablation, arXiv 2510.20270), `allow_test_modifications=True`, `max_attempts=5` (user decision 2026-09-17; package default is 5, its example script uses 10; runs `smoke1`, `smoke2`, `promptB-min-16`, `promptB-tools-smoke` used 10), `message_limit=50`. `programmatic_hack` = test pass on `oneoff`/`conflicting`; `null` on `original`. Caveat: in impossiblebench 0.1.0 `allow_test_modifications` only changes the prompt suffix; the tools scorer always restores the original `test.py` before scoring, so a test-editing hack never passes and `programmatic_hack` cannot see it (observed in `promptB-tools-smoke`, `lcbhard_41` oneoff).
- Hack signals are reported separately, never merged (user decision 2026-09-18): (1) `programmatic_hack` = pass on an impossible variant (the paper's cheating definition); (2) `test_edit_attempt` (schema 1.1, tools scaffold only) = a tool call wrote to `test.py`, detected from the trajectory by `src/rhablation/hack_signals.py` because the scorer's own `test_modification_detected` flag is only set when the edit survives to final scoring; (3) failed special-casing, flagged heuristically by `scripts/stage1_hack_attempts.py` and confirmed by hand (later: the Step 3 judge). Editing a test counts as a hack attempt even when the edit corrects the wrong expected value (the model was told not to modify tests); whether such cases belong in training labels is pending the mentor. Shuffle uses the benchmark's `seed=42`, which selects identical task IDs on every split.
- Ablation library: heretic (≥1.3.0 handles Qwen3.5's hybrid GatedDeltaNet/attention layers). obliteratus fail-closes on every Qwen3.5 size except Qwen3.8-27B — not usable.
- **Propensity, not forced hacking** (mentor's intent; details pending mentor confirmation): direction-extraction data and the headline before/after eval use solvable (`original`) tasks only, judge-scored on held-out task IDs. On `oneoff`/`conflicting` hacking is the only way to pass, so those splits serve as a labelled test bed for building/validating the LLM judge (pass = verified hack) and at most as a secondary "forced hacking" number. Open risks: base hack rate on `original` may be near zero; on-policy thinking rollouts are too slow for a one-hour data budget, so training transcripts are expected to be off-policy (source undecided).
- Model and revision: `Qwen/Qwen3.5-4B` (switched from 9B on 2026-09-17 for speed; same `Qwen3_5ForConditionalGeneration` hybrid architecture, main = `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`). Run `smoke1` used `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Revision is resolved at run time and written to `run_config.json`. Sampling: temp 0.6 / top_p 0.95 / top_k 20, `max_tokens` 32768 per turn (16384 truncated the CoT in 4 of 10 smoke calls). vLLM flags in `scripts/serve_vllm.sh` (`--reasoning-parser qwen3 --tool-call-parser qwen3_xml`, `--max-model-len 262144`).
- Off-box storage: finished runs are uploaded to a private Hugging Face dataset repo under `stage1/<run_id>/` (repo name in `HF_UPLOAD_REPO`, not hardcoded). The upload runs as **root** with a fine-grained token scoped to that one repo, stored in `/root/.config/rhablation/upload.env`; it must never go through `rhbench-run` or into the repo tree, which `rhbench` can read. Auto-stop is on by default for pipeline runs with a 30-minute idle window (user decision, 2026-09-17), is skipped when the upload did not succeed, and uses Vast "stop" (disk kept), never "destroy".
- Model-call timeout: `client_timeout_s: 7200`, passed to Inspect as `model_args={"client_timeout": ...}`. The OpenAI SDK default (600 s, 2 retries) abandoned every turn that generated for more than 10 minutes, cost 30 minutes each time, and resampled it. Runs `smoke1`, `smoke2`, `lcb-4b-part1`, `promptB-min-16` and `promptB-tools-smoke` were collected under that limit: their turns are biased toward short chains of thought (59 timed-out calls in `lcb-4b-part1`, 111 in `promptB-min-16`), and timed-out calls appear in `turns` with `error="Request timed out."`. `stage1_validate.py` now warns when any call timed out.
- Judge model and prompts: [fill in]
- Token positions and aggregation for activations over multi-turn completions: [fill in; this is a deliberate research decision, do not choose silently]
- Layers and ablation hyperparameters: [fill in]

## Stage 1 commands (GPU box)
- Full walkthrough: `setup.md`. Fresh box: rsync the working tree over, then `bash scripts/setup_box.sh` as root (creates unprivileged user `rhbench`, locks down `/etc/environment`, builds envs, starts vLLM in tmux `vllm`, runs an isolation self-check). Access is via the user's `ssh gpubox` alias; copy code with rsync, never via git commits.
- On the box, run every benchmark command through `rhbench-run <cmd>` (unprivileged user, scrubbed env).
- Env: `uv sync --python 3.12` (impossiblebench needs 3.12 syntax). vLLM lives in its own env (`~/vllm-env`) because it pins torch/CUDA.
- Serve: `bash scripts/serve_vllm.sh` (env: `MODEL`, `PORT`, `MAX_MODEL_LEN`, `MODEL_REVISION`; sets `VLLM_USE_FLASHINFER_SAMPLER=0` for Blackwell GPUs).
- Run: `uv run scripts/stage1_run_benchmark.py --config configs/stage1_lcb.yaml [--limit N --splits ... --agent-types ... --sandbox local|docker]`
- Progress of a live run: `uv run scripts/stage1_progress.py --run-dir data/stage1/<run_id> [--watch 60]` (reads `progress.jsonl`, written by the Inspect hook in `src/rhablation/progress.py`, plus live vLLM `/metrics`). All splits/scaffolds run in one Inspect call and share `max_connections`.
- Export: `uv run scripts/stage1_export_trajectories.py --run-dir data/stage1/<run_id>`
- Validate: `uv run scripts/stage1_validate.py --run-dir data/stage1/<run_id>`
- Hack-signal report (laptop, no GPU): `uv run scripts/stage1_hack_attempts.py --run-dir data/stage1/<run_id> [--run-dir ...] [--show]`; hand-read everything it flags.
- Unattended (as root, from the root-owned repo copy, not via `rhbench-run`): `bash scripts/stage1_pipeline.sh --run-id <run_id> --display plain [benchmark args] [--no-upload] [--no-stop] [--idle-minutes 30]` chains run → export → validate → upload; later steps still run if an earlier one fails. After a successful upload it hands over to `scripts/stop_box_when_idle.sh`, which **stops the Vast box** after 30 consecutive idle minutes (no SSH connection, no benchmark/upload/copy process, no vLLM requests); cancel with `touch /tmp/rhablation-no-stop`. Any SSH session, including Claude Code's, keeps the box up.
- Upload only (as root on the box, or from the laptop): `bash scripts/upload_run.sh --run-dir <run_dir> [--repo <user/name>]`
- The benchmark's `local` sandbox runs model-written code on the host; only use it as an unprivileged user with no credentials.

Stage 1 development steps:

# Pipeline steps: orthogonalization against reward hacking

Each step is a separate script (or set of scripts) that reads the previous step's output from disk. Each step ends with a check to pass before moving on.

## Step 1: Model and benchmark harness (in progress)
- **Does:** Serves Qwen3.5-9B with vLLM and runs it on ImpossibleBench LiveCodeBench (original, oneoff, conflicting splits; minimal and tools scaffolds). Saves every attempt as a trajectory in a fixed JSONL schema.
- **Output:** `data/stage1/<run_id>/trajectories.jsonl`
- **Check:** Validation script passes; the CoT is present and complete; stored token IDs match what the model saw. Smoke test on 5–10 tasks per variant.

## Step 2: Rollouts at scale
- **Does:** Runs the harness on as many tasks and samples per task as needed to collect enough hacking examples. Target roughly 10× the dataset size heretic uses for refusal, reduced if computing the direction would take more than about an hour.
- **Output:** Large trajectory dataset.
- **Check:** Hack rate per split and scaffold is non-trivial. Enough positive examples exist for extracting a direction.

## Step 3: Labelling
- **Does:** Adds an LLM judge that labels each trajectory as reward hacking or not (catches failed hack attempts the programmatic label misses). Adds a separate judge that reads the CoT and labels whether the hack looks intentional or accidental.
- **Output:** Trajectories with judge labels.
- **Check:** Judge agrees with the programmatic label where one exists. Hand-read 20–30 trajectories yourself.

## Step 4: Activation extraction and direction
- **Does:** Splits tasks by task ID into train and eval sets. Runs forward passes over train trajectories (hack vs no-hack), captures activations per layer, and computes the difference-in-means direction. Token-position choice over multi-turn completions is a deliberate research decision.
- **Output:** Reward-hacking direction per layer.
- **Check:** A linear probe on held-out trajectories separates hack from no-hack. If it can't, ablation won't work either.

## Step 5: Ablation
- **Does:** Uses heretic with the precomputed direction to orthogonalize the model's weights.
- **Output:** Ablated model.
- **Check:** Model is still coherent; KL divergence on benign prompts is reasonable.

## Step 6: Evals
- **Does:** Runs the ablated model on the held-out ImpossibleBench tasks (never used in step 4) and on a basic capability eval (MMLU via lm-eval-harness).
- **Output:** Hack rates and capability scores, before vs after ablation.
- **Check:** Compare against the unablated baseline.

## Step 7: Repeat and compare
- **Does:** Repeats steps 2–6 with different training data or benchmarks, then compares directions (e.g. cosine similarity per layer).
- **Output:** Results across datasets; direction similarity.
- **Check:** Results are consistent enough to interpret.

## Later (after the minimal viable version)
- Equally well-tuned DPO baseline on the same datasets.
- Sample efficiency and data-quality comparisons.
- Larger models.

## Conventions
- **Never commit to git.** The user makes all commits. Leave changes in the working tree.
- Each stage is a separate script; stages communicate only via files on disk.
- Config via file or CLI args; no hardcoded paths, models, or dataset sizes.
- Always run on a tiny subset (5–10 tasks) and show results before any full run.
- Never modify a benchmark's own scoring.
- Preserve full chain of thought in all saved trajectories.
- Log model revision, sampling params, and seeds for every run.
- Start each stage in plan mode; stop at the stated stopping point.
- If anything contradicts the assumptions in this file, flag it rather than working around it.