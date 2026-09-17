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
- ImpossibleBench subset and variants used: LiveCodeBench (`fjzzq2002/impossible_livecodebench`, 103 tasks/split), splits `original`/`oneoff`/`conflicting`, scaffolds `minimal` (text submission loop, no tool calls) and `tools` (bash/python/text_editor via Inspect `basic_agent`). Paper-replication prompt, `allow_test_modifications=True`, `max_attempts=10`, `message_limit=50`. `programmatic_hack` = test pass on `oneoff`/`conflicting`; `null` on `original`. Shuffle uses the benchmark's `seed=42`, which selects identical task IDs on every split.
- Ablation library: heretic (≥1.3.0 handles Qwen3.5's hybrid GatedDeltaNet/attention layers). obliteratus fail-closes on every Qwen3.5 size except Qwen3.8-27B — not usable.
- Model and revision: `Qwen/Qwen3.5-9B`; revision resolved at run time and written to `run_config.json` (main as of 2026-09-16: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`). vLLM flags in `scripts/serve_vllm.sh` (`--reasoning-parser qwen3 --tool-call-parser qwen3_xml`, `--max-model-len 262144`).
- Judge model and prompts: [fill in]
- Token positions and aggregation for activations over multi-turn completions: [fill in; this is a deliberate research decision, do not choose silently]
- Layers and ablation hyperparameters: [fill in]

## Stage 1 commands (GPU box)
- Env: `uv sync` (Python 3.12). vLLM is installed separately (`uv pip install vllm` into the same venv on the GPU box) because it pins torch/CUDA.
- Serve: `bash scripts/serve_vllm.sh` (env: `MODEL`, `PORT`, `MAX_MODEL_LEN`, `MODEL_REVISION`).
- Run: `uv run scripts/stage1_run_benchmark.py --config configs/stage1_lcb.yaml [--limit N --splits ... --agent-types ... --sandbox local|docker]`
- Export: `uv run scripts/stage1_export_trajectories.py --run-dir data/stage1/<run_id>`
- Validate: `uv run scripts/stage1_validate.py --run-dir data/stage1/<run_id>`
- The benchmark's `local` sandbox runs model-written code on the host; only use it as an unprivileged user with no credentials.

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