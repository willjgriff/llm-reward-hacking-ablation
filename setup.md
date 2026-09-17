# Setup: running stage 1 (ImpossibleBench rollouts)

Stage 1 runs `Qwen/Qwen3.5-9B` (served by vLLM) on Impossible-LiveCodeBench and writes
one JSON record per trajectory to `data/stage1/<run_id>/trajectories.jsonl`.

## Requirements

- Linux box with an NVIDIA GPU (tested target: 1× RTX PRO 6000, 95 GB). ~24 GB VRAM is enough for the 9B model.
- ~35 GB free disk: model weights 19.3 GB, vLLM/torch env ~10 GB, benchmark env ~2 GB.
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Internet access for the first run (HF model + dataset download, no HF token needed — both are public).
- Docker (optional but recommended, see "Isolation").

## 1. Get the code onto the box

Either `git clone` the repo, or from the Mac:

```bash
rsync -a --exclude .venv --exclude data --exclude .git ./ gpubox:~/llm-reward-hacking-ablation/
```

## 2. Install

Two environments: the benchmark env (this repo) and a separate vLLM env, because vLLM pins
torch/CUDA and its own transformers version.

```bash
cd ~/llm-reward-hacking-ablation
uv sync                                    # benchmark env: .venv/ (Python 3.12)

uv venv ~/vllm-env --python 3.12           # vLLM env
uv pip install --python ~/vllm-env/bin/python vllm
```

## 3. Isolation (read before running)

The benchmark executes model-written Python. ImpossibleBench's default sandbox is Docker
(`sandbox: docker` in `configs/stage1_lcb.yaml`); each sample runs in a throwaway
`aisiuk/inspect-tool-support` container with no network.

If Docker is not available, `sandbox: local` runs that code directly on the host in a temp
dir. Only do this as a dedicated unprivileged user with no credentials:

```bash
# as an admin user, once:
sudo useradd -m -s /bin/bash rhbench && sudo chmod 700 /home/rhbench
# then repeat steps 1–2 inside /home/rhbench as the rhbench user (no SSH keys, no HF token).
```

vLLM itself never runs model-written code, so it can run as your normal user; the benchmark
process (step 5) runs as `rhbench` and talks to `localhost:8000` only.

## 4. Serve the model

In its own terminal / tmux window:

```bash
source ~/vllm-env/bin/activate
bash scripts/serve_vllm.sh          # env overrides: MODEL, PORT, MAX_MODEL_LEN, MODEL_REVISION
```

Wait for `Application startup complete`, then check:

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool | grep -E '"id"|max_model_len'
```

`max_model_len` must be ≥ `max_attempts × sampling.max_tokens + 8192` (10 × 16384 + 8192 with
the default config); the run script refuses to start otherwise. Flags in `serve_vllm.sh`
(`--reasoning-parser qwen3 --tool-call-parser qwen3_xml --language-model-only`) are required
for thinking + tool calls; don't drop them.

## 5. Run the benchmark

Smoke test first (2 tasks, one split, no tool calls):

```bash
uv run scripts/stage1_run_benchmark.py --limit 2 --splits original --agent-types minimal --sandbox docker
```

Full stage-1 run (8 tasks × 3 splits, both scaffolds):

```bash
uv run scripts/stage1_run_benchmark.py --agent-types minimal tools
```

All settings live in `configs/stage1_lcb.yaml`; any of them can be overridden on the CLI
(`--limit`, `--splits`, `--agent-types`, `--sandbox`, `--samples-per-task`, `--task-ids`,
`--model`, `--vllm-base-url`, `--run-id`). The script prints the `run_id` and the selected
task ids (identical across splits by construction), then the benchmark's own accuracy per task.

Outputs: `data/stage1/<run_id>/run_config.json` (config, resolved model revision, package
versions, task ids) and `data/stage1/<run_id>/logs/*.eval` (Inspect logs, one per split×scaffold).
`inspect view --log-dir data/stage1/<run_id>/logs` opens them in a browser.

## 6. Export trajectories

```bash
uv run scripts/stage1_export_trajectories.py --run-dir data/stage1/<run_id>
```

Writes `data/stage1/<run_id>/trajectories.jsonl` — one record per (task, split, scaffold, epoch),
schema in `src/rhablation/schema.py`. Each record has the full message history, the chain of
thought per assistant turn, the exact prompt/completion text and token ids per turn, the final
code, the benchmark score and raw test output, `programmatic_hack` (true/false on impossible
variants, null on `original`), and `status`.

## 7. Validate

```bash
uv run scripts/stage1_validate.py --run-dir data/stage1/<run_id>
```

Exit code 1 if any record fails the schema. Prints counts per status, pass/hack rate per
(variant, scaffold), integrity warnings (missing reasoning, token-id mismatches), and one full
example trajectory (`--example-id <trajectory_id>` to pick one, `--full` for full token arrays).

## Troubleshooting

- `model not served`: the `model` in the config must match the id in `/v1/models` exactly (`Qwen/Qwen3.5-9B`).
- `max_model_len < required`: raise `MAX_MODEL_LEN` for `serve_vllm.sh` or lower `max_attempts`/`sampling.max_tokens`.
- Tool calls appearing as raw XML in content (tools scaffold): vLLM too old; needs ≥ 0.20 (fixes for Qwen3.5 thinking + tool calls).
- `text_editor` tool failing with `sandbox: local`: Inspect installs `inspect-tool-support` on the host on first use; it needs write access to the running user's home.
- Everything downloads into `~/.cache/huggingface`; set `HF_HOME` to move it.
