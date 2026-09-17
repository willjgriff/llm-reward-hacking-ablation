# Setup: running stage 1 (ImpossibleBench rollouts)

Stage 1 runs `Qwen/Qwen3.5-9B` (served by vLLM) on Impossible-LiveCodeBench and writes
one JSON record per trajectory to `data/stage1/<run_id>/trajectories.jsonl`.

## Requirements

- Linux box with an NVIDIA GPU, root access (tested: 1× RTX PRO 6000 Blackwell, 95 GB, Vast.ai container). ~24 GB VRAM is enough for the 9B model.
- ~35 GB free disk: model weights 19.3 GB, vLLM/torch env ~10 GB, benchmark env ~2 GB.
- Internet access on first run. Model and dataset are public; no HF token is needed, and none should be left on the box.
- An SSH alias on your machine, e.g. in `~/.ssh/config`:
  ```
  Host gpubox
      HostName <host>
      Port <port>
      User root
      IdentityFile ~/.ssh/<key>
  ```
  `ssh gpubox hostname` must work without prompts.



## Quick path (fresh box, ~15–20 min, mostly unattended)

From the repo root on your machine:

> ```bash
> rsync -a --exclude .venv --exclude data --exclude .git --exclude .claude ./ gpubox:/workspace/llm-reward-hacking-ablation/
> ssh gpubox 'cd /workspace/llm-reward-hacking-ablation && bash scripts/setup_box.sh'
> ```

`scripts/setup_box.sh` (idempotent; re-run it after a reboot) does all of the following:

1. Checks GPU, free disk, and installs `uv`/`tmux`/`rsync` if missing.
2. Picks the sandbox mode: Docker if `docker info` works, otherwise `local` (see "Isolation").
3. Creates the unprivileged user `rhbench`, copies the repo to `/home/rhbench/llm-reward-hacking-ablation`.
4. Locks down credentials: `chmod 700 /root /workspace <repo>`, `chmod 600 /etc/environment`.
5. Installs the launcher `/usr/local/bin/rhbench-run`.
6. Builds the benchmark env as `rhbench` (`uv sync --python 3.12`) and the separate vLLM env (`~/vllm-env`) as root.
7. Starts vLLM in tmux session `vllm` (log: `/tmp/vllm.log`) and waits until `/v1/models` answers.
8. Runs an isolation self-check as `rhbench` and prints the next commands.

Flags: `--no-vllm-install`, `--no-serve`.

## Isolation

The benchmark executes model-written Python. ImpossibleBench's default sandbox is Docker
(throwaway `aisiuk/inspect-tool-support` containers, no network). Rented GPU containers
usually cannot run Docker, so the fallback is Inspect's `local` sandbox, which runs that
code directly on the host. To keep that safe:

- The benchmark only ever runs through `rhbench-run <command>`, which executes the command
inside the repo as `rhbench` (no sudo, no SSH keys, no HF token) with a scrubbed
environment (`env -i`).
- Hosts like Vast.ai inject `JUPYTER_TOKEN`, `CONTAINER_API_KEY`, etc. through a
world-readable `/etc/environment`, and run a root Jupyter server on the box. The setup
script makes that file root-only so `rhbench` cannot obtain those tokens.
Undo with `chmod 644 /etc/environment`.
- vLLM runs as root; it never executes model-written code. `rhbench` only talks to it on `localhost:8000`.



## Run

Always smoke-test first (2 tasks, one split, no tool calls, ~10–15 min):

```bash
ssh gpubox
tmux new -s bench
rhbench-run uv run scripts/stage1_run_benchmark.py --limit 2 --splits original --agent-types minimal --sandbox local --display plain
```

Full stage-1 run (8 tasks × 3 splits × both scaffolds):

```bash
rhbench-run uv run scripts/stage1_run_benchmark.py --agent-types minimal tools --sandbox local --display plain
```

All settings live in `configs/stage1_lcb.yaml`; CLI overrides: `--limit`, `--splits`,
`--agent-types`, `--sandbox`, `--samples-per-task`, `--task-ids`, `--model`,
`--vllm-base-url`, `--run-id`. The script prints the `run_id` and the selected task ids
(identical across splits by construction), then the benchmark's own accuracy per task.
Before starting it checks that vLLM serves the configured model with a large enough
`max_model_len`.

## Export and validate

```bash
rhbench-run uv run scripts/stage1_export_trajectories.py --run-dir data/stage1/<run_id>
rhbench-run uv run scripts/stage1_validate.py --run-dir data/stage1/<run_id>
```

- Export writes `trajectories.jsonl`: one record per (task, split, scaffold, epoch), schema in
`src/rhablation/schema.py` — full messages, chain of thought per assistant turn, exact
prompt/completion text and token ids per turn, final code, benchmark score and raw test
output, `programmatic_hack` (true/false on impossible variants, null on `original`), `status`.
- Validate exits 1 on any schema violation; prints counts per status, pass/hack rate per
(variant, scaffold), integrity warnings, and one example (`--example-id`, `--full`).

Copy results back to your machine:

```bash
rsync -a gpubox:/home/rhbench/llm-reward-hacking-ablation/data/stage1/<run_id> data/stage1/
uv run inspect view --log-dir data/stage1/<run_id>/logs     # browse the raw Inspect logs
```



## Stopping and restarting the box

- Stop (not destroy) keeps the disk: user, envs, model weights. Processes (vLLM, tmux) do not survive.
- After a restart the SSH host/port may change: update `~/.ssh/config`.
- Re-run `bash scripts/setup_box.sh`: it skips what exists, restarts vLLM, and re-applies the
lock-down (the host may regenerate `/etc/environment` on boot).



## Manual steps (what the script does, if you need to do it by hand)

```bash
useradd -m -s /bin/bash rhbench
rsync -a --exclude .venv --exclude data --exclude .git --exclude .claude ./ /home/rhbench/llm-reward-hacking-ablation/
chown -R rhbench:rhbench /home/rhbench && chmod 700 /root /workspace && chmod 600 /etc/environment
# benchmark env (as rhbench):   uv sync --python 3.12
# vLLM env (as root):           uv venv ~/vllm-env --python 3.12 && uv pip install --python ~/vllm-env/bin/python vllm
# serve (as root, in tmux):     source ~/vllm-env/bin/activate && bash scripts/serve_vllm.sh
```



## Troubleshooting

- **vLLM dies at startup with** `FlashInfer requires GPUs with sm75 or higher` (Blackwell GPUs):
`serve_vllm.sh` already sets `VLLM_USE_FLASHINFER_SAMPLER=0`; make sure you start vLLM through it.
- **Can't scroll vLLM output:** it is in `/tmp/vllm.log`; find the real error with
`grep -n "Error" /tmp/vllm.log | head`.
- `SyntaxError: f-string expression part cannot include a backslash` when importing
`impossiblebench`: the venv is on Python < 3.12. Rebuild with `uv sync --python 3.12`.
- `model not served`**:** the `model` in the config must equal the id in `/v1/models` (`Qwen/Qwen3.5-9B`).
- `max_model_len < required`**:** raise `MAX_MODEL_LEN` for `serve_vllm.sh` or lower `sampling.max_tokens`.
- **Many turns with** `finish_reason: length`**:** the thinking budget (`sampling.max_tokens`) is too small for the task.
- **Validator warns** `missing token ids`**:** Inspect occasionally logs a completed model call without
the raw request/response; reasoning and content are still saved, token ids are `null` for that turn.
- **Tool calls appear as raw XML in content (tools scaffold):** vLLM too old for Qwen3.5 thinking + tool calls.
- Downloads go to `~/.cache/huggingface`; set `HF_HOME` to move them.

