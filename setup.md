# Setup: running stage 1 (ImpossibleBench rollouts)

Stage 1 runs `Qwen/Qwen3.5-4B` (served by vLLM; model set in `configs/stage1_lcb.yaml`) on Impossible-LiveCodeBench and writes
one JSON record per trajectory to `data/stage1/<run_id>/trajectories.jsonl`.

## Requirements

- Linux box with an NVIDIA GPU, root access (tested: 1× RTX PRO 6000 Blackwell, 95 GB, Vast.ai container). ~16 GB VRAM is enough for the 4B model (~24 GB for the 9B).
- ~35 GB free disk: model weights 9.3 GB (4B) or 19.3 GB (9B), vLLM/torch env ~10 GB, benchmark env ~2 GB.
- For the `Qwen/Qwen3.8-27B` sweep (`sweep/`): 1× H100 80 GB works in bf16 (~56 GB weights) with a reduced context (`MAX_MODEL_LEN=32768`); more or larger GPUs mainly buy throughput. ≥32 vCPUs / 64 GB RAM because sandboxed tests run on the CPU. Disk: 150 GB minimum, 200 GB comfortable. Weights 56 GB (one copy in the HF cache; environments never contain the model), +28 GB only if the FP8 fallback is downloaded; vLLM env ~10 GB (mostly torch + CUDA libraries); project env ~2 GB; each further benchmark env 1–6 GB (6 GB when it pulls its own torch); caches ~5 GB; data ~0.7 MB raw / 0.07 MB gzipped per transcript in the compact format. Fixed cost ≈ 75–85 GB without FP8, so 80 GB cannot work and 100 GB leaves no room for the fallback.
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

1. Checks GPU, free disk, and installs `uv`/`tmux`/`rsync` if missing. Enables tmux mouse scrolling and a 50k-line scrollback in root's `~/.tmux.conf` (hold Shift, or Option in macOS Terminal/iTerm, while dragging to select text natively).
2. Picks the sandbox mode: Docker if `docker info` works, otherwise `local` (see "Isolation").
3. Creates the unprivileged user `rhbench`, copies the repo to `/home/rhbench/llm-reward-hacking-ablation`.
4. Locks down credentials: `chmod 700 /root /workspace <repo>`, `chmod 600 /etc/environment`.
5. Installs the launcher `/usr/local/bin/rhbench-run`.
6. Builds the benchmark env as `rhbench` (`uv sync --python 3.12`) and the separate vLLM env (`~/vllm-env`) as root.
7. Starts vLLM in tmux session `vllm` (log: `/tmp/vllm.log`) and waits until `/v1/models` answers.
8. Runs an isolation self-check as `rhbench` and prints the next commands.

Flags: `--no-vllm-install`, `--no-serve`. To serve a model other than the 4B, set `MODEL`, e.g. `MODEL=Qwen/Qwen3.5-9B bash scripts/setup_box.sh`; it must match `model` in the run's config (`configs/stage1_lcb_9b_promptA.yaml` for the 9B).

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

Check on a running job from a second shell (add `--watch 60` to keep refreshing):

```bash
rhbench-run uv run scripts/stage1_progress.py --run-dir data/stage1/<run_id>
```

It shows done / running / total per split, passes so far (on `oneoff`/`conflicting` a pass is a
verified hack), errors, which limit stopped samples (`msg-lim` = the 50-message limit, `oth-lim` =
any other sample-level limit), how many samples had a model call end at `max_tokens` (`maxtok`) or
have its thinking cut at `thinking_token_budget` (`think-cut`), samples per hour, a rough ETA, minutes since the last sample finished,
and live vLLM load (running/queued requests, generated tokens per second). All splits and
scaffolds run in one Inspect call and share `max_connections`.

Unattended run (start it in tmux and leave; your laptop can be off). As root, from the
root-owned repo copy (`/workspace/...`, not `/home/rhbench/...`), one command runs the
benchmark, exports, validates and uploads the run directory (see "Off-box storage"):

```bash
bash scripts/stage1_pipeline.sh --run-id lcb-4b-part1 --offset 0 --limit 34 --sandbox local --display plain
```

It takes the same arguments as `stage1_run_benchmark.py`, plus `--no-upload`. A failing step
does not stop the later ones, so a crashed run is still uploaded; the step summary and exit
code say what failed. The full output is in `/tmp/stage1_<run_id>.log` and is uploaded as
`pipeline.log`.

**Auto-stop.** After a successful upload the pipeline hands over to
`scripts/stop_box_when_idle.sh`, which stops the box to end GPU charges once it has been idle
for 30 minutes in a row (`--idle-minutes N` to change, `--no-stop` to disable). Idle means all of:
no inbound SSH connection (your terminal, VS Code, Claude Code, an rsync/scp pull), no
benchmark/export/upload/copy process, and no vLLM requests in flight. While you stay connected it
keeps waiting, and stops 30 minutes after you leave.

- Keep the box up: `touch /tmp/rhablation-no-stop` (remove the file before the next run), or kill
  the tmux session. Status lines go to the terminal and `/tmp/stop_box_when_idle.log`.
- If the upload failed, or with `--no-upload`, the box is **not** stopped: the only copy of the
  run would be on its disk.
- It is a Vast "stop", not "destroy": the disk is kept and storage is still billed. Restarting
  can be delayed if someone else rents the GPU meanwhile. After a restart follow "Stopping and
  restarting the box" below.
- It uses the instance-scoped `CONTAINER_API_KEY` that Vast puts in `/etc/environment` (root-only
  after set-up); the pipeline checks at start that this works, so it fails early elsewhere.

Run the dataset in chunks without overlap (same shuffled order every time), e.g. the first
third now and the rest later:

```bash
rhbench-run uv run scripts/stage1_run_benchmark.py --run-id lcb-4b-part1 --offset 0  --limit 34 --sandbox local --display plain
rhbench-run uv run scripts/stage1_run_benchmark.py --run-id lcb-4b-part2 --offset 34 --limit 0  --sandbox local --display plain   # --limit 0 = all remaining
```

All settings live in `configs/stage1_lcb.yaml`; CLI overrides: `--limit`, `--offset`, `--splits`,
`--agent-types`, `--sandbox`, `--samples-per-task`, `--task-ids`, `--model`,
`--sample-time-limit-s` (wall-clock limit per sample; the sample is exported with status `timeout`),
`--no-seed` (needed for distinct samples when `--samples-per-task` > 1: a fixed seed is sent unchanged with every request),
`--vllm-base-url`, `--run-id`. The script prints the `run_id` and the selected task ids
(identical across splits by construction), then the benchmark's own accuracy per task.
Before starting it checks that vLLM serves the configured model with a large enough
`max_model_len`.

## Export and validate

```bash
rhbench-run uv run scripts/stage1_export_trajectories.py --run-dir data/stage1/<run_id>
rhbench-run uv run scripts/stage1_validate.py --run-dir data/stage1/<run_id>
```

- Storage format: records are written in the **compact** form (schema 1.2): each model call stores only the part of its prompt that differs from the previous call (`prompt_prefix_len` + the new ids), and the decoded prompt strings are left out, because ~90% of a full record was the same prompt repeated. `rhablation.compact.expand_trajectory(record, tokenizer)` restores the full record exactly; the validator does this before its checks. `--full` writes the old layout, `--gzip` writes `trajectories.jsonl.gz` (all readers accept it). `uv run scripts/stage1_compact_run.py --run-dir <run_dir> [--gzip]` converts an existing run into a new file next to the original (261 MB → 5 MB gzipped for `promptB-solvable19-tb4096`).
- Export writes `trajectories.jsonl`: one record per (task, split, scaffold, epoch), schema in
`src/rhablation/schema.py` — full messages, chain of thought per assistant turn, exact
prompt/completion text and token ids per turn, final code, benchmark score and raw test
output, `programmatic_hack` (true/false on impossible variants, null on `original`), `status`.
- Validate exits 1 on any schema violation; prints counts per status, pass/hack rate per
(variant, scaffold), integrity warnings, and one example (`--example-id`, `--full`).

Copy results back to your machine, either straight from the box or from the Hugging Face
dataset repo (works after the box is gone):

```bash
rsync -a gpubox:/home/rhbench/llm-reward-hacking-ablation/data/stage1/<run_id> data/stage1/
# or, with HF_TOKEN set (e.g. `set -a; . ~/.config/rhablation/upload.env; set +a`):
uvx --from huggingface_hub hf download <user>/<repo> --repo-type dataset --include "stage1/<run_id>/*" --local-dir data/
uv run inspect view --log-dir data/stage1/<run_id>/logs     # browse the raw Inspect logs
```

## Off-box storage (private Hugging Face dataset)

The box cannot push to your laptop, and its disk is gone once the instance is destroyed, so
finished runs are uploaded to a private HF dataset repo under `stage1/<run_id>/`.

One-time, on huggingface.co:

1. Create a **private dataset** repo, e.g. `<user>/rhablation-runs`.
2. Create a **fine-grained token** whose only permission is write access to that repo. A rented
   box is not a place for an account-wide token.
3. On your laptop, outside the repo (the set-up rsync would copy a repo-level `.env` into the
   tree that `rhbench` can read):

   ```bash
   mkdir -p ~/.config/rhablation && chmod 700 ~/.config/rhablation
   printf 'HF_TOKEN=hf_...\nHF_UPLOAD_REPO=<user>/rhablation-runs\n' > ~/.config/rhablation/upload.env
   chmod 600 ~/.config/rhablation/upload.env
   ```

On every new box, after `setup_box.sh`:

```bash
ssh gpubox 'install -d -m 700 /root/.config/rhablation'
scp ~/.config/rhablation/upload.env gpubox:/root/.config/rhablation/upload.env
ssh gpubox 'chmod 600 /root/.config/rhablation/upload.env'
```

The token lives under `/root` (mode 700, covered by the isolation self-check), and the upload
runs as root, never through `rhbench-run`. Upload a run by hand (also works from the laptop):

```bash
bash scripts/upload_run.sh --run-dir /home/rhbench/llm-reward-hacking-ablation/data/stage1/<run_id>
```

It refuses to upload a run directory that contains symlinks. `--dest-prefix <path>` uploads to
`<path>/<run_id>/` instead of `stage1/<run_id>/` (the unattended sweep uses `sweep/<model>/<benchmark>`).

## Unattended Claude Code on the box

For long investigations (many models × benchmarks) Claude Code runs on the box itself, as root, in
bypass-permissions mode, inside tmux. Everything for this lives in `sweep/`.

```bash
# laptop (the only two things the box cannot get by itself: the code and the HF token)
rsync -a --exclude .venv --exclude data --exclude .git --exclude .claude ./ gpubox:/workspace/llm-reward-hacking-ablation/
ssh gpubox 'mkdir -p /root/.config/rhablation && chmod 700 /root/.config/rhablation'
scp ~/.config/rhablation/upload.env gpubox:/root/.config/rhablation/upload.env
# everything else happens on the box
ssh gpubox bash /workspace/llm-reward-hacking-ablation/sweep/box_claude.sh   # [--model claude-fable-5-1] [--no-mission] [--watchdog-minutes 120] [--no-watchdog]
ssh -t gpubox tmux attach -t claude
```

In the tmux session: accept the bypass-permissions notice (and `/login` if Claude is not logged in yet:
open the printed URL on the laptop, paste the code back). Claude was started with a one-line prompt that
tells it to read and carry out `sweep/mission.md`, so nothing needs pasting; detach with `Ctrl-b d`.
The script:

- runs `scripts/setup_box.sh --no-serve` first if the box is not set up yet (the agent starts vLLM itself,
  on `127.0.0.1` only, with the mission's model);
- installs Claude Code (native installer) and starts `IS_SANDBOX=1 claude --dangerously-skip-permissions`
  in tmux session `claude` (root needs `IS_SANDBOX=1` for that flag);
- copies `sweep/CLAUDE.unattended.md` to `/root/.claude/CLAUDE.md`: no stopping for approval, all
  model-written code through `rhbench-run`, reports in `box_report/`, uploads under `sweep/` in the HF repo;
- checks that `rhbench` cannot read `/root/.claude` (where the login token is stored);
- arms a watchdog (tmux session `watchdog` = `scripts/stop_box_when_idle.sh --idle-minutes 120`). A `claude`
  process alone does not count as busy, but a session transcript written to in the last 15 minutes does,
  so long stretches of coding do not stop the box while a session that stalled (usage limit, API error)
  or finished still lets it stop, disk kept. Resume from the directory the session was started in (the
  repo; the 2026-09-21 run was started in `/workspace` because the Vast shell start-up changed directory) with
  `cd /workspace/llm-reward-hacking-ablation && IS_SANDBOX=1 claude --dangerously-skip-permissions --continue`.

Code the agent writes is saved to `sweep/code/` in the same private HF dataset repo by
`sweep/upload_code.sh` (a mirror of the box's code tree; every upload is a commit, so earlier versions
stay in the history; it refuses to upload if a file contains a credential). That is the only way code
leaves the box: the tree arrives without `.git`, the box holds no GitHub credentials, and the rules forbid
git and any other service, so the GitHub repo cannot be touched from the box. Run data folders stay
code-free (`upload_run.sh` refuses `.py`/`.sh`/`.ipynb`). Get the code back with `sweep/pull_box.sh`
(below); after the box has stopped: `bash sweep/pull_box.sh --from-hf <user/name>`. The set-up rsync stays
as it is: it is ~50 files / ~0.5 MB, `setup_box.sh` needs `pyproject.toml` to build the env, and the
existing harness, schema, exporter and validator save the agent from rebuilding them.

Progress over SSH: `bash sweep/status.sh [--watch 60]` (top of `SUMMARY.md`, live progress of the newest run,
watchdog log, free disk), or the usual `rhbench-run uv run scripts/stage1_progress.py --run-dir <run_dir>`:
every runner the agent writes, also for environments that do not use Inspect, must produce the
`run_config.json` + `progress.jsonl` that script reads.

Progress without SSH (an open SSH session keeps the box up): `sweep/box_report/SUMMARY.md` in the HF
dataset repo, refreshed after every model × benchmark pair. Claude spends no tokens while a benchmark
runs; it is re-invoked when the background run exits.

When finished, run `/logout` in the session: a stopped Vast disk otherwise keeps the login token.

### Keeping the laptop safe from the box

The box runs model-written code and an agent with web access as root, so treat its contents as untrusted.
SSH only pulls, so the ways back to the laptop are the ones you open yourself:

- Use plain `ssh`/tmux for this box, not a VS Code Remote-SSH window (a compromised remote can run
  code on the local machine through it). Keep agent and X11 forwarding off (`ssh -G gpubox | grep -i forward`).
- Never rsync the box's tree over the working tree: `scripts/`, `pyproject.toml`, `.claude/`, `.vscode/`,
  `CLAUDE.md` and `.git/hooks` all get executed or obeyed later. Use `bash sweep/pull_box.sh`: markdown
  reports land in `data/box_report/<time>/`, the code lands in `~/rhablation-box-quarantine/<time>/`
  with symlinks skipped and execute bits stripped, and you read the diff and apply changes by hand.
  `--from-hf <user/name>` does the same from `sweep/code/` in the HF repo; treat that code as equally untrusted.
- Transcripts come back through the HF dataset repo. Only load JSON/JSONL/safetensors/
  `np.load(allow_pickle=False)`; never `torch.load` or unpickle a file from the box.
- **Account connectors.** A subscription `/login` on the box brings the account's claude.ai connectors with
  it (found 2026-09-21: `claude mcp list` on the box showed Google Drive and Claude Docs connected). In
  bypass mode the agent, or anything that prompt-injects it, could use them without asking. That is not a
  path to the laptop, but it is personal data reachable from a rented box. `sweep/box_claude.sh` therefore
  starts Claude with `--strict-mcp-config` (no MCP servers at all), writes `mcp__*` deny rules to
  `/root/.claude/settings.json`, and the rules file forbids adding servers. Keep the flag when resuming by
  hand. If you have a Claude account without connectors, log the box in with that one instead.
- Credentials on the box must all be narrow and revocable: the Claude login, the one-repo HF token, the
  instance-scoped Vast key. No SSH keys, GitHub tokens or account-wide HF tokens.



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
- `model not served`**:** the `model` in the config must equal the id in `/v1/models`. To serve a different model: `MODEL=<id> bash scripts/serve_vllm.sh` and set the same id in the config (or pass `--model`).
- `max_model_len < required`**:** raise `MAX_MODEL_LEN` for `serve_vllm.sh` or lower `sampling.max_tokens`.
- **Many turns with** `finish_reason: length`**:** the thinking budget (`sampling.max_tokens`) is too small for the task.
- **Validator warns** `missing token ids`**:** Inspect occasionally logs a completed model call without
the raw request/response; reasoning and content are still saved, token ids are `null` for that turn.
- **Tool calls appear as raw XML in content (tools scaffold):** vLLM too old for Qwen3.5 thinking + tool calls.
- **Most tools samples fail at once with** `SandboxInjectionError` (`tar: Unexpected EOF in archive`, `[PYI-...] Failed to execute script`) **on a fresh box:**
with `--sandbox local`, Inspect installs its `inspect-sandbox-tools` binary into one shared host path (`/var/tmp/.da7be258e003d428`), and many tools samples
starting together extract into it at the same time (seen in `9b-promptA-smoke`: 20 of 24). The install is complete afterwards and stays on disk, so re-run the tools
samples; on a new box run a 1-2 task tools smoke (`--agent-types tools --limit 2`) before any large tools run.
- **A run never finishes: a few tools samples stay open, vLLM is idle, and `ps -u rhbench` shows an old `python test.py` at 100% CPU with parent PID 1:**
under `--sandbox local` a timed-out tool call kills its shell but not the shell's children, and an infinite loop behind a pipe (`python test.py | head -50`) keeps the pipe open, so the call never returns
(run `9b-promptA-full`, three samples blocked for hours). `src/rhablation/orphan_reaper.py` now kills such processes `orphan_grace_s` (default 120) seconds after they are orphaned and logs each kill to
`<run_dir>/reaped_processes.jsonl`. Check it on a box with `rhbench-run uv run scripts/stage1_check_orphan_reaper.py` (no GPU needed; `--without-reaper` reproduces the hang). By hand: `kill` those PIDs.
- **Throughput collapses late in a tools run, vLLM shows queued requests:** long tools contexts fill the KV cache (`vllm:kv_cache_usage_perc` near 1, `num_preemptions_total` climbing in `/metrics`);
preempted requests are recomputed. Results are unaffected; lower `max_connections` (96 was too many for the 9B tools scaffold on an 80 GB H100).
- Downloads go to `~/.cache/huggingface`; set `HF_HOME` to move them.

