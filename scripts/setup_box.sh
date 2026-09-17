#!/usr/bin/env bash
# One-time setup of a fresh GPU box for stage 1. Run as root from anywhere inside the repo:
#   bash scripts/setup_box.sh [--no-vllm-install] [--no-serve]
# Safe to re-run: every step checks whether it is already done.
set -euo pipefail

INSTALL_VLLM=1
SERVE=1
for arg in "$@"; do
  case "$arg" in
    --no-vllm-install) INSTALL_VLLM=0 ;;
    --no-serve) SERVE=0 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_USER="${BENCH_USER:-rhbench}"
BENCH_HOME="/home/$BENCH_USER"
BENCH_REPO="$BENCH_HOME/$(basename "$REPO_DIR")"
VLLM_ENV="${VLLM_ENV:-$HOME/vllm-env}"
PORT="${PORT:-8000}"
MIN_FREE_GB=35
LAUNCHER=/usr/local/bin/rhbench-run

step() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (needed to create the unprivileged benchmark user)"

step "Hardware and disk"
nvidia-smi -L || die "no NVIDIA GPU visible"
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free disk on /: ${free_gb}G (need ~${MIN_FREE_GB}G)"
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "not enough free disk"

step "Tools"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi
uv --version
for tool in tmux rsync curl; do
  command -v "$tool" >/dev/null || { apt-get update -qq && apt-get install -y -qq "$tool"; }
done

step "Sandbox mode"
if docker info >/dev/null 2>&1; then
  SANDBOX=docker
  echo "Docker works: the benchmark can use its default Docker sandbox (--sandbox docker)."
else
  SANDBOX=local
  echo "No usable Docker: model-written code will run on this host as the unprivileged user '$BENCH_USER' (--sandbox local)."
fi

step "Unprivileged benchmark user"
id -u "$BENCH_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$BENCH_USER"
if id -nG "$BENCH_USER" | tr ' ' '\n' | grep -qxE 'sudo|wheel|docker|root'; then
  die "$BENCH_USER is in a privileged group; remove it before continuing"
fi
rsync -a --exclude .venv --exclude data --exclude .git --exclude .claude --exclude __pycache__ "$REPO_DIR/" "$BENCH_REPO/"
chown -R "$BENCH_USER:$BENCH_USER" "$BENCH_HOME"
chmod 750 "$BENCH_HOME"

step "Lock down credentials"
PRIVATE_PATHS=()
chmod 700 /root
if [ -d /workspace ]; then chmod 700 /workspace; PRIVATE_PATHS+=(/workspace); fi
case "$REPO_DIR" in "$BENCH_HOME"/*) ;; *) chmod 700 "$REPO_DIR"; PRIVATE_PATHS+=("$REPO_DIR") ;; esac
# Rented hosts (e.g. Vast.ai) inject API/Jupyter tokens here and leave the file world-readable.
if [ -f /etc/environment ]; then chmod 600 /etc/environment; fi
rm -f "$BENCH_HOME/.cache/huggingface/token"

step "Launcher: $LAUNCHER"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
# Run a command inside the benchmark repo as '$BENCH_USER' with a scrubbed environment.
#   rhbench-run uv run scripts/stage1_run_benchmark.py --sandbox local
exec runuser -u $BENCH_USER -- env -i HOME=$BENCH_HOME \\
  PATH=$BENCH_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 TERM=xterm \\
  bash -c 'cd "\$1" && shift && exec "\$@"' _ $BENCH_REPO "\$@"
EOF
chmod 755 "$LAUNCHER"

step "Benchmark environment (Python 3.12)"
"$LAUNCHER" uv sync --python 3.12
"$LAUNCHER" uv run python -c "import inspect_ai, impossiblebench; print('benchmark env ok, inspect_ai', inspect_ai.__version__)"

if [ "$INSTALL_VLLM" = 1 ]; then
  step "vLLM environment: $VLLM_ENV"
  if [ ! -x "$VLLM_ENV/bin/vllm" ]; then
    uv venv "$VLLM_ENV" --python 3.12 --clear
    uv pip install --python "$VLLM_ENV/bin/python" vllm
  fi
  "$VLLM_ENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
fi

if [ "$SERVE" = 1 ]; then
  step "vLLM server (tmux session 'vllm', log /tmp/vllm.log)"
  if curl -sf -m 5 "localhost:$PORT/v1/models" >/dev/null; then
    echo "already serving on :$PORT"
  else
    tmux kill-session -t vllm 2>/dev/null || true
    tmux new-session -d -s vllm "source '$VLLM_ENV/bin/activate' && cd '$REPO_DIR' && PORT=$PORT bash scripts/serve_vllm.sh 2>&1 | tee /tmp/vllm.log"
    echo "waiting for startup (first start downloads the model weights, ~9 GB for the 4B)..."
    for _ in $(seq 1 360); do
      curl -sf -m 5 "localhost:$PORT/v1/models" >/dev/null && break
      tmux has-session -t vllm 2>/dev/null || { tail -30 /tmp/vllm.log; die "vLLM exited; see /tmp/vllm.log"; }
      sleep 5
    done
    curl -sf -m 5 "localhost:$PORT/v1/models" >/dev/null || die "vLLM not up after 30 min; see /tmp/vllm.log"
  fi
  curl -s "localhost:$PORT/v1/models" | python3 -c 'import sys, json; [print("serving:", m["id"], "max_model_len", m.get("max_model_len")) for m in json.load(sys.stdin)["data"]]'
fi

step "Isolation self-check (as $BENCH_USER, scrubbed environment)"
"$LAUNCHER" bash -c '
  fail=0
  leaked=$(env | cut -d= -f1 | grep -iE "token|key|secret|passw" || true)
  if [ -n "$leaked" ]; then echo "FAIL credential-like env vars: $leaked"; fail=1; else echo "ok   no credential-like env vars"; fi
  for path in /etc/environment /root "$@"; do
    if [ -r "$path" ]; then echo "FAIL can read $path"; fail=1; else echo "ok   cannot read $path"; fi
  done
  if sudo -n true 2>/dev/null; then echo "FAIL has sudo"; fail=1; else echo "ok   no sudo"; fi
  if [ -e "$HOME/.ssh" ] || [ -e "$HOME/.cache/huggingface/token" ]; then echo "FAIL ssh keys or HF token present"; fail=1; else echo "ok   no ssh keys / HF token"; fi
  exit $fail
' _ "${PRIVATE_PATHS[@]}" || die "isolation self-check failed"

step "Done"
cat <<EOF
Smoke test (2 tasks):
  rhbench-run uv run scripts/stage1_run_benchmark.py --limit 2 --splits original --agent-types minimal --sandbox $SANDBOX --display plain
Then:
  rhbench-run uv run scripts/stage1_export_trajectories.py --run-dir data/stage1/<run_id>
  rhbench-run uv run scripts/stage1_validate.py --run-dir data/stage1/<run_id>
Outputs: $BENCH_REPO/data/stage1/<run_id>/
After a reboot: re-run this script (it restarts vLLM and re-applies the lock-down).
EOF
