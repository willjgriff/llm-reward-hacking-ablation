#!/usr/bin/env bash
# Adds the capability evals (cheatbench/mission_capability.md) to the CheatBench VM. Run as root on the VM, from the
# root-owned repo copy, after cheatbench/setup_cpu_vm.sh:
#   bash cheatbench/setup_tbench.sh
# Safe to re-run.
#
#   1. Docker (Ubuntu's docker.io + compose v2 + buildx) for Terminal-Bench 2.0, whose 89 tasks each run in their
#      own container. Harbor drives them as root (Docker access is root-equivalent anyway); model-written commands
#      run only inside the task containers. `cheatbench` stays out of the docker group.
#   2. Harbor at the commit CheatBench pins for Terminus-2 (4641fd5, archive sha256-checked), in /opt/harbor, so
#      Terminal-Bench episodes use the same agent code as the CheatBench rows.
#   3. lm-eval 0.4.9 (CPU torch) for `cheatbench` in /home/cheatbench/lmeval, plus cheatbench/tools/lmeval_chat.py,
#      which adds the rows' sampling settings (top_p, top_k, reasoning effort low) to every chat request.
#      GSM8K and IFEval run as `cheatbench` against the tunnel; nothing in them executes model output.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARBOR_COMMIT=4641fd590f190411c57e1cbd824a9f6786cbe92e
HARBOR_ARCHIVE_SHA256=ca4079f8189f290134c8619c66a131b21a598893e4d454cceb64a8f53b8cfc79   # CheatBench core/agents/terminus_2.py
HARBOR_DIR=/opt/harbor
LMEVAL_VERSION=0.4.9
RUN_USER=cheatbench
LMEVAL_DIR="/home/$RUN_USER/lmeval"
MIN_FREE_GB=150

step() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root"
id "$RUN_USER" >/dev/null 2>&1 || die "no $RUN_USER user: run cheatbench/setup_cpu_vm.sh first"
command -v uv >/dev/null || die "no uv: run cheatbench/setup_cpu_vm.sh first"
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free disk on /: ${free_gb}G (Terminal-Bench images: allow ~${MIN_FREE_GB}G)"
[ "$free_gb" -ge "$MIN_FREE_GB" ] || echo "WARNING: little disk for 89 task images; prune images between rows"

step "Docker"
export DEBIAN_FRONTEND=noninteractive
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  apt-get update -q
  apt-get install -y -q --no-install-recommends docker.io docker-compose-v2 docker-buildx
fi
systemctl enable --now docker
docker version --format 'docker {{.Server.Version}}'
docker compose version
docker run --rm hello-world >/dev/null && echo "ok   containers run"
if id -nG "$RUN_USER" | tr ' ' '\n' | grep -qxE 'sudo|admin|docker|lxd'; then
  die "$RUN_USER is in a privileged group; remove it"
fi

step "Harbor @ ${HARBOR_COMMIT:0:7} in $HARBOR_DIR"
if [ "$(cat "$HARBOR_DIR/commit" 2>/dev/null)" = "$HARBOR_COMMIT" ] && "$HARBOR_DIR/.venv/bin/harbor" --help >/dev/null 2>&1; then
  echo "already installed"
else
  rm -rf "$HARBOR_DIR"; mkdir -p "$HARBOR_DIR/src"
  tmp=$(mktemp)
  curl -fsSL "https://codeload.github.com/harbor-framework/harbor/tar.gz/$HARBOR_COMMIT" -o "$tmp"
  echo "$HARBOR_ARCHIVE_SHA256  $tmp" | sha256sum -c -
  tar -xzf "$tmp" --strip-components=1 -C "$HARBOR_DIR/src"; rm -f "$tmp"
  (cd "$HARBOR_DIR/src" && UV_PROJECT_ENVIRONMENT="$HARBOR_DIR/.venv" uv sync -q --frozen --no-dev --no-editable \
    --package harbor --python 3.12)
  echo "$HARBOR_COMMIT" > "$HARBOR_DIR/commit"
fi
ln -sf "$HARBOR_DIR/.venv/bin/harbor" /usr/local/bin/harbor
harbor --help >/dev/null && echo "ok   harbor CLI"

step "lm-eval $LMEVAL_VERSION for $RUN_USER in $LMEVAL_DIR"
if ! sudo -u "$RUN_USER" "$LMEVAL_DIR/bin/python" -c "import lm_eval, langdetect" 2>/dev/null; then
  sudo -u "$RUN_USER" -H bash -c "cd ~ && uv venv -q -p 3.12 '$LMEVAL_DIR' && \
    VIRTUAL_ENV='$LMEVAL_DIR' uv pip install -q --torch-backend cpu 'lm_eval[ifeval,api]==$LMEVAL_VERSION' 'transformers<5'"
fi
install -m 644 -o "$RUN_USER" -g "$RUN_USER" "$REPO_DIR/cheatbench/tools/lmeval_chat.py" "$LMEVAL_DIR/lmeval_chat.py"
sudo -u "$RUN_USER" -H bash -c "cd ~ && '$LMEVAL_DIR/bin/python' -I '$LMEVAL_DIR/lmeval_chat.py' --self-test"

step "Done"
cat <<EOF
Terminal-Bench (root):  harbor run -d terminal-bench@2.0 -a terminus-2 -m openai/cb_<row> ...  (cheatbench/mission_capability.md)
lm-eval (as $RUN_USER): $LMEVAL_DIR/bin/python -I $LMEVAL_DIR/lmeval_chat.py --row <row> --tasks gsm8k_cot_llama,ifeval ...
EOF
