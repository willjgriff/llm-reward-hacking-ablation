#!/usr/bin/env bash
# One-time setup of a CPU-only VM (Hetzner Cloud CCX or similar: x86_64, Ubuntu 24.04, a real
# kernel) that runs CheatBench episodes against the model vLLM serves on the GPU box.
# Run as root on the VM, from the root-owned repo copy (cheatbench/README.md):
#   BOX_HOST=<gpu box ip> BOX_PORT=<gpu box ssh port> bash cheatbench/setup_cpu_vm.sh [--smoke]
# Without BOX_HOST the tunnel step is skipped (sandbox only). Safe to re-run.
#
# Why a separate VM: CheatBench's only runtime is rootless enroot (user/pid/net namespaces via
# `unshare`, squashfuse + fuse-overlayfs, slirp4netns), which Vast's unprivileged Docker
# containers block. vLLM stays bound to 127.0.0.1 on the GPU box; this VM reaches it through an
# SSH tunnel whose key the GPU box only accepts for that one port (cheatbench/allow_tunnel_key.sh).
#
# Users:
#   cheatbench  runs run.py and therefore every episode (model-written code). No sudo, no keys.
#   rhtunnel    owns the tunnel key and runs autossh. cheatbench cannot read the key.
#   root        the only holder of credentials: the HF upload token (/root/.config/rhablation/upload.env,
#               the fine-grained one-repo token the GPU box uses, so the VM uploads results without the
#               laptop) and the Claude Code login of cheatbench/hetzner_claude.sh. No OpenRouter or other
#               paid-API keys (judge off the VM), nothing in Hetzner cloud-init user-data: open-network
#               episodes share the host network and can reach the metadata service.
set -euo pipefail

SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

BOX_HOST="${BOX_HOST:-}"
BOX_PORT="${BOX_PORT:-22}"
BOX_USER="${BOX_USER:-root}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVED_MODEL="${SERVED_MODEL:-cb_base}"   # the row the GPU box serves after cheatbench/gpubox_setup.sh
CB_REPO="${CB_REPO:-https://github.com/centerforaisafety/cheatbench}"
CB_COMMIT="${CB_COMMIT:-1bd70d4f7509966e069ef602e5c4df77d66d2802}"   # read 2026-10-23
ENROOT_VERSION="${ENROOT_VERSION:-4.2.1}"
UV_VERSION="${UV_VERSION:-0.8.17}"   # root's uvx, for scripts/upload_run.sh
RUN_USER=cheatbench
TUNNEL_USER=rhtunnel
RUN_HOME="/home/$RUN_USER"
CB_DIR="$RUN_HOME/cheatbench"
TUNNEL_DIR=/etc/rhtunnel
MIN_FREE_GB=40

step() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
as_run() { sudo -u "$RUN_USER" -H bash -c "cd '$CB_DIR' && $*"; }

[ "$(id -u)" = 0 ] || die "run as root"

step "Host"
[ "$(uname -m)" = x86_64 ] || die "x86_64 only (some CheatBench tasks ship x86_64 binaries)"
. /etc/os-release; echo "$PRETTY_NAME, kernel $(uname -r)"
[ "${ID:-}" = ubuntu ] || echo "WARNING: tested for Ubuntu; package names may differ"
[ -f /.dockerenv ] && die "this is a Docker container; CheatBench's namespaces need a VM"
echo "cpus $(nproc), mem $(free -g | awk '/Mem:/{print $2}')G (CheatBench sizes 1 CPU + 4 GB per episode)"
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "free disk on /: ${free_gb}G (need ~${MIN_FREE_GB}G)"
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "not enough free disk"

step "Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends \
  git curl ca-certificates jq sudo \
  squashfs-tools squashfuse fuse-overlayfs fuse3 slirp4netns iptables iproute2 util-linux \
  pigz parallel zstd tmux rsync \
  python3 python3-venv python3-dev build-essential libcairo2 \
  autossh openssh-client
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version' \
  || die "CheatBench needs Python >= 3.11"
if [ "$(/opt/rhtools/bin/uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
  python3 -m venv /opt/rhtools
  /opt/rhtools/bin/pip install -q "uv==$UV_VERSION"
fi
ln -sf /opt/rhtools/bin/uv /usr/local/bin/uv
ln -sf /opt/rhtools/bin/uvx /usr/local/bin/uvx
uv --version

step "enroot $ENROOT_VERSION"
if dpkg -s enroot >/dev/null 2>&1 && [ "$(dpkg-query -W -f='${Version}' enroot)" = "$ENROOT_VERSION-1" ]; then
  echo "already installed"
else
  tmp=$(mktemp -d)
  base="https://github.com/NVIDIA/enroot/releases/download/v$ENROOT_VERSION"
  for p in "enroot_${ENROOT_VERSION}-1_amd64.deb" "enroot+caps_${ENROOT_VERSION}-1_amd64.deb"; do
    curl -fsSL -o "$tmp/$p" "$base/$p"
  done
  apt-get install -y -q "$tmp"/*.deb
  rm -rf "$tmp"
fi
enroot version
# enroot's runtime dir defaults to ${XDG_RUNTIME_DIR:-/run}/enroot, and `sudo -u cheatbench` leaves XDG_RUNTIME_DIR
# unset, so CheatBench's image build failed with "mkdir: cannot create directory '/run/enroot'". Episode starts are
# unaffected (core/sandbox/runtime.py exports ENROOT_RUNTIME_PATH itself, and an exported value beats this file).
echo 'ENROOT_RUNTIME_PATH /tmp/enroot-runtime-$(id -u)' > /etc/enroot/enroot.conf.d/50-rhablation-runtime.conf

step "Unprivileged user namespaces"
# Ubuntu >= 23.10 lets AppArmor deny unprivileged userns; CheatBench's sandbox needs them.
conf=/etc/sysctl.d/60-cheatbench-userns.conf
{
  [ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ] \
    && echo "kernel.apparmor_restrict_unprivileged_userns = 0"
  [ -e /proc/sys/kernel/unprivileged_userns_clone ] && echo "kernel.unprivileged_userns_clone = 1"
  echo "user.max_user_namespaces = 63000"
} > "$conf"
sysctl -q -p "$conf"
[ -e /dev/fuse ] || die "/dev/fuse missing"
chmod 666 /dev/fuse

step "Users"
id "$RUN_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$RUN_USER"
id "$TUNNEL_USER" >/dev/null 2>&1 || useradd -r -m -d "$TUNNEL_DIR" -s /usr/sbin/nologin "$TUNNEL_USER"
chmod 750 "$RUN_HOME"
if id -nG "$RUN_USER" | tr ' ' '\n' | grep -qxE 'sudo|admin|docker|lxd'; then
  die "$RUN_USER is in a privileged group; remove it"
fi
sudo -u "$RUN_USER" unshare --user --map-root-user --mount --pid --fork --mount-proc true \
  || die "unprivileged unshare still fails for $RUN_USER"
echo "unprivileged namespaces ok for $RUN_USER"
chmod 700 /root
install -d -m 700 /root/.config /root/.config/rhablation
for p in /root /root/.config/rhablation/upload.env "$TUNNEL_DIR/id_ed25519" /usr/local/sbin/request-gpubox-stop \
         /usr/local/sbin/request-gpubox-row; do
  if sudo -u "$RUN_USER" test -r "$p" 2>/dev/null && [ -e "$p" ]; then die "$RUN_USER can read $p"; fi
done
echo "ok   $RUN_USER cannot read /root (upload token, Claude login, done and row keys) or the tunnel key"
[ -f /root/.config/rhablation/upload.env ] \
  || echo "NOTE: /root/.config/rhablation/upload.env missing: copy it from the laptop before an unattended run"

step "Tunnel to the GPU box"
# The address the box sees this VM connect from (the box key is restricted to it).
VM_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')"
DONE_KEY=/root/.ssh/gpubox_done
ROW_KEY=/root/.ssh/gpubox_row
allow_cmd() { echo "FROM=$VM_IP DONE_KEY='$(cat "$DONE_KEY.pub")' ROW_KEY='$(cat "$ROW_KEY.pub")' bash cheatbench/allow_tunnel_key.sh '$(cat "$TUNNEL_DIR/id_ed25519.pub")'"; }
if [ -z "$BOX_HOST" ]; then
  echo "BOX_HOST not set: skipping (re-run with BOX_HOST/BOX_PORT to add it)"
else
  install -d -m 700 -o "$TUNNEL_USER" -g "$TUNNEL_USER" "$TUNNEL_DIR"
  key="$TUNNEL_DIR/id_ed25519"
  [ -f "$key" ] || sudo -u "$TUNNEL_USER" ssh-keygen -q -t ed25519 -N '' -C "rhtunnel@$(hostname)" -f "$key"
  # Trust on first use; check the fingerprint against the box (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`).
  if ! sudo -u "$TUNNEL_USER" ssh-keygen -F "[$BOX_HOST]:$BOX_PORT" -f "$TUNNEL_DIR/known_hosts" >/dev/null 2>&1; then
    ssh-keyscan -p "$BOX_PORT" -t ed25519 "$BOX_HOST" 2>/dev/null > "$TUNNEL_DIR/known_hosts.new"
    [ -s "$TUNNEL_DIR/known_hosts.new" ] || die "ssh-keyscan $BOX_HOST:$BOX_PORT returned nothing"
    mv "$TUNNEL_DIR/known_hosts.new" "$TUNNEL_DIR/known_hosts"
    chown "$TUNNEL_USER:$TUNNEL_USER" "$TUNNEL_DIR/known_hosts"
    echo "box host key (compare on the box):"; ssh-keygen -lf "$TUNNEL_DIR/known_hosts"
  fi
  cat > "$TUNNEL_DIR/box.env" <<EOF
BOX_HOST=$BOX_HOST
BOX_PORT=$BOX_PORT
BOX_USER=$BOX_USER
VLLM_PORT=$VLLM_PORT
AUTOSSH_GATETIME=0
EOF
  cat > /etc/systemd/system/rhtunnel.service <<'EOF'
[Unit]
Description=SSH tunnel to vLLM on the GPU box (127.0.0.1 only)
After=network-online.target
Wants=network-online.target

[Service]
User=rhtunnel
EnvironmentFile=/etc/rhtunnel/box.env
ExecStart=/usr/bin/autossh -M 0 -N -T \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/etc/rhtunnel/known_hosts -i /etc/rhtunnel/id_ed25519 \
  -L 127.0.0.1:${VLLM_PORT}:127.0.0.1:${VLLM_PORT} -p ${BOX_PORT} ${BOX_USER}@${BOX_HOST}
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now rhtunnel.service
  systemctl restart rhtunnel.service
  # Second key, root's, for one thing only: the box runs a forced command that asks its watchdog to stop
  # the box as soon as nothing there is busy (instead of after the idle window).
  install -d -m 700 /root/.ssh
  [ -f "$DONE_KEY" ] || ssh-keygen -q -t ed25519 -N '' -C "gpubox-done@$(hostname)" -f "$DONE_KEY"
  cat > /usr/local/sbin/request-gpubox-stop <<'EOF'
#!/usr/bin/env bash
# Ask the GPU box to stop itself once nothing there is busy. Run as root when done with the box.
set -euo pipefail
. /etc/rhtunnel/box.env
exec ssh -T -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/etc/rhtunnel/known_hosts -i /root/.ssh/gpubox_done \
  -p "$BOX_PORT" "$BOX_USER@$BOX_HOST"
EOF
  chmod 700 /usr/local/sbin/request-gpubox-stop
  # Third key, root's: the box runs cheatbench/serve_row.sh with the row named here and nothing else.
  [ -f "$ROW_KEY" ] || ssh-keygen -q -t ed25519 -N '' -C "gpubox-row@$(hostname)" -f "$ROW_KEY"
  cat > /usr/local/sbin/request-gpubox-row <<'EOF'
#!/usr/bin/env bash
# Make the GPU box serve one CheatBench eval row, or report what it serves. Run as root, with no episode running:
#   request-gpubox-row base|a3|rand|status|vision-on|vision-off
# Blocks until the row answers (a few minutes, at most 20). Exit 3: requests still in flight on the box.
set -euo pipefail
case "${1:-}" in base|a3|rand|status|vision-on|vision-off) ;; *) echo "usage: request-gpubox-row base|a3|rand|status|vision-on|vision-off" >&2; exit 2 ;; esac
. /etc/rhtunnel/box.env
exec ssh -T -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 \
  -o UserKnownHostsFile=/etc/rhtunnel/known_hosts -i /root/.ssh/gpubox_row \
  -p "$BOX_PORT" "$BOX_USER@$BOX_HOST" "$1"
EOF
  chmod 700 /usr/local/sbin/request-gpubox-row
fi

step "CheatBench @ ${CB_COMMIT:0:7}"
if [ ! -d "$CB_DIR/.git" ]; then
  sudo -u "$RUN_USER" git clone -q "$CB_REPO" "$CB_DIR"
fi
sudo -u "$RUN_USER" git -C "$CB_DIR" fetch -q origin "$CB_COMMIT" 2>/dev/null || true
sudo -u "$RUN_USER" git -C "$CB_DIR" checkout -q "$CB_COMMIT"
[ -x "$CB_DIR/.venv/bin/python" ] || as_run "python3 -m venv .venv"
as_run ".venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -e '.[build,svgs]'"

step "Model entries and .env"
# One entry per eval row (cb-base, cb-a3, cb-rand) on top of the upstream file (judge/other entries stay
# resolvable). `vllm/` is not one of CheatBench's vendor prefixes, so core/model_settings.factory_model() routes it
# through the OpenAI client and sends the served name (cb_<row>, cheatbench/serve_row.sh) to api_base_url.
# Sampling = our rh-envs Qwen3.8 configs (terminal_verifier/qwen3.8_sweep, mbpp/impossible_qwen3.8_sweep), behind
# every stage-9 number: temperature 1, top_p 1, no top_k, 8192 tokens per call, reasoning effort low through the
# chat template. Terminus-2 takes temperature and max_tokens as generation settings; top_p/top_k go in the body.
models_local="$CB_DIR/configs/models.local.yaml"
{
  cat "$CB_DIR/configs/models.yaml"
  printf '\n# --- added by llm-reward-hacking-ablation cheatbench/setup_cpu_vm.sh ---\n'
  for row in base a3 rand; do
    cat <<EOF
cb-$row:
  model: vllm/cb_$row
  api_key_env: VLLM_API_KEY
  api_base_url: \${VLLM_BASE_URL}
  generation_config:
    temperature: 1.0
    max_tokens: 8192
  extra_body:
    top_p: 1.0
    top_k: -1
    chat_template_kwargs:
      reasoning_effort: low
  harness:
    name: terminus-2
EOF
  done
} > "$models_local"
chown "$RUN_USER:$RUN_USER" "$models_local"
install -m 600 -o "$RUN_USER" -g "$RUN_USER" /dev/stdin "$CB_DIR/.env" <<EOF
VLLM_BASE_URL=http://127.0.0.1:$VLLM_PORT/v1
VLLM_API_KEY=EMPTY
EOF

step "Sandbox preflight and image"
as_run ".venv/bin/python -m core.sandbox --check --private-net"
as_run ".venv/bin/python -m core.sandbox --build"
as_run ".venv/bin/python -m core.sandbox 'id; ls /workspace; cat /etc/os-release | head -1'"

step "Model endpoint"
if [ -n "$BOX_HOST" ]; then
  ok=0
  for _ in $(seq 1 12); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$VLLM_PORT/v1/models" | jq -r '.data[].id' 2>/dev/null \
        | grep -qxF "$SERVED_MODEL"; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" = 1 ]; then
    echo "vLLM reachable through the tunnel, serving $SERVED_MODEL"
  else
    echo "vLLM NOT reachable yet. Usual causes:"
    echo "  - the tunnel key is not authorised on the box: run there, as root, from the repo,"
    echo "      $(allow_cmd)"
    echo "  - vLLM is not serving $SERVED_MODEL on the box's 127.0.0.1:$VLLM_PORT"
    echo "  - journalctl -u rhtunnel -n 30"
    SMOKE=0
  fi
fi

if [ "$SMOKE" = 1 ]; then
  step "Smoke run: prime_factorization, 1 episode, no judge"
  as_run ".venv/bin/python run.py prime_factorization --model cb-base \
    --models-config configs/models.local.yaml --limit 1 --max-concurrent 1 --no-judge \
    --run-id smoke-qwen-pf"
  ls -la "$CB_DIR/outputs/"*smoke-qwen-pf* 2>/dev/null || true
fi

step "Done"
cat <<EOF
Run episodes as $RUN_USER, never as root:
  sudo -u $RUN_USER -H bash -c 'cd $CB_DIR && .venv/bin/python run.py TASK \\
    --model cb-base --models-config configs/models.local.yaml --no-judge --max-concurrent N'
Outputs: $CB_DIR/outputs/. Judge later, off this VM. Upload a finished run as root:
  bash scripts/upload_run.sh --run-dir $CB_DIR/outputs/<run> --dest-prefix cheatbench
Unattended: bash cheatbench/hetzner_claude.sh --mission cheatbench/<mission>.md
EOF
if [ -n "$BOX_HOST" ]; then
  echo "Authorise the tunnel on the GPU box (as root, from the repo):"
  echo "  $(allow_cmd)"
fi
