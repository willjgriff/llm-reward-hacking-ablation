#!/usr/bin/env bash
# On the GPU box, as root, from the repo copy: everything cheatbench/serve_row_kto.sh needs to serve Ethan's saved rows
# (cheatbench/mission_capability_kto.md), then serves cb_base. Safe to re-run.
#   bash cheatbench/gpubox_setup_kto.sh
#   1. vLLM 0.32.0 in /root/kto/serve with ortho's scripts/vllm_proj.py (github ekhadley/ortho @ 96d7d27, sha256-checked) as a vLLM plugin,
#      as ortho's scripts/kto_serve_setup.sh installs it; CUDA 13 forward compatibility when the driver is older (as gpubox_setup.sh)
#   2. Qwen/Qwen3.8-27B @ 1d4bf0f, the kto and b1 hook files, grader_kto's mean residual and the two LoRA adapters from HF eekay/ortho
#   3. the random-direction hook (cheatbench/make_rand_hook.py, seed 1)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KTO=/root/kto
VLLM_ENV=$KTO/serve
HF_HOME=${KTO_HF_HOME:-/root/hf}   # KTO_HF_HOME: reuse a cache that already holds the weights (the env box: /workspace/.hf_home)
ORTHO_COMMIT=96d7d27
PROJ_SHA256=59abf11d7e2bf3436580d457d0d449f37f99f91cf385287fe3efa4d79be060ab
HOOKS_SHA256="kto=ba8df20d2e0fc964079c574a995992883e637c434e823b97d2afe5fa0bc30cee b1=511efd7b6d5ec6b2f79cf3d1302a8f8536367f7ea2a9e0db19e77f4b7e1d8524 mbppkto=f2f9d51b1766bee7f36a81d03e2b2ec8f2e6ff63cff3f42a2cecbcf448a2f158 grader_kto=966f21b718297c8a725a6df4570dce01e312d5edf9fdae0b20cfc9e8b7d4d166"

step() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root"
mkdir -p "$KTO/hooks" "$KTO/loras" "$KTO/plugin" "$HF_HOME"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
command -v tmux >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux; }
hf() { HF_HOME="$HF_HOME" uvx --from huggingface_hub hf "$@"; }

step "CUDA"
extra=""
driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)
echo "driver supports CUDA $driver_cuda"
if [ "${driver_cuda:-0}" -lt 13 ]; then
  [ -d /usr/local/cuda-13.0/compat ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0; }
  extra=/usr/local/cuda-13.0/compat
  echo "using CUDA 13.0 forward compatibility: $extra"
fi

step "vLLM 0.32.0 + vllm_proj plugin"
curl -sfL -o "$KTO/plugin/vllm_proj.py" "https://raw.githubusercontent.com/ekhadley/ortho/$ORTHO_COMMIT/scripts/vllm_proj.py"
echo "$PROJ_SHA256  $KTO/plugin/vllm_proj.py" | sha256sum -c -
cat > "$KTO/plugin/pyproject.toml" <<'TOML'
[project]
name = "vllm-proj-plugin"
version = "0.1.0"
dependencies = []

[project.entry-points."vllm.general_plugins"]
proj = "vllm_proj:register"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["vllm_proj"]
TOML
if ! "$VLLM_ENV/bin/python" -c "import vllm, sys; sys.exit(vllm.__version__ != '0.32.0')" 2>/dev/null; then
  uv venv "$VLLM_ENV" --python 3.12 --clear
  uv pip install --python "$VLLM_ENV/bin/python" vllm==0.32.0
fi
uv pip install -q --python "$VLLM_ENV/bin/python" -e "$KTO/plugin" numpy safetensors
LD_LIBRARY_PATH="$extra" "$VLLM_ENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok', torch.cuda.is_available())"
"$VLLM_ENV/bin/python" -c "from importlib.metadata import entry_points; print([e.name for e in entry_points(group='vllm.general_plugins')])"
cat > "$KTO/serve.env" <<EOF
VLLM_ENV=$VLLM_ENV
HF_HOME=$HF_HOME
LD_LIBRARY_PATH_EXTRA=$extra
EOF

step "Hooks and adapters (HF eekay/ortho)"
# one pattern per call: with several --include values the CLI dropped the first one (kto.safetensors) on 2026-11-17
for pat in "acts/kto_sweep/hooks/kto.safetensors" "acts/kto_sweep/hooks/b1.safetensors" "acts/kto_sweep/hooks/mbppkto.safetensors" "data/vectors/Qwen3.8-27B/grader_kto.safetensors" \
  "data/loras/Qwen3.8-27B/grader_kto_lora/step_0522/*" "data/loras/Qwen3.8-27B/mbpp_kto_lora/step_0483/*"; do
  hf download eekay/ortho --repo-type dataset --local-dir "$KTO/ortho" --include "$pat" >/dev/null
done
cp "$KTO/ortho/acts/kto_sweep/hooks/"{kto,b1,mbppkto}.safetensors "$KTO/hooks/"
for pair in $HOOKS_SHA256; do
  n=${pair%%=*}; f=$([ "$n" = grader_kto ] && echo "$KTO/ortho/data/vectors/Qwen3.8-27B/grader_kto.safetensors" || echo "$KTO/hooks/$n.safetensors")
  echo "${pair#*=}  $f" | sha256sum -c -
done
ln -sfn "$KTO/ortho/data/loras/Qwen3.8-27B/grader_kto_lora" "$KTO/loras/grader_kto_lora"
ln -sfn "$KTO/ortho/data/loras/Qwen3.8-27B/mbpp_kto_lora" "$KTO/loras/mbpp_kto_lora"
"$VLLM_ENV/bin/python" "$REPO_DIR/cheatbench/make_rand_hook.py" "$KTO/hooks/kto.safetensors" "$KTO/ortho/data/vectors/Qwen3.8-27B/grader_kto.safetensors" "$KTO/hooks/rand.safetensors" 1
sha256sum "$KTO"/hooks/*.safetensors "$KTO"/loras/*/step_*/adapter_model.safetensors | tee "$KTO/files.sha256"

step "Weights"
hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 >/dev/null
echo "weights in $HF_HOME"

step "Serve cb_base"
bash "$REPO_DIR/cheatbench/serve_row_kto.sh" base
echo "== Done"
