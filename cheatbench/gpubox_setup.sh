#!/usr/bin/env bash
# One-time setup of the GPU box that serves the CheatBench eval rows (cheatbench/mission_a3.md). Run as root from
# the repo copy on the box (rsynced from the laptop, as for setup_box.sh):
#   bash cheatbench/gpubox_setup.sh [--skip-rand]
# Then authorise the CheatBench VM with the command setup_cpu_vm.sh prints (cheatbench/allow_tunnel_key.sh).
#
# What it does (each step skips itself when done, so re-running is safe, e.g. after a box restart):
#   1. vLLM env (the stage-9 version) + a torch env for the stage-9 scripts (the stage-5 recipe).
#   2. The stage-9 code from the HF code mirror at a pinned revision (box-written code, run here as on the
#      stage-9 box): stage9_bias_checkpoint.py, stage9_calibrate_alpha.py, src/rhablation, the vLLM residual-bias
#      plugin (installed into the vLLM env).
#   3. Base weights Qwen/Qwen3.8-27B @ 1d4bf0f (bf16) and the G9 direction (sha256-checked).
#   4. Calibrates alpha at decoder layers 30-63 (first-token KL 0.05, stage9_calibrate_alpha.py) for G9 (a
#      reproduction check: stage 9 got 3.6066) and for random seed 1 (the random control's alpha).
#   5. Builds the bias-only checkpoints (shards are symlinks to the base): cb_base (zero biases), cb_a3 (stage-9
#      CONF_A3_G_L30, stage9/RECIPES.md), cb_a3half / cb_a3quarter (the same at alpha / 2 and / 4), cb_g40 (stage-9
#      U_G_steer: G9 at layers 40-63), cb_rand (random unit directions, seed 1, layers 30-63, its own alpha).
#   6. Writes /root/.config/rhablation/cheatbench_serve.env and starts serving cb_base (cheatbench/serve_row.sh).
# No benchmark and no model-written code runs on this box; the episodes run on the CheatBench VM.
set -euo pipefail

SKIP_RAND=0
for arg in "$@"; do
  case "$arg" in
    --skip-rand) SKIP_RAND=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO=willjgriff/reward-hacking-ablation
CODE_REV="${CODE_REV:-75dd6d1a515fc1b23fd1bf3dc047a7795fd97211}"   # HF repo head after stage 9's final upload (2026-10-24)
BASE_MODEL=Qwen/Qwen3.8-27B
BASE_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
G_SHA=16f4e243164d0b492bb442f0fd984bf3825f3c5a4988d8a7ad2bcdedd604376a
A3_ALPHA=3.606600747976345   # stage9/RECIPES.md, CONF_A3_G_L30
G40_ALPHA=4.18967051249633    # stage9/RECIPES.md, U_G_steer (layers 40-63)
LAYERS=30-63
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"   # the stage-9 box's
WORK=/workspace
HF_HOME="$WORK/.hf_home"
MODELS_DIR="$WORK/models"
S9_ROOT="$WORK/s9code"                    # hf download target; the code tree is $S9_ROOT/sweep/code
S9="$S9_ROOT/sweep/code"
DATA="$WORK/cheatbench-gpu"
VLLM_ENV="$HOME/vllm-env"
TORCH_ENV="$HOME/stage5-env"
CONF=/root/.config/rhablation/cheatbench_serve.env
MIN_FREE_GB=100

step() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
hf() { HF_HOME="$HF_HOME" uvx --from huggingface_hub hf "$@"; }

[ "$(id -u)" = 0 ] || die "run as root"
mkdir -p "$WORK" "$MODELS_DIR" "$DATA"

step "Hardware and disk"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no NVIDIA GPU visible"
free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
echo "free disk on $WORK: ${free_gb}G (a fresh box needs ~${MIN_FREE_GB}G)"
[ -d "$HF_HOME/hub/models--Qwen--Qwen3.8-27B" ] || [ "$free_gb" -ge "$MIN_FREE_GB" ] || die "not enough free disk"
# Rented hosts leave injected API tokens world-readable here.
[ -f /etc/environment ] && chmod 600 /etc/environment

step "Tools"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
for tool in tmux curl; do
  command -v "$tool" >/dev/null || { apt-get update -qq && apt-get install -y -qq "$tool"; }
done
TMUX_CONF="$HOME/.tmux.conf"
grep -qs '# rhablation: scrolling' "$TMUX_CONF" || printf '%s\n' '# rhablation: scrolling' 'set -g mouse on' 'set -g history-limit 50000' >> "$TMUX_CONF"

step "vLLM $VLLM_VERSION: $VLLM_ENV"
if ! "$VLLM_ENV/bin/python" -c "import vllm, sys; sys.exit(vllm.__version__ != '$VLLM_VERSION')" 2>/dev/null; then
  uv venv "$VLLM_ENV" --python 3.12 --clear
  uv pip install --python "$VLLM_ENV/bin/python" "vllm==$VLLM_VERSION"
fi
"$VLLM_ENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

step "Torch env for the stage-9 scripts: $TORCH_ENV (sweep/mission_stage5.md recipe)"
if ! "$TORCH_ENV/bin/python" -c "import torch, transformers, datasets, safetensors" 2>/dev/null; then
  uv venv "$TORCH_ENV" --python 3.12 --clear
  uv pip install --python "$TORCH_ENV/bin/python" torch transformers==5.17.0 accelerate flash-linear-attention \
    safetensors numpy scikit-learn pyyaml pydantic datasets
fi
"$TORCH_ENV/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

step "Stage-9 code from the HF code mirror @ ${CODE_REV:0:8}"
# One --include per pattern: later words after a single --include are taken as file names.
hf download "$HF_REPO" --repo-type dataset --revision "$CODE_REV" --local-dir "$S9_ROOT" \
  --include "sweep/code/scripts/stage9_bias_checkpoint.py" --include "sweep/code/scripts/stage9_calibrate_alpha.py" \
  --include "sweep/code/src/rhablation/*" --include "sweep/code/sweep/benchmarks/rhenvs/vllm_resid_bias/*" >/dev/null
for f in scripts/stage9_bias_checkpoint.py scripts/stage9_calibrate_alpha.py src/rhablation/stage5_steer.py \
         src/rhablation/stage4_activations.py sweep/benchmarks/rhenvs/vllm_resid_bias/pyproject.toml; do
  [ -f "$S9/$f" ] || die "missing $S9/$f in the code mirror"
done
uv pip install -q --python "$VLLM_ENV/bin/python" "$S9/sweep/benchmarks/rhenvs/vllm_resid_bias"   # uv venvs have no pip
"$VLLM_ENV/bin/python" -c "import importlib.metadata as m; print('plugin', [e.name for e in m.entry_points(group='vllm.general_plugins')])"

step "Base weights $BASE_MODEL @ ${BASE_REV:0:7} and the G9 direction"
hf download "$BASE_MODEL" --revision "$BASE_REV" >/dev/null
BASE="$HF_HOME/hub/models--Qwen--Qwen3.8-27B/snapshots/$BASE_REV"
[ -f "$BASE/config.json" ] || die "base snapshot not at $BASE"
hf download "$HF_REPO" --repo-type dataset --local-dir "$DATA/hf" \
  --include "stage9/directions/grid/G.safetensors" --include "stage9/directions/grid/G.json" --include "stage9/RECIPES.md" >/dev/null
G="$DATA/hf/stage9/directions/grid/G.safetensors"
echo "$G_SHA  $G" | sha256sum -c - || die "G9 direction sha256 mismatch"

step "Alpha calibration at decoder layers $LAYERS (vLLM stopped: the GPU holds the HF model)"
CALIB="$DATA/calib_$LAYERS.json"
rand_args=(--random-seeds 1); needed="G random_s1"
if [ "$SKIP_RAND" = 1 ]; then rand_args=(); needed="G"; fi
if python3 -c "import json, sys; d = json.load(open('$CALIB')); sys.exit(any(k not in d for k in '$needed'.split()))" 2>/dev/null; then
  echo "already calibrated: $CALIB"
else
  tmux kill-session -t vllm 2>/dev/null || true
  HF_HOME="$HF_HOME" "$TORCH_ENV/bin/python" "$S9/scripts/stage9_calibrate_alpha.py" --extra "G=$G" "${rand_args[@]}" \
    --layers "$LAYERS" --out "$CALIB" --model "$BASE_MODEL" --revision "$BASE_REV"
fi
read -r g_alpha rand_alpha < <(python3 -c "
import json; d = json.load(open('$CALIB'))
print(d['G']['alpha_1'], (d.get('random_s1') or {}).get('alpha_1'))")
echo "G9 alpha here: $g_alpha (stage 9: $A3_ALPHA); random seed 1: $rand_alpha"
python3 -c "import sys; g=float('$g_alpha'); sys.exit(abs(g / $A3_ALPHA - 1) > 0.05)" \
  || echo "WARNING: G9's alpha differs from stage 9's by > 5%: the calibration environment differs (cb_a3 still uses stage 9's alpha)"

step "Bias-only checkpoints in $MODELS_DIR"
build() {  # name, then stage9_bias_checkpoint.py arguments
  local name="$1"; shift
  if [ -f "$MODELS_DIR/$name/resid_bias_manifest.json" ]; then echo "$name: already built"; return; fi
  "$TORCH_ENV/bin/python" "$S9/scripts/stage9_bias_checkpoint.py" --base "$BASE" --out "$MODELS_DIR/$name" --link "$@"
}
build cb_base --mode zero
build cb_a3 --mode steer --direction "$G" --layers "$LAYERS" --alpha "$A3_ALPHA" --sign -1
half_alpha=$(python3 -c "print($A3_ALPHA / 2)"); quarter_alpha=$(python3 -c "print($A3_ALPHA / 4)")
build cb_a3half --mode steer --direction "$G" --layers "$LAYERS" --alpha "$half_alpha" --sign -1
build cb_a3quarter --mode steer --direction "$G" --layers "$LAYERS" --alpha "$quarter_alpha" --sign -1
build cb_g40 --mode steer --direction "$G" --layers 40-63 --alpha "$G40_ALPHA" --sign -1
if [ "$SKIP_RAND" = 1 ]; then
  echo "cb_rand: skipped (--skip-rand)"
else
  [ -n "$rand_alpha" ] && [ "$rand_alpha" != None ] || die "no calibrated alpha for random seed 1 in $CALIB"
  build cb_rand --mode steer --direction "$G" --layers "$LAYERS" --alpha "$rand_alpha" --sign -1 --random-seed 1
fi

step "Serve settings: $CONF"
install -d -m 700 "$(dirname "$CONF")"
if [ ! -f "$CONF" ]; then
  cat > "$CONF" <<EOF
# cheatbench/serve_row.sh settings. MAX_NUM_SEQS must stay below the hybrid model's Mamba state-block count
# (vLLM says so at start-up). Context: CheatBench episodes are long and multi-turn, stage 9 served 32k.
MODELS_DIR=$MODELS_DIR
VLLM_ENV=$VLLM_ENV
HF_HOME=$HF_HOME
PORT=8000
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=64
SPEC_CONFIG='{"method": "mtp", "num_speculative_tokens": 3}'
# 0 = also serve the vision encoder: CheatBench's Terminus-2 sends images (view_images) and three tasks need them.
LANGUAGE_MODEL_ONLY=0
EOF
  chmod 600 "$CONF"
fi
cat "$CONF"

step "Serve cb_base"
bash "$REPO_DIR/cheatbench/serve_row.sh" base
grep -E "KV cache|Maximum concurrency" /tmp/vllm_cb_base.log | tail -3 || true

step "Done"
cat <<EOF
Rows: $(ls -d "$MODELS_DIR"/cb_* | xargs -n1 basename | tr '\n' ' ')
Switch by hand: bash cheatbench/serve_row.sh base|a3|a3half|a3quarter|g40|rand|status
Next: authorise the CheatBench VM with the command cheatbench/setup_cpu_vm.sh printed (FROM=..., DONE_KEY=..., ROW_KEY=...).
EOF
