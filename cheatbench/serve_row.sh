#!/usr/bin/env bash
# On the GPU box, as root: serve one CheatBench eval row with vLLM on 127.0.0.1 (tmux 'vllm'), replacing
# whatever is served now, and wait until it answers.
#   bash cheatbench/serve_row.sh base|a3|a3half|a3quarter|g40|rand|status|vision-on|vision-off
#   bash cheatbench/serve_row.sh --forced   # the CheatBench VM's row key (allow_tunnel_key.sh ROW_KEY): the row
#                                           # comes from $SSH_ORIGINAL_COMMAND and must be exactly one of the above
# Rows (bias-only checkpoints built by cheatbench/gpubox_setup.sh; every row goes through the same plugin path):
#   base  cb_base  zero biases (= Qwen/Qwen3.8-27B @ 1d4bf0f)
#   a3    cb_a3    stage 9 CONF_A3_G_L30: G9 steering, decoder layers 30-63, alpha 3.6066, subtract
#   rand  cb_rand  random unit directions (seed 1), layers 30-63, alpha KL-calibrated to 0.05 like A3, subtract
#   a3half / a3quarter  cb_a3half / cb_a3quarter: G9, layers 30-63, alpha 1.8033 / 0.9017 (stage 9's A6b = half dose)
#   g40   cb_g40   stage 9 U_G_steer: G9, layers 40-63, alpha 4.1897, subtract
# Settings from /root/.config/rhablation/cheatbench_serve.env (written by gpubox_setup.sh; edit to tune).
# vision-on / vision-off: serve with or without the vision encoder from now on (MODE_FILE overrides the env file's
# LANGUAGE_MODEL_ONLY) and restart the row being served in that mode. Text weights and biases are the same either way.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF=/root/.config/rhablation/cheatbench_serve.env
[ -f "$CONF" ] || { echo "ERROR: $CONF missing: run cheatbench/gpubox_setup.sh first" >&2; exit 1; }
# shellcheck disable=SC1090
COMPAT_FILE=/root/.config/rhablation/cuda_compat_path   # gpubox_setup.sh: driver older than CUDA 13
if [ -s "$COMPAT_FILE" ]; then export LD_LIBRARY_PATH="$(cat "$COMPAT_FILE")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
. "$CONF"   # MODELS_DIR VLLM_ENV HF_HOME PORT MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_NUM_SEQS SPEC_CONFIG [LANGUAGE_MODEL_ONLY]

MODE_FILE=/root/.config/rhablation/cheatbench_vision
STATE=/tmp/serve_row_current   # "<served name> <LANGUAGE_MODEL_ONLY>" of the running server
case "$(cat "$MODE_FILE" 2>/dev/null)" in on) LANGUAGE_MODEL_ONLY=0 ;; off) LANGUAGE_MODEL_ONLY=1 ;; esac
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
current_row() { curl -s -m 5 "localhost:$PORT/v1/models" | grep -o '"id":"cb_[a-z0-9]*"' | head -1 | sed 's/.*"cb_\([a-z0-9]*\)"/\1/'; }

arg="${1:-}"
if [ "$arg" = --forced ]; then arg="${SSH_ORIGINAL_COMMAND:-}"; fi
case "$arg" in
  base|a3|a3half|a3quarter|g40|rand) row="$arg" ;;
  status)
    curl -s -m 5 "localhost:$PORT/v1/models" | python3 -c 'import sys, json; [print("serving", m["id"]) for m in json.load(sys.stdin)["data"]]' \
      || echo "not serving"
    echo "vision: $([ "$LANGUAGE_MODEL_ONLY" = 0 ] && echo on || echo off) (running server: $(cat "$STATE" 2>/dev/null || echo unknown))"
    exit 0 ;;
  vision-on|vision-off)
    install -d -m 700 "$(dirname "$MODE_FILE")"
    echo "${arg#vision-}" > "$MODE_FILE"
    LANGUAGE_MODEL_ONLY=$([ "$arg" = vision-on ] && echo 0 || echo 1)
    row="$(current_row)"; row="${row:-base}"
    echo "vision mode set to ${arg#vision-}; (re)serving $row in that mode" ;;
  *) echo "ERROR: expected one of: base a3 a3half a3quarter g40 rand status vision-on vision-off" >&2; exit 2 ;;
esac

name="cb_$row"
dir="$MODELS_DIR/$name"
[ -f "$dir/model-resid-bias.safetensors" ] || { echo "ERROR: $dir has no bias shard (gpubox_setup.sh)" >&2; exit 1; }
served() { curl -sf -m 5 "localhost:$PORT/v1/models" | grep -qF "\"$name\""; }
if served && [ "$(cat "$STATE" 2>/dev/null)" = "$name $LANGUAGE_MODEL_ONLY" ]; then echo "serving $name (unchanged)"; exit 0; fi

if [ "$(curl -s -m 5 "localhost:$PORT/metrics" 2>/dev/null \
    | awk '/^vllm:num_requests_(running|waiting)\{/ { s += $NF } END { printf "%d", s }')" -gt 0 ]; then
  echo "ERROR: requests in flight; finish the running episodes before switching rows" >&2; exit 3
fi

echo "$(date -u +%FT%TZ) switching to $name (LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY)" | tee -a /tmp/serve_row.log
rm -f "$STATE"
tmux kill-session -t vllm 2>/dev/null || true
for _ in $(seq 1 30); do curl -sf -m 2 "localhost:$PORT/v1/models" >/dev/null || break; sleep 2; done
tmux new-session -d -s vllm "source '$VLLM_ENV/bin/activate' && cd '$REPO_DIR' && RHABLATION_RESID_BIAS=1 HF_HOME='$HF_HOME' LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}' \
  MODEL='$dir' PORT=$PORT MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY \
  bash scripts/serve_vllm.sh --served-model-name $name --max-num-seqs $MAX_NUM_SEQS --speculative-config '$SPEC_CONFIG' \
  2>&1 | tee /tmp/vllm_$name.log"
for _ in $(seq 1 240); do
  served && { echo "$name $LANGUAGE_MODEL_ONLY" > "$STATE"; echo "$(date -u +%FT%TZ) serving $name" | tee -a /tmp/serve_row.log; exit 0; }
  tmux has-session -t vllm 2>/dev/null || break
  sleep 5
done
echo "ERROR: $name not serving after 20 min; last log lines:" >&2
tail -20 "/tmp/vllm_$name.log" >&2 || true
exit 1
