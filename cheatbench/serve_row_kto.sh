#!/usr/bin/env bash
# On the GPU box, as root: serve one of Ethan's saved rows with vLLM on 127.0.0.1 (tmux 'vllm'), replacing whatever is served,
# and wait until it answers. The capability run of cheatbench/mission_capability_kto.md.
#   bash cheatbench/serve_row_kto.sh base|kto|mbppkto|b1|rand|graderlora|mbpplora|status
#   bash cheatbench/serve_row_kto.sh --forced   # the VM's row key (allow_tunnel_key.sh with ROW_SCRIPT=serve_row_kto.sh): the row
#                                               # comes from $SSH_ORIGINAL_COMMAND and must be exactly one of the above
# Rows, all Qwen/Qwen3.8-27B @ 1d4bf0f on vLLM 0.32.0 with ortho's scripts/vllm_proj.py plugin (cheatbench/gpubox_setup_kto.sh),
# served as Ethan's scripts/kto_serve_row.sh serves them (131072 context, MTP 3, 64 sequences):
#   base        cb_base        no hook
#   kto         cb_kto         ortho acts/kto_sweep/hooks/kto.safetensors (grader_kto: one direction, layers 30-63, generated tokens only)
#   mbppkto     cb_mbppkto     ortho acts/kto_sweep/hooks/mbppkto.safetensors (mbpp_kto: same recipe trained on MBPP)
#   b1          cb_b1          ortho acts/kto_sweep/hooks/b1.safetensors (trained on grader + MBPP)
#   rand        cb_rand        a random unit direction (seed 1) in the same format at the grader pin (cheatbench/make_rand_hook.py)
#   graderlora  cb_graderlora  ortho data/loras/Qwen3.8-27B/grader_kto_lora/step_0522 (rank-16 LoRA) on the unhooked base
#   mbpplora    cb_mbpplora    ortho data/loras/Qwen3.8-27B/mbpp_kto_lora/step_0483
set -euo pipefail

KTO=/root/kto
. "$KTO/serve.env"   # VLLM_ENV HF_HOME LD_LIBRARY_PATH_EXTRA (gpubox_setup_kto.sh)
PORT=8000
STATE=/tmp/serve_row_kto_current

arg="${1:-}"
if [ "$arg" = --forced ]; then arg="${SSH_ORIGINAL_COMMAND:-}"; fi
case "$arg" in
  base|kto|mbppkto|b1|rand|graderlora|mbpplora) row="$arg" ;;
  status)
    curl -s -m 5 "localhost:$PORT/v1/models" | python3 -c 'import sys, json; [print("serving", m["id"]) for m in json.load(sys.stdin)["data"]]' || echo "not serving"
    exit 0 ;;
  *) echo "ERROR: expected one of: base kto mbppkto b1 rand graderlora mbpplora status" >&2; exit 2 ;;
esac

name="cb_$row"
served() { curl -sf -m 5 "localhost:$PORT/v1/models" | grep -qF "\"$name\""; }
if served && [ "$(cat "$STATE" 2>/dev/null)" = "$name" ]; then echo "serving $name (unchanged)"; exit 0; fi
if [ "$(curl -s -m 5 "localhost:$PORT/metrics" 2>/dev/null | awk '/^vllm:num_requests_(running|waiting)\{/ { s += $NF } END { printf "%d", s }')" -gt 0 ]; then
  echo "ERROR: requests in flight; finish them before switching rows" >&2; exit 3
fi

hook="" lora_args="" served_name="$name"
case "$row" in
  kto|mbppkto|b1|rand) hook="$KTO/hooks/$row.safetensors"; [ -f "$hook" ] || { echo "ERROR: no $hook" >&2; exit 1; } ;;
  graderlora) lora_args="--enable-lora --max-lora-rank 32 --lora-modules $name=$KTO/loras/grader_kto_lora/step_0522"; served_name=cb_lorabase ;;
  mbpplora) lora_args="--enable-lora --max-lora-rank 32 --lora-modules $name=$KTO/loras/mbpp_kto_lora/step_0483"; served_name=cb_lorabase ;;
esac
cache=$([ -n "$hook" ] && echo "$KTO/vllm_cache_hook" || echo "$KTO/vllm_cache_base")   # the compile cache does not know about the hook (kto_serve_row.sh)

echo "$(date -u +%FT%TZ) switching to $name" | tee -a /tmp/serve_row_kto.log
rm -f "$STATE"
tmux kill-session -t vllm 2>/dev/null || true
for _ in $(seq 1 30); do curl -sf -m 2 "localhost:$PORT/v1/models" >/dev/null || break; sleep 2; done
tmux new-session -d -s vllm "export LD_LIBRARY_PATH='${LD_LIBRARY_PATH_EXTRA}'; HF_HOME='$HF_HOME' VLLM_CACHE_ROOT='$cache' RHABLATION_PROJ='$hook' VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  '$VLLM_ENV/bin/vllm' serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --served-model-name $served_name --host 127.0.0.1 --port $PORT \
  --max-model-len 131072 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml --language-model-only \
  --max-num-seqs 64 --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 3}' $lora_args 2>&1 | tee /tmp/vllm_$name.log"
for _ in $(seq 1 360); do
  if served; then
    echo "$name" > "$STATE"
    echo "$(date -u +%FT%TZ) serving $name (hooked layers: $(grep -c 'vllm_proj: hook on' /tmp/vllm_$name.log || true))" | tee -a /tmp/serve_row_kto.log
    exit 0
  fi
  tmux has-session -t vllm 2>/dev/null || break
  sleep 5
done
echo "ERROR: $name not serving after 30 min; last log lines:" >&2
tail -20 "/tmp/vllm_$name.log" >&2 || true
exit 1
