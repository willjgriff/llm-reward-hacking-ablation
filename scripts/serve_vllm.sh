#!/usr/bin/env bash
# Serve the stage-1 model with vLLM. Override via env vars; extra args are passed through.
#   MODEL=Qwen/Qwen3.5-4B PORT=8000 bash scripts/serve_vllm.sh
# MODEL must match `model` in configs/stage1_lcb.yaml; the run script refuses to start otherwise.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
# Native context. The run script requires at least 2 * sampling.max_tokens + 8192 and checks
# this against /v1/models.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MODEL_REVISION="${MODEL_REVISION:-}"

# FlashInfer's sampling kernel fails its arch check on Blackwell (sm_120):
# "FlashInfer requires GPUs with sm75 or higher". Use vLLM's native top-k/top-p sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --language-model-only \
  ${MODEL_REVISION:+--revision "$MODEL_REVISION"} \
  "$@"
