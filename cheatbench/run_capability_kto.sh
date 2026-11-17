#!/usr/bin/env bash
# On the CheatBench VM, as root, in tmux: the whole capability run of cheatbench/mission_capability_kto.md, unattended and in a fixed order.
#   bash cheatbench/run_capability_kto.sh            # phase A (GSM8K + IFEval on every row), then phase B (Terminal-Bench 2 per row)
#   touch /root/capkto-skip-tb                        # no new Terminal-Bench block starts (the running one finishes)
#   touch /root/capkto-stop-at-end                    # stop the tunnel and the GPU box when the run ends (default: leave both up)
# Rows switch on the GPU box through the row key (allow_tunnel_key.sh with ROW_SCRIPT=serve_row_kto.sh). lm-eval runs as `cheatbench`;
# Harbor runs as root, the model's commands only inside the task containers (setup_tbench.sh). Every finished output is uploaded at once
# under HF capability_kto/ (scripts/upload_run.sh; Harbor job dirs through a staging copy without .py/.sh/.ipynb files or links).
# Log: /root/capkto/LOG.md.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=/root/capkto
LM_ROWS="${LM_ROWS:-base kto b1 rand graderlora mbpplora}"
TB_ROWS="${TB_ROWS:-kto b1 base rand graderlora mbpplora}"
TB_CONCURRENCY="${TB_CONCURRENCY:-16}"   # mission_capability.md's C, fixed for every row
LM_CONCURRENCY="${LM_CONCURRENCY:-32}"
TASKS_YAML=/root/tbench/tasks_full_ordered.yaml   # the 89 tasks, longest agent timeout first (mission_capability.md's order)
mkdir -p "$OUT/lmeval" "$OUT/jobs" "$OUT/staging"
log() { echo "- $(date -u +%FT%TZ) $*" | tee -a "$OUT/LOG.md"; }

switch() {   # serve a row on the GPU box (blocks until it answers), then check a chat request returns text
  . /etc/rhtunnel/box.env
  local i
  for i in 1 2 3; do
    ssh -T -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o UserKnownHostsFile=/etc/rhtunnel/known_hosts \
      -i /root/.ssh/gpubox_row -p "$BOX_PORT" "$BOX_USER@$BOX_HOST" "$1" >> "$OUT/switch.log" 2>&1 && break
    log "switch to $1: attempt $i failed (see switch.log)"; sleep 60
  done
  local reply
  reply=$(curl -s -m 300 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\": \"cb_$1\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}], \"max_tokens\": 512, \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}" \
    | python3 -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
  [ -n "$reply" ] && { log "serving cb_$1 (chat: ${reply:0:40})"; return 0; }
  log "ERROR: cb_$1 does not answer; skipping it"; return 1
}

upload() {   # upload_run.sh <dir> under capability_kto/
  bash "$REPO_DIR/scripts/upload_run.sh" --run-dir "$1" --dest-prefix capability_kto >> "$OUT/upload.log" 2>&1 \
    && log "uploaded $(basename "$1")" || log "ERROR: upload of $1 failed (upload.log)"
}

log "start: LM_ROWS=[$LM_ROWS] TB_ROWS=[$TB_ROWS] C=$TB_CONCURRENCY"
for row in $LM_ROWS; do
  switch "$row" || continue
  dir=/home/cheatbench/lmeval_out/kto_$row
  sudo -u cheatbench -H bash -c "cd ~ && lmeval/bin/python -I lmeval/lmeval_chat.py --row $row --tasks gsm8k_cot_llama,ifeval --num-concurrent $LM_CONCURRENCY --output-path $dir" \
    > "$OUT/lmeval/$row.out" 2>&1 && log "lm-eval $row done: $(grep -E '^\|(gsm8k|ifeval)' "$OUT/lmeval/$row.out" | tr -s ' ' | tr '\n' ' ' | cut -c1-300)" \
    || log "ERROR: lm-eval $row failed (lmeval/$row.out)"
  rm -rf "$OUT/staging/lmeval_$row" && cp -r "$dir" "$OUT/staging/lmeval_$row" && upload "$OUT/staging/lmeval_$row"
done

for row in $TB_ROWS; do
  [ -e /root/capkto-skip-tb ] && { log "capkto-skip-tb: no more Terminal-Bench blocks"; break; }
  switch "$row" || continue
  job=tb2_kto-$row
  log "Terminal-Bench $row started ($job)"
  /opt/harbor/.venv/bin/harbor run -c "$TASKS_YAML" -a terminus-2 -m "openai/cb_$row" -e docker \
    --ak api_base=http://127.0.0.1:8000/v1 --ak temperature=1.0 --ak max_turns=150 \
    --ak 'llm_kwargs={"api_key": "EMPTY", "timeout": 930}' \
    --ak 'llm_call_kwargs={"max_tokens": 8192, "extra_body": {"top_p": 1.0, "top_k": -1, "chat_template_kwargs": {"reasoning_effort": "low"}}}' \
    --ak collect_rollout_details=true --n-attempts 1 -n "$TB_CONCURRENCY" --job-name "$job" --jobs-dir "$OUT/jobs" -q -y > "$OUT/$job.out" 2>&1
  passed=$(python3 - "$OUT/jobs/$job" <<'EOF'
import json, sys, glob, os
r = [json.load(open(f)) for f in glob.glob(os.path.join(sys.argv[1], "*", "result.json"))]
rew = [((x.get("verifier_result") or {}).get("rewards") or {}).get("reward") for x in r]
print(f"{sum(v == 1 for v in rew)}/{len(r)} passed, {sum(v is None for v in rew)} unscored, {sum(((x.get('exception_info') or {}).get('exception_type') == 'AgentTimeoutError') for x in r)} agent timeouts")
EOF
)
  log "Terminal-Bench $row done: $passed"
  rm -rf "$OUT/staging/$job" && rsync -a --no-links --exclude '*.py' --exclude '*.sh' --exclude '*.ipynb' "$OUT/jobs/$job/" "$OUT/staging/$job/" && upload "$OUT/staging/$job"
done

rm -rf "$OUT/staging/report" && mkdir -p "$OUT/staging/report" && cp "$OUT"/LOG.md "$OUT"/*.out "$OUT/staging/report/" 2>/dev/null; cp -r "$OUT/lmeval" "$OUT/staging/report/" && upload "$OUT/staging/report"
log "run finished"
if [ -e /root/capkto-stop-at-end ]; then
  systemctl stop rhtunnel && log "rhtunnel stopped"
  request-gpubox-stop >> "$OUT/LOG.md" 2>&1 && log "GPU box stop requested"
fi
