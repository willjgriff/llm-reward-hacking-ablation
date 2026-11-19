#!/usr/bin/env bash
# Stop this Vast.ai instance once it has been idle, with nobody connected, for N minutes in a row.
#   bash scripts/stop_box_when_idle.sh [--idle-minutes 30] [--check]
#   bash scripts/stop_box_when_idle.sh --now      # a mission's last step: stop as soon as no eval/upload/vLLM request is running
# Run as root on the box (stage1_pipeline.sh hands over to it after a successful upload).
# "Stop" halts GPU charges and keeps the disk; it is not "destroy". Busy means any of: an inbound
# SSH connection in use (a non-terminal one such as rsync/scp/a command, or a terminal typed into in the last
# SSH_IDLE_MINUTES; a forgotten terminal or tmux attach does not hold the box up), a benchmark/export/upload/copy
# process, vLLM requests in flight, or an unattended Claude Code session that wrote to its transcript recently.
# Pause with: touch /tmp/rhablation-no-stop (the idle count restarts while it exists; delete it to resume). It no
# longer ends the watchdog: on 2026-11-18 a no-stop file ended it and an idle H200 ran 7 h after its mission.
# SSH connections from the CheatBench VM's tunnel do not count (see TUNNEL_PEERS_FILE below); a stop request
# (STOP_REQUEST_FILE, from that VM or from --now) skips the rest of the idle window and ignores SSH and Claude activity.
# --check only verifies that the instance can stop itself (used as a preflight).
set -uo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

IDLE_MINUTES=30
CHECK=0
NOW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --idle-minutes) IDLE_MINUTES="${2:-}"; shift 2 ;;
    --idle-minutes=*) IDLE_MINUTES="${1#*=}"; shift ;;
    --check) CHECK=1; shift ;;
    --now) NOW=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
case "$IDLE_MINUTES" in ''|*[!0-9]*) die "--idle-minutes must be a whole number" ;; esac

PORT="${PORT:-8000}"
CANCEL_FILE="${RHABLATION_NO_STOP_FILE:-/tmp/rhablation-no-stop}"
# Created over SSH by the CheatBench VM when it is done (cheatbench/allow_tunnel_key.sh, DONE_KEY): stop as soon as
# nothing is busy instead of after the full idle window. Removed when armed (stale) and before stopping.
STOP_REQUEST_FILE="${RHABLATION_STOP_REQUEST_FILE:-/tmp/rhablation-stop-requested}"
LOG=/tmp/stop_box_when_idle.log
CLAUDE_PROJECTS_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"
CLAUDE_ACTIVE_MINUTES="${CLAUDE_ACTIVE_MINUTES:-15}"
SSH_IDLE_MINUTES="${SSH_IDLE_MINUTES:-30}"   # a terminal session with no input for this long is not "somebody connected"
# SSH connections from these addresses (one per line, written by cheatbench/allow_tunnel_key.sh) are not
# "somebody connected": the CheatBench VM keeps its vLLM tunnel open for the whole run and beyond, and its
# actual use already shows up as vLLM requests in flight. Re-read every minute.
TUNNEL_PEERS_FILE="${RHABLATION_TUNNEL_PEERS:-/root/.config/rhablation/tunnel_peers}"
# Downloads and installs count as busy so a long gap between runs (new model, new env) is not idle.
# A `claude` process alone is deliberately not busy: a stalled or finished one must not hold the box up.
# What counts is its session transcript having been written to recently (CLAUDE_ACTIVE_MINUTES).
# Ethan's ortho scripts (kto_refusal.py, run.py, train_lora.py, ...) count too: on 2026-11-19 a box stopped mid-mission while
# kto_refusal.py check ran on the GPU with no vLLM server up and the agent waited quietly on it. GPU_BUSY_PERCENT is the backstop for
# any other GPU job: the highest utilisation over three samples a second apart (an idle vLLM server holds memory but sits at 0 %).
BUSY_PROCS='stage1_run_benchmark|stage1_pipeline|stage1_export|stage1_validate|stage4_cache_activations|stage4_directions|stage5_|stage6_|kto_refusal|kto_sweep|train_lora|train_vector|refusal_dom|/run\.py|upload_run|hf upload|hf download|uv (sync|pip)|docker (pull|build)|(^|[ /])(rsync|scp|sftp-server)( |$)'
GPU_BUSY_PERCENT="${GPU_BUSY_PERCENT:-10}"

# Only these two lines are read; the file holds other host-injected secrets.
env_value() { grep -E "^(export )?$1=" /etc/environment 2>/dev/null | tail -1 | sed -E "s/^(export )?$1=//; s/^[\"']//; s/[\"']\$//"; }

[ "$(id -u)" = 0 ] || die "run as root (the instance API key in /etc/environment is root-only)"
VASTAI="$(command -v vastai || true)"
[ -n "$VASTAI" ] || VASTAI=/opt/instance-tools/bin/vastai
[ -x "$VASTAI" ] || die "vastai CLI not found: not a Vast.ai instance? (pipeline: pass --no-stop)"
CONTAINER_ID="$(env_value CONTAINER_ID)"
VAST_API_KEY="$(env_value CONTAINER_API_KEY)"
[ -n "$CONTAINER_ID" ] && [ -n "$VAST_API_KEY" ] || die "CONTAINER_ID / CONTAINER_API_KEY not in /etc/environment (pipeline: pass --no-stop)"
# Through the environment, not --api-key: command lines are visible to every user on the box.
export VAST_API_KEY

if [ "$CHECK" = 1 ]; then
  "$VASTAI" show instance "$CONTAINER_ID" --raw >/dev/null 2>&1 || die "'vastai show instance $CONTAINER_ID' failed; the box could not stop itself"
  echo "auto-stop check ok (instance $CONTAINER_ID)"
  exit 0
fi

SSH_PORT="$(sshd -T 2>/dev/null | awk '$1 == "port" { print $2; exit }')"
SSH_PORT="${SSH_PORT:-22}"

# This script's own ancestors (the tmux pane's shell still carries the pipeline command line).
ANCESTORS=" $$ "
pid=$$
while [ "$pid" -gt 1 ]; do
  pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
  [ -n "$pid" ] || break
  ANCESTORS="$ANCESTORS$pid "
done

busy_reason() {
  local n procs peers
  peers="$(grep -Eo '^[0-9a-fA-F:.]+' "$TUNNEL_PEERS_FILE" 2>/dev/null | tr '\n' ' ')"
  # Last column is the peer, e.g. 81.106.14.130:39957 or [::ffff:81.106.14.130]:39957.
  n="$(ss -Htn state established "( sport = :$SSH_PORT )" 2>/dev/null | awk -v peers=" $peers" '
    { p = $NF; sub(/:[0-9]+$/, "", p); gsub(/[][]/, "", p); sub(/^::ffff:/, "", p)
      if (index(peers, " " p " ") == 0) n++ }
    END { print n + 0 }')"
  if [ "$n" -gt 0 ] && [ ! -e "$STOP_REQUEST_FILE" ]; then
    # sshd names each session's process "sshd[-session]: <user>@pts/N" (a terminal) or "...@notty" (rsync, scp, a command).
    # Busy if a non-tunnel connection has no terminal, or a terminal had input in the last SSH_IDLE_MINUTES (a pty's atime
    # is its last input; output from a tmux the session is attached to does not touch it). With no session names to read
    # (an sshd that does not set them) every connection counts, as before.
    local ttys active=0 t
    ttys="$(ps -eo args= 2>/dev/null | sed -n 's/^sshd[-a-z]*: [^ ]*@\(pts\/[0-9]*\).*/\1/p' | sort -u)"
    if [ -z "$ttys" ] || [ "$n" -gt "$(echo "$ttys" | grep -c .)" ]; then echo "$n ssh connection(s)"; return; fi
    for t in $ttys; do
      [ -n "$(find "/dev/$t" -amin "-$SSH_IDLE_MINUTES" 2>/dev/null)" ] && active=$((active + 1))
    done
    if [ "$active" -gt 0 ]; then echo "$n ssh connection(s), $active typed into in the last $SSH_IDLE_MINUTES min"; return; fi
  fi
  procs="$(pgrep -f "$BUSY_PROCS" 2>/dev/null | while read -r p; do
    case "$ANCESTORS" in *" $p "*) continue ;; esac
    case "$(ps -o comm= -p "$p" 2>/dev/null)" in tmux*|claude|'') continue ;; esac   # claude: its start prompt names the mission file (kto_refusal...), and a claude process alone is not busy
    ps -o comm= -p "$p"
  done | sort -u | tr '\n' ' ')"
  if [ -n "$procs" ]; then echo "running: $procs"; return; fi
  n="$(curl -s -m 5 "localhost:$PORT/metrics" 2>/dev/null \
    | awk '/^vllm:num_requests_(running|waiting)\{/ { s += $NF } END { printf "%d", s }')"
  if [ "${n:-0}" -gt 0 ]; then echo "$n vLLM request(s) in flight"; return; fi
  n="$(for _ in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 1; done | sort -n | tail -1 | tr -d ' ')"
  if [ "${n:-0}" -ge "$GPU_BUSY_PERCENT" ] 2>/dev/null; then echo "GPU at ${n} %"; return; fi
  # An unattended Claude Code session that is working appends to its transcript all the time; one that
  # hit a usage limit, crashed or finished does not.
  if [ ! -e "$STOP_REQUEST_FILE" ] && [ -n "$(find "$CLAUDE_PROJECTS_DIR" -name '*.jsonl' -mmin "-$CLAUDE_ACTIVE_MINUTES" -print -quit 2>/dev/null)" ]; then
    echo "claude active in the last $CLAUDE_ACTIVE_MINUTES min"; return
  fi
}

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

say "auto-stop armed: instance $CONTAINER_ID stops after $IDLE_MINUTES idle minutes in a row. Cancel: touch $CANCEL_FILE"
if [ "$NOW" = 1 ]; then touch "$STOP_REQUEST_FILE"; say "--now: stopping as soon as no eval, upload or vLLM request is running (SSH and Claude activity ignored)"
elif [ -e "$STOP_REQUEST_FILE" ]; then rm -f "$STOP_REQUEST_FILE"; say "removed a stale stop request"; fi
idle=0
while :; do
  if [ -e "$CANCEL_FILE" ]; then idle=0; say "paused by $CANCEL_FILE (delete it to resume)"; sleep 60; continue; fi
  reason="$(busy_reason)"
  if [ -n "$reason" ]; then
    idle=0
    say "busy: $reason$([ -e "$STOP_REQUEST_FILE" ] && echo " (stop requested; waiting for that)")"
  else
    idle=$((idle + 1))
    if [ -e "$STOP_REQUEST_FILE" ]; then say "stop requested ($STOP_REQUEST_FILE) and nothing busy"; break; fi
    say "idle $idle/$IDLE_MINUTES min"
  fi
  [ "$idle" -lt "$IDLE_MINUTES" ] || break
  sleep 60
done

rm -f "$STOP_REQUEST_FILE"
until [ -e "$CANCEL_FILE" ]; do
  say "stopping instance $CONTAINER_ID (disk is kept; restart it from the Vast console)"
  "$VASTAI" stop instance "$CONTAINER_ID" 2>&1 | tee -a "$LOG"
  # A successful stop kills this process; still being here means it has not taken effect.
  sleep 300
  say "still running 5 minutes after the stop request; retrying"
done
say "cancelled by $CANCEL_FILE; box left running"
