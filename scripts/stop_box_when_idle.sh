#!/usr/bin/env bash
# Stop this Vast.ai instance once it has been idle, with nobody connected, for N minutes in a row.
#   bash scripts/stop_box_when_idle.sh [--idle-minutes 30] [--check]
# Run as root on the box (stage1_pipeline.sh hands over to it after a successful upload).
# "Stop" halts GPU charges and keeps the disk; it is not "destroy". Busy means any of: an inbound
# SSH connection (terminal, VS Code, rsync/scp), a benchmark/export/upload/copy process, vLLM
# requests in flight, or an unattended Claude Code session that wrote to its transcript recently. Cancel with: touch /tmp/rhablation-no-stop
# --check only verifies that the instance can stop itself (used as a preflight).
set -uo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

IDLE_MINUTES=30
CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --idle-minutes) IDLE_MINUTES="${2:-}"; shift 2 ;;
    --idle-minutes=*) IDLE_MINUTES="${1#*=}"; shift ;;
    --check) CHECK=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
case "$IDLE_MINUTES" in ''|*[!0-9]*) die "--idle-minutes must be a whole number" ;; esac

PORT="${PORT:-8000}"
CANCEL_FILE="${RHABLATION_NO_STOP_FILE:-/tmp/rhablation-no-stop}"
LOG=/tmp/stop_box_when_idle.log
CLAUDE_PROJECTS_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"
CLAUDE_ACTIVE_MINUTES="${CLAUDE_ACTIVE_MINUTES:-15}"
# Downloads and installs count as busy so a long gap between runs (new model, new env) is not idle.
# A `claude` process alone is deliberately not busy: a stalled or finished one must not hold the box up.
# What counts is its session transcript having been written to recently (CLAUDE_ACTIVE_MINUTES).
BUSY_PROCS='stage1_run_benchmark|stage1_pipeline|stage1_export|stage1_validate|stage4_cache_activations|stage4_directions|upload_run|hf upload|hf download|uv (sync|pip)|docker (pull|build)|(^|[ /])(rsync|scp|sftp-server)( |$)'

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
  local n procs
  n="$(ss -Htn state established "( sport = :$SSH_PORT )" 2>/dev/null | wc -l)"
  if [ "$n" -gt 0 ]; then echo "$n ssh connection(s)"; return; fi
  procs="$(pgrep -f "$BUSY_PROCS" 2>/dev/null | while read -r p; do
    case "$ANCESTORS" in *" $p "*) continue ;; esac
    case "$(ps -o comm= -p "$p" 2>/dev/null)" in tmux*|'') continue ;; esac
    ps -o comm= -p "$p"
  done | sort -u | tr '\n' ' ')"
  if [ -n "$procs" ]; then echo "running: $procs"; return; fi
  n="$(curl -s -m 5 "localhost:$PORT/metrics" 2>/dev/null \
    | awk '/^vllm:num_requests_(running|waiting)\{/ { s += $NF } END { printf "%d", s }')"
  if [ "${n:-0}" -gt 0 ]; then echo "$n vLLM request(s) in flight"; return; fi
  # An unattended Claude Code session that is working appends to its transcript all the time; one that
  # hit a usage limit, crashed or finished does not.
  if [ -n "$(find "$CLAUDE_PROJECTS_DIR" -name '*.jsonl' -mmin "-$CLAUDE_ACTIVE_MINUTES" -print -quit 2>/dev/null)" ]; then
    echo "claude active in the last $CLAUDE_ACTIVE_MINUTES min"; return
  fi
}

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

say "auto-stop armed: instance $CONTAINER_ID stops after $IDLE_MINUTES idle minutes in a row. Cancel: touch $CANCEL_FILE"
idle=0
while :; do
  if [ -e "$CANCEL_FILE" ]; then say "cancelled by $CANCEL_FILE; box left running"; exit 0; fi
  reason="$(busy_reason)"
  if [ -n "$reason" ]; then
    idle=0
    say "busy: $reason"
  else
    idle=$((idle + 1))
    say "idle $idle/$IDLE_MINUTES min"
  fi
  [ "$idle" -lt "$IDLE_MINUTES" ] || break
  sleep 60
done

until [ -e "$CANCEL_FILE" ]; do
  say "stopping instance $CONTAINER_ID (disk is kept; restart it from the Vast console)"
  "$VASTAI" stop instance "$CONTAINER_ID" 2>&1 | tee -a "$LOG"
  # A successful stop kills this process; still being here means it has not taken effect.
  sleep 300
  say "still running 5 minutes after the stop request; retrying"
done
say "cancelled by $CANCEL_FILE; box left running"
