#!/usr/bin/env bash
# Install Claude Code on the GPU box and start it unattended (bypass-permissions) in tmux. Run as root,
# after scripts/setup_box.sh:
#   ssh -t gpubox bash /workspace/llm-reward-hacking-ablation/sweep/box_claude.sh [--watchdog-minutes 120] [--no-watchdog]
# Then: tmux attach -t claude, /login, paste sweep/mission.md, detach with Ctrl-b d.
# The watchdog is scripts/stop_box_when_idle.sh: it stops (never destroys) the box after N idle minutes
# with nobody connected, so a stalled Claude does not keep the GPU billing. Cancel: touch /tmp/rhablation-no-stop
# Safe to re-run: every step checks whether it is already done.
set -euo pipefail

WATCHDOG=1
WATCHDOG_MINUTES=120
while [ $# -gt 0 ]; do
  case "$1" in
    --watchdog-minutes) WATCHDOG_MINUTES="${2:-}"; shift 2 ;;
    --watchdog-minutes=*) WATCHDOG_MINUTES="${1#*=}"; shift ;;
    --no-watchdog) WATCHDOG=0; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER=/usr/local/bin/rhbench-run
RULES_SRC="$REPO_DIR/sweep/CLAUDE.unattended.md"
RULES_DST="$HOME/.claude/CLAUDE.md"
RULES_MARKER='<!-- rhablation: unattended box rules -->'
CANCEL_FILE="${RHABLATION_NO_STOP_FILE:-/tmp/rhablation-no-stop}"

step() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (vLLM, the upload token and the auto-stop all live under root)"
case "$WATCHDOG_MINUTES" in ''|*[!0-9]*) die "--watchdog-minutes must be a whole number" ;; esac
[ -x "$LAUNCHER" ] || die "$LAUNCHER missing: run 'bash scripts/setup_box.sh' first"
[ -f "$RULES_SRC" ] || die "missing $RULES_SRC"

step "Claude Code"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v claude >/dev/null; then
  curl -fsSL https://claude.ai/install.sh | bash
fi
CLAUDE_BIN="$(command -v claude)" || die "claude not on PATH after install (expected $HOME/.local/bin/claude)"
"$CLAUDE_BIN" --version

step "Unattended rules: $RULES_DST"
mkdir -p "$HOME/.claude"
chmod 700 "$HOME" "$HOME/.claude"
if [ -f "$RULES_DST" ] && ! grep -qF "$RULES_MARKER" "$RULES_DST"; then
  backup="$RULES_DST.bak-$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$RULES_DST" "$backup"
  echo "existing file was not ours; moved to $backup"
fi
cp "$RULES_SRC" "$RULES_DST"
grep -qF "$RULES_MARKER" "$RULES_DST" || die "$RULES_SRC lacks the marker line: $RULES_MARKER"

step "Isolation check (the benchmark user must not reach Claude's login)"
if "$LAUNCHER" test -r "$HOME/.claude"; then die "the benchmark user can read $HOME/.claude; re-run scripts/setup_box.sh"; fi
echo "ok   benchmark user cannot read $HOME/.claude"

step "tmux session 'claude'"
if tmux has-session -t claude 2>/dev/null; then
  echo "already running; left untouched"
else
  # A shell stays behind if claude exits, so the session can be resumed with: claude --continue
  tmux new-session -d -s claude -c "$REPO_DIR"
  tmux send-keys -t claude "IS_SANDBOX=1 '$CLAUDE_BIN' --dangerously-skip-permissions" Enter
  echo "started in $REPO_DIR"
fi

if [ "$WATCHDOG" = 1 ]; then
  step "Watchdog: stop the box after $WATCHDOG_MINUTES idle minutes"
  if tmux has-session -t watchdog 2>/dev/null; then
    echo "already armed; left untouched (log: /tmp/stop_box_when_idle.log)"
  elif [ -e "$CANCEL_FILE" ]; then
    echo "WARNING: $CANCEL_FILE exists, so auto-stop is cancelled. Remove it and re-run to arm the watchdog."
  elif bash "$REPO_DIR/scripts/stop_box_when_idle.sh" --check; then
    tmux new-session -d -s watchdog -c "$REPO_DIR" "bash scripts/stop_box_when_idle.sh --idle-minutes $WATCHDOG_MINUTES"
    echo "armed (log: /tmp/stop_box_when_idle.log)"
  else
    echo "WARNING: this box cannot stop itself; no watchdog. A stalled session will keep billing."
  fi
fi

step "Next"
cat <<EOF
  tmux attach -t claude        # accept the bypass-permissions notice, then /login (open the URL on your laptop)
  paste the contents of $REPO_DIR/sweep/mission.md (benchmarks and models filled in)
  detach with Ctrl-b d; the session keeps running after you disconnect
Progress:  $REPO_DIR/box_report/SUMMARY.md and LOG.md (also uploaded to the HF repo under sweep/box_report/)
Resume after a stop or a usage-limit pause:  cd $REPO_DIR && IS_SANDBOX=1 claude --dangerously-skip-permissions --continue
When finished: /logout inside claude (the login token otherwise stays on the stopped disk).
EOF
