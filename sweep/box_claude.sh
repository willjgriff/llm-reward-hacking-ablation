#!/usr/bin/env bash
# The one on-box command for the unattended sweep. Run as root after the repo tree (and upload.env) were
# copied over from the laptop:
#   ssh gpubox bash /workspace/llm-reward-hacking-ablation/sweep/box_claude.sh [--model claude-fable-5-1] [--mission sweep/mission.md] [--no-mission] [--watchdog-minutes 120] [--no-watchdog]
# Sets the box up if needed (scripts/setup_box.sh --no-serve), installs Claude Code and the unattended
# rules, and starts Claude in tmux (bypass-permissions) with a prompt that points it at the mission file
# (--mission, relative to the repo; default sweep/mission.md).
# Then: tmux attach -t claude, /login if needed, accept the bypass notice, detach with Ctrl-b d.
# The watchdog is scripts/stop_box_when_idle.sh: it stops (never destroys) the box after N idle minutes
# with nobody connected, so a stalled Claude does not keep the GPU billing. Cancel: touch /tmp/rhablation-no-stop
# Safe to re-run: every step checks whether it is already done.
set -euo pipefail

WATCHDOG=1
WATCHDOG_MINUTES=120
MODEL_ID=claude-fable-5-1
START_MISSION=1
MISSION=sweep/mission.md
while [ $# -gt 0 ]; do
  case "$1" in
    --watchdog-minutes) WATCHDOG_MINUTES="${2:-}"; shift 2 ;;
    --watchdog-minutes=*) WATCHDOG_MINUTES="${1#*=}"; shift ;;
    --no-watchdog) WATCHDOG=0; shift ;;
    --model) MODEL_ID="${2:-}"; shift 2 ;;
    --mission) MISSION="${2:-}"; shift 2 ;;
    --no-mission) START_MISSION=0; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
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
[ -f "$RULES_SRC" ] || die "missing $RULES_SRC"
case "$MISSION" in ''|/*|*..*|*[!A-Za-z0-9._/-]*) die "--mission must be a plain path relative to the repo, e.g. sweep/mission_stage4.md" ;; esac
[ -f "$REPO_DIR/$MISSION" ] || die "missing $REPO_DIR/$MISSION"
case "$MODEL_ID" in ''|*[!A-Za-z0-9._-]*) die "--model must be a plain model id or alias" ;; esac

if [ ! -x "$LAUNCHER" ]; then
  step "Box set-up (scripts/setup_box.sh --no-serve; the agent starts vLLM itself with the mission's model)"
  bash "$REPO_DIR/scripts/setup_box.sh" --no-serve
fi
[ -x "$LAUNCHER" ] || die "$LAUNCHER still missing after set-up"
UPLOAD_ENV="${RHABLATION_UPLOAD_ENV:-$HOME/.config/rhablation/upload.env}"
[ -f "$UPLOAD_ENV" ] || echo "WARNING: $UPLOAD_ENV is missing: nothing will be uploaded to Hugging Face until you copy it over (setup.md 'Off-box storage')."

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

step "No MCP servers or account connectors for the box agent"
# A subscription login brings the account's claude.ai connectors (Google Drive, Docs, ...) with it. In
# bypass mode the agent, or anything that prompt-injects it, could use them without asking. The session is
# started with --strict-mcp-config (no MCP servers at all); the deny rules cover a session started by hand.
SETTINGS="$HOME/.claude/settings.json"
python3 - "$SETTINGS" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
settings = json.loads(path.read_text()) if path.exists() and path.read_text().strip() else {}
deny = settings.setdefault("permissions", {}).setdefault("deny", [])
for rule in ("mcp__*", "mcp__claude_ai_Google_Drive", "mcp__claude_ai_Claude_Docs"):
    if rule not in deny:
        deny.append(rule)
path.write_text(json.dumps(settings, indent=2) + "\n")
print("deny rules in", path, ":", deny)
PY
chmod 600 "$SETTINGS"
echo "connectors this login would load without the flag:"
(cd "$REPO_DIR" && timeout 60 "$CLAUDE_BIN" mcp list 2>&1 | grep -v "^Checking\|^$" | sed 's/^/  /') || true

step "Isolation check (the benchmark user must not reach Claude's login)"
if "$LAUNCHER" test -r "$HOME/.claude"; then die "the benchmark user can read $HOME/.claude; re-run scripts/setup_box.sh"; fi
echo "ok   benchmark user cannot read $HOME/.claude"

step "tmux session 'claude'"
if tmux has-session -t claude 2>/dev/null; then
  echo "already running; left untouched"
else
  # A shell stays behind if claude exits, so the session can be resumed with: claude --continue
  tmux new-session -d -s claude -c "$REPO_DIR"
  prompt=""
  if [ "$START_MISSION" = 1 ]; then
    prompt=" 'Read $MISSION in this repository and carry it out from start to finish. Follow /root/.claude/CLAUDE.md.'"
  fi
  # cd explicitly: the Vast image's shell start-up changes to \$WORKSPACE after tmux has set the directory,
  # and Claude's project (and what `--continue` finds later) is the directory it was started in.
  tmux send-keys -t claude "cd '$REPO_DIR' && IS_SANDBOX=1 '$CLAUDE_BIN' --dangerously-skip-permissions --strict-mcp-config --model $MODEL_ID$prompt" Enter
  echo "started in $REPO_DIR (no MCP servers, model $MODEL_ID$([ "$START_MISSION" = 1 ] && echo ", mission $MISSION"))"
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
  tmux attach -t claude        # accept the bypass-permissions notice (and /login if asked); the mission then starts
  detach with Ctrl-b d; the session keeps running after you disconnect
Progress:  bash $REPO_DIR/sweep/status.sh [--watch 60]   (summary, live progress of the newest run, watchdog, disk)
           or: rhbench-run uv run scripts/stage1_progress.py --run-dir <run_dir>
           An open SSH session keeps the box from auto-stopping; without SSH read sweep/box_report/SUMMARY.md in the HF repo.
Resume after a stop or a usage-limit pause:  cd $REPO_DIR && IS_SANDBOX=1 claude --dangerously-skip-permissions --strict-mcp-config --continue
When finished: /logout inside claude (the login token otherwise stays on the stopped disk).
EOF
