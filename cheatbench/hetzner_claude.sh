#!/usr/bin/env bash
# Start an unattended Claude Code session on the CheatBench VM (Hetzner). Run as root, from the root-owned
# repo copy, after cheatbench/setup_cpu_vm.sh has passed its --smoke run and upload.env was copied over:
#   bash cheatbench/hetzner_claude.sh --mission cheatbench/<mission>.md [--model claude-fable-5-1] [--effort xhigh]
# Installs Claude Code and the VM rules (cheatbench/CLAUDE.unattended.md → /root/.claude/CLAUDE.md) and starts
# Claude in tmux 'claude' (bypass-permissions, no MCP servers) with a prompt pointing at the mission.
# Then: tmux attach -t claude, /login if needed, accept the bypass notice, detach with Ctrl-b d.
# This VM never stops or deletes itself (Hetzner bills it until it is deleted). The GPU box stops itself
# when the agent's final step runs request-gpubox-stop, or failing that once no vLLM request has been in
# flight for its watchdog window (cheatbench/allow_tunnel_key.sh).
# Safe to re-run: every step checks whether it is already done.
set -euo pipefail

MODEL_ID=claude-fable-5-1
EFFORT=
MISSION=
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL_ID="${2:-}"; shift 2 ;;
    --effort) EFFORT="${2:-}"; shift 2 ;;
    --mission) MISSION="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER=cheatbench
RULES_SRC="$REPO_DIR/cheatbench/CLAUDE.unattended.md"
RULES_DST="$HOME/.claude/CLAUDE.md"
RULES_MARKER='<!-- rhablation: unattended cheatbench vm rules -->'

step() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root (the upload token and the Claude login live under /root)"
[ -n "$MISSION" ] || die "--mission is required, e.g. --mission cheatbench/mission_smoke.md"
case "$MISSION" in /*|*..*|*[!A-Za-z0-9._/-]*) die "--mission must be a plain path relative to the repo" ;; esac
[ -f "$REPO_DIR/$MISSION" ] || die "missing $REPO_DIR/$MISSION"
[ -f "$RULES_SRC" ] || die "missing $RULES_SRC"
case "$MODEL_ID" in ''|*[!A-Za-z0-9._-]*) die "--model must be a plain model id or alias" ;; esac
case "$EFFORT" in ''|low|medium|high|xhigh|max) ;; *) die "--effort must be one of low, medium, high, xhigh, max" ;; esac
id "$RUN_USER" >/dev/null 2>&1 || die "no $RUN_USER user: run cheatbench/setup_cpu_vm.sh first"
UPLOAD_ENV="${RHABLATION_UPLOAD_ENV:-$HOME/.config/rhablation/upload.env}"
[ -f "$UPLOAD_ENV" ] || echo "WARNING: $UPLOAD_ENV is missing: nothing will be uploaded until you copy it over."

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

step "No MCP servers or account connectors"
# Same reason as sweep/box_claude.sh: a subscription login brings the account's claude.ai connectors along.
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

step "Isolation check (the episode user must not reach credentials)"
for p in "$HOME/.claude" "$UPLOAD_ENV"; do
  if [ -e "$p" ] && sudo -u "$RUN_USER" test -r "$p" 2>/dev/null; then die "$RUN_USER can read $p"; fi
done
echo "ok   $RUN_USER cannot read $HOME/.claude or the upload token"

step "tmux session 'claude'"
if tmux has-session -t claude 2>/dev/null; then
  echo "already running; left untouched"
else
  tmux new-session -d -s claude -c "$REPO_DIR"
  effort_arg=""
  [ -n "$EFFORT" ] && effort_arg=" --effort $EFFORT"
  prompt=" 'Read $MISSION in this repository and carry it out from start to finish. Follow /root/.claude/CLAUDE.md.'"
  tmux send-keys -t claude "cd '$REPO_DIR' && IS_SANDBOX=1 '$CLAUDE_BIN' --dangerously-skip-permissions --strict-mcp-config --model $MODEL_ID$effort_arg$prompt" Enter
  echo "started in $REPO_DIR (no MCP servers, model $MODEL_ID$([ -n "$EFFORT" ] && echo ", effort $EFFORT"), mission $MISSION)"
fi

step "Next"
cat <<EOF
  tmux attach -t claude        # accept the bypass-permissions notice (and /login if asked); the mission then starts
  detach with Ctrl-b d; the session keeps running after you disconnect
Resume:  cd $REPO_DIR && IS_SANDBOX=1 claude --dangerously-skip-permissions --strict-mcp-config${effort_arg:-} --continue
When finished: /logout inside claude, pull anything you want, then delete the server in the Hetzner console.
EOF
