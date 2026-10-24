#!/usr/bin/env bash
# Run as root on the GPU box: authorise the CheatBench VM's tunnel key for port forwarding to
# vLLM on 127.0.0.1:PORT and nothing else (no shell, no other ports, no agent/X11 forwarding),
# from the VM's address only, and let the box stop itself once the VM is done with it.
#   FROM=<vm public ipv4> DONE_KEY='ssh-ed25519 AAAA... gpubox-done@host' [ROW_KEY='ssh-ed25519 ...'] [PORT=8000] [WATCHDOG_MINUTES=90] \
#     [RESTART_WATCHDOG=1] bash cheatbench/allow_tunnel_key.sh 'ssh-ed25519 AAAA... rhtunnel@host'
# The VM's address goes to /root/.config/rhablation/tunnel_peers, which scripts/stop_box_when_idle.sh
# reads: the always-open tunnel then does not count as somebody connected, and the box stops after
# WATCHDOG_MINUTES with no vLLM request in flight (and nothing else busy). Arms that watchdog if none
# is running; a watchdog started before tunnel_peers existed counts the tunnel as busy, so restart it
# with RESTART_WATCHDOG=1. DONE_KEY (optional) is a second VM key whose only possible effect is a forced
# command that creates the watchdog's stop-request file: the VM's request-gpubox-stop then stops the box
# as soon as nothing on it is busy, instead of after the idle window. ROW_KEY (optional) is a third VM key whose
# only possible effect is cheatbench/serve_row.sh with one of `base a3 rand status` (the VM's request-gpubox-row):
# it switches which eval row vLLM serves (cheatbench/mission_a3.md).
# Vast may rewrite /root/.ssh/authorized_keys on restart: re-run after one.
set -euo pipefail

PORT="${PORT:-8000}"
FROM="${FROM:-}"
WATCHDOG_MINUTES="${WATCHDOG_MINUTES:-90}"
RESTART_WATCHDOG="${RESTART_WATCHDOG:-0}"
DONE_KEY="${DONE_KEY:-}"
ROW_KEY="${ROW_KEY:-}"
KEY="${1:-}"
AUTH=/root/.ssh/authorized_keys
PEERS="${RHABLATION_TUNNEL_PEERS:-/root/.config/rhablation/tunnel_peers}"
CANCEL_FILE="${RHABLATION_NO_STOP_FILE:-/tmp/rhablation-no-stop}"
STOP_REQUEST_FILE="${RHABLATION_STOP_REQUEST_FILE:-/tmp/rhablation-stop-requested}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root"
[[ "$KEY" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+(\ .*)?$ ]] || die "pass one ssh-ed25519 public key"
[[ "$FROM" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || die "set FROM=<the VM's public IPv4> (setup_cpu_vm.sh prints the full command)"
[ -z "$DONE_KEY" ] || [[ "$DONE_KEY" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+(\ .*)?$ ]] || die "DONE_KEY must be one ssh-ed25519 public key"
[ -z "$ROW_KEY" ] || [[ "$ROW_KEY" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+(\ .*)?$ ]] || die "ROW_KEY must be one ssh-ed25519 public key"
[ -z "$ROW_KEY" ] || [ -f "$REPO_DIR/cheatbench/serve_row.sh" ] || die "missing $REPO_DIR/cheatbench/serve_row.sh"
case "$WATCHDOG_MINUTES" in ''|*[!0-9]*) die "WATCHDOG_MINUTES must be a whole number" ;; esac

install -d -m 700 /root/.ssh
touch "$AUTH"; chmod 600 "$AUTH"
# Replace any earlier line for the same key.
authorise() {
  local blob
  blob=$(awk '{print $2}' <<<"$2")
  if grep -qF "$blob" "$AUTH"; then
    grep -vF "$blob" "$AUTH" > "$AUTH.tmp" && cat "$AUTH.tmp" > "$AUTH" && rm -f "$AUTH.tmp"
  fi
  # Vast writes the file without a final newline; appending to it glued our line onto the user's key.
  if [ -s "$AUTH" ] && [ -n "$(tail -c1 "$AUTH")" ]; then echo >> "$AUTH"; fi
  echo "$1 $2" >> "$AUTH"
  grep -F "$blob" "$AUTH"
}
echo "tunnel key, 127.0.0.1:$PORT only, from $FROM:"
authorise "restrict,port-forwarding,permitopen=\"127.0.0.1:$PORT\",from=\"$FROM\",command=\"/bin/false\"" "$KEY"
if [ -n "$DONE_KEY" ]; then
  echo "done key, forced command only, from $FROM:"
  authorise "restrict,from=\"$FROM\",command=\"touch $STOP_REQUEST_FILE && echo stop requested\"" "$DONE_KEY"
fi
if [ -n "$ROW_KEY" ]; then
  [ -f /root/.config/rhablation/cheatbench_serve.env ] || echo "WARNING: run cheatbench/gpubox_setup.sh before the VM switches rows"
  echo "row key, forced command only (serve_row.sh, row from the client's command, checked there), from $FROM:"
  authorise "restrict,from=\"$FROM\",command=\"bash $REPO_DIR/cheatbench/serve_row.sh --forced\"" "$ROW_KEY"
fi

install -d -m 700 "$(dirname "$PEERS")"
touch "$PEERS"; chmod 600 "$PEERS"
grep -qxF "$FROM" "$PEERS" || echo "$FROM" >> "$PEERS"
echo "watchdog ignores SSH from: $(tr '\n' ' ' < "$PEERS")"

if [ "$RESTART_WATCHDOG" = 1 ] && tmux has-session -t watchdog 2>/dev/null; then
  tmux kill-session -t watchdog
  echo "old watchdog stopped"
fi
if tmux has-session -t watchdog 2>/dev/null; then
  echo "a watchdog is already running and was left alone. If it was started before tunnel_peers existed,"
  echo "it counts the tunnel as busy and the box never stops: re-run with RESTART_WATCHDOG=1."
elif [ -e "$CANCEL_FILE" ]; then
  echo "WARNING: $CANCEL_FILE exists, so auto-stop is cancelled; the box keeps billing until stopped by hand."
elif bash "$REPO_DIR/scripts/stop_box_when_idle.sh" --check; then
  tmux new-session -d -s watchdog -c "$REPO_DIR" "bash scripts/stop_box_when_idle.sh --idle-minutes $WATCHDOG_MINUTES"
  echo "watchdog armed: the box stops after $WATCHDOG_MINUTES idle minutes (log: /tmp/stop_box_when_idle.log)"
else
  echo "WARNING: this box cannot stop itself; no watchdog. Stop it by hand when the VM is done."
fi
