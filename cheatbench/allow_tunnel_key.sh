#!/usr/bin/env bash
# Run as root on the GPU box: authorise the CheatBench VM's tunnel key for port forwarding to
# vLLM on 127.0.0.1:PORT and nothing else (no shell, no other ports, no agent/X11 forwarding),
# from the VM's address only, and let the box stop itself once the VM is done with it.
#   FROM=<vm public ipv4> [PORT=8000] [WATCHDOG_MINUTES=120] [RESTART_WATCHDOG=1] \
#     bash cheatbench/allow_tunnel_key.sh 'ssh-ed25519 AAAA... rhtunnel@host'
# The VM's address goes to /root/.config/rhablation/tunnel_peers, which scripts/stop_box_when_idle.sh
# reads: the always-open tunnel then does not count as somebody connected, and the box stops after
# WATCHDOG_MINUTES with no vLLM request in flight (and nothing else busy). Arms that watchdog if none
# is running; a watchdog started before tunnel_peers existed counts the tunnel as busy, so restart it
# with RESTART_WATCHDOG=1. Vast may rewrite /root/.ssh/authorized_keys on restart: re-run after one.
set -euo pipefail

PORT="${PORT:-8000}"
FROM="${FROM:-}"
WATCHDOG_MINUTES="${WATCHDOG_MINUTES:-120}"
RESTART_WATCHDOG="${RESTART_WATCHDOG:-0}"
KEY="${1:-}"
AUTH=/root/.ssh/authorized_keys
PEERS="${RHABLATION_TUNNEL_PEERS:-/root/.config/rhablation/tunnel_peers}"
CANCEL_FILE="${RHABLATION_NO_STOP_FILE:-/tmp/rhablation-no-stop}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root"
[[ "$KEY" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+(\ .*)?$ ]] || die "pass one ssh-ed25519 public key"
[[ "$FROM" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || die "set FROM=<the VM's public IPv4> (setup_cpu_vm.sh prints the full command)"
case "$WATCHDOG_MINUTES" in ''|*[!0-9]*) die "WATCHDOG_MINUTES must be a whole number" ;; esac

line="restrict,port-forwarding,permitopen=\"127.0.0.1:$PORT\",from=\"$FROM\",command=\"/bin/false\" $KEY"
blob=$(awk '{print $2}' <<<"$KEY")
install -d -m 700 /root/.ssh
touch "$AUTH"; chmod 600 "$AUTH"
if grep -qF "$blob" "$AUTH"; then
  grep -vF "$blob" "$AUTH" > "$AUTH.tmp" && cat "$AUTH.tmp" > "$AUTH" && rm -f "$AUTH.tmp"
fi
echo "$line" >> "$AUTH"
echo "authorised for 127.0.0.1:$PORT only, from $FROM:"
grep -F "$blob" "$AUTH"

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
