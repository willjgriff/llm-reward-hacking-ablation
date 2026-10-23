#!/usr/bin/env bash
# Run as root on the GPU box: authorise the CheatBench VM's tunnel key for port forwarding to
# vLLM on 127.0.0.1:PORT and nothing else (no shell, no other ports, no agent/X11 forwarding).
#   [PORT=8000] bash cheatbench/allow_tunnel_key.sh 'ssh-ed25519 AAAA... rhtunnel@host'
# Vast may rewrite /root/.ssh/authorized_keys when the instance restarts; re-run after a restart.
set -euo pipefail

PORT="${PORT:-8000}"
KEY="${1:-}"
AUTH=/root/.ssh/authorized_keys

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run as root"
[[ "$KEY" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+(\ .*)?$ ]] || die "pass one ssh-ed25519 public key"

line="restrict,port-forwarding,permitopen=\"127.0.0.1:$PORT\",command=\"/bin/false\" $KEY"
blob=$(awk '{print $2}' <<<"$KEY")
install -d -m 700 /root/.ssh
touch "$AUTH"; chmod 600 "$AUTH"
if grep -qF "$blob" "$AUTH"; then
  grep -vF "$blob" "$AUTH" > "$AUTH.tmp" && cat "$AUTH.tmp" > "$AUTH" && rm -f "$AUTH.tmp"
fi
echo "$line" >> "$AUTH"
echo "authorised for 127.0.0.1:$PORT only:"
grep -F "$blob" "$AUTH"
