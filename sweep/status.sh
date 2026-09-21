#!/usr/bin/env bash
# "How is the unattended run going?" in one command. Read-only. Run as root on the box:
#   bash sweep/status.sh [--watch 60]
# Shows the top of box_report/SUMMARY.md, live progress of the most recent run (scripts/stage1_progress.py,
# which works for every runner because they all write run_config.json + progress.jsonl), the auto-stop
# watchdog's last lines, and free disk. Note: your SSH session keeps the box from auto-stopping; without
# SSH, read sweep/box_report/SUMMARY.md in the HF dataset repo instead.
set -uo pipefail

WATCH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=(--watch "${2:-60}"); shift 2 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_REPO="${BENCH_REPO:-/home/${BENCH_USER:-rhbench}/$(basename "$REPO_DIR")}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/tmp/stop_box_when_idle.log}"
LAUNCHER=/usr/local/bin/rhbench-run

echo "== $REPO_DIR/box_report/SUMMARY.md"
if [ -f "$REPO_DIR/box_report/SUMMARY.md" ]; then head -40 "$REPO_DIR/box_report/SUMMARY.md"; else echo "(not written yet)"; fi

echo
echo "== watchdog ($WATCHDOG_LOG)"
if [ -f "$WATCHDOG_LOG" ]; then tail -3 "$WATCHDOG_LOG"; else echo "(no log: watchdog not armed)"; fi

echo
echo "== disk"
df -h / | tail -1

echo
# Newest run = the run directory whose progress.jsonl changed last.
newest="$(find "$BENCH_REPO/data" -name progress.jsonl -print0 2>/dev/null | xargs -0 -r ls -t 2>/dev/null | head -1)"
latest="${newest:+$(dirname "$newest")}"
if [ -z "$latest" ]; then
  echo "== no run with a progress.jsonl under $BENCH_REPO/data yet"
  exit 0
fi
echo "== latest run: $latest"
if [ -x "$LAUNCHER" ]; then
  exec "$LAUNCHER" uv run scripts/stage1_progress.py --run-dir "$latest" ${WATCH[@]+"${WATCH[@]}"}
else
  cd "$REPO_DIR" && exec uv run scripts/stage1_progress.py --run-dir "$latest" ${WATCH[@]+"${WATCH[@]}"}
fi
