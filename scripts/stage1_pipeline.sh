#!/usr/bin/env bash
# Unattended stage 1 on the GPU box: benchmark -> export -> validate -> upload to the HF dataset repo.
# Run as root from the root-owned copy of the repo (not the one under /home/rhbench), inside tmux:
#   bash scripts/stage1_pipeline.sh --run-id lcb-4b-part1 --display plain [stage1_run_benchmark.py args] [--no-upload] [--no-stop] [--idle-minutes 30]
# A failing step does not stop the later ones, so a crashed or partly invalid run is still uploaded.
# After a successful upload it hands over to stop_box_when_idle.sh, which stops the box (disk kept)
# once nobody is connected and nothing is running for --idle-minutes in a row.
set -uo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_USER="${BENCH_USER:-rhbench}"
BENCH_HOME="/home/$BENCH_USER"
BENCH_REPO="$BENCH_HOME/$(basename "$REPO_DIR")"
UPLOAD_ENV="${RHABLATION_UPLOAD_ENV:-$HOME/.config/rhablation/upload.env}"

[ "$(id -u)" = 0 ] || die "run as root (the upload token is root-only; the benchmark itself still runs as $BENCH_USER)"
command -v rhbench-run >/dev/null || die "rhbench-run not found; run scripts/setup_box.sh first"
# Root must not execute scripts that the benchmark user can rewrite.
case "$REPO_DIR" in "$BENCH_HOME"/*) die "run this from the root-owned repo copy, not from $BENCH_HOME" ;; esac

UPLOAD=1
STOP=1
IDLE_MINUTES=30
RUN_ID=""
OUTPUT_DIR="data/stage1"
BENCH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-upload) UPLOAD=0; shift ;;
    --no-stop) STOP=0; shift ;;
    --idle-minutes) IDLE_MINUTES="${2:-}"; shift 2 ;;
    --idle-minutes=*) IDLE_MINUTES="${1#*=}"; shift ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) BENCH_ARGS+=("$1"); shift ;;
  esac
done
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
case "$OUTPUT_DIR" in /*) RUN_DIR="$OUTPUT_DIR/$RUN_ID" ;; *) RUN_DIR="$BENCH_REPO/$OUTPUT_DIR/$RUN_ID" ;; esac
[ ! -e "$RUN_DIR" ] || die "$RUN_DIR already exists; pick another --run-id"

if [ "$UPLOAD" = 1 ]; then
  # Fail now rather than after hours of compute.
  [ -f "$UPLOAD_ENV" ] || die "no upload config at $UPLOAD_ENV (see setup.md 'Off-box storage'), or pass --no-upload"
  grep -q '^HF_TOKEN=' "$UPLOAD_ENV" && grep -q '^HF_UPLOAD_REPO=' "$UPLOAD_ENV" \
    || die "$UPLOAD_ENV must define HF_TOKEN and HF_UPLOAD_REPO"
fi
if [ "$STOP" = 1 ]; then
  bash "$REPO_DIR/scripts/stop_box_when_idle.sh" --check --idle-minutes "$IDLE_MINUTES" \
    || die "the box cannot stop itself; fix that or pass --no-stop"
fi

LOG="/tmp/stage1_$RUN_ID.log"
: > "$LOG"
RESULTS=()
FAILED=0
UPLOADED=0
run_step() {
  local name="$1"; shift
  printf '\n== %s (%s)\n' "$name" "$(date -u +%H:%M:%SZ)" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; then
    RESULTS+=("ok    $name")
  else
    RESULTS+=("FAIL  $name")
    FAILED=1
  fi
}

run_step benchmark rhbench-run uv run scripts/stage1_run_benchmark.py --run-id "$RUN_ID" --output-dir "$OUTPUT_DIR" ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
if [ -d "$RUN_DIR" ]; then
  run_step export rhbench-run uv run scripts/stage1_export_trajectories.py --run-dir "$RUN_DIR"
  run_step validate rhbench-run uv run scripts/stage1_validate.py --run-dir "$RUN_DIR"
  if [ "$UPLOAD" = 1 ]; then
    cp --remove-destination "$LOG" "$RUN_DIR/pipeline.log"  # never write through a planted symlink
    run_step upload bash "$REPO_DIR/scripts/upload_run.sh" --run-dir "$RUN_DIR"
    case "${RESULTS[${#RESULTS[@]}-1]}" in ok*) UPLOADED=1 ;; esac
  fi
else
  echo "benchmark did not create $RUN_DIR; nothing to export or upload" | tee -a "$LOG"
fi

{
  printf '\n== Summary for run %s (log: %s)\n' "$RUN_ID" "$LOG"
  printf '%s\n' "${RESULTS[@]}"
} | tee -a "$LOG"

if [ "$STOP" = 1 ]; then
  if [ "$UPLOADED" = 1 ]; then
    exec bash "$REPO_DIR/scripts/stop_box_when_idle.sh" --idle-minutes "$IDLE_MINUTES"
  fi
  # A stopped box cannot always be restarted at once (the GPU may get rented out meanwhile).
  echo "NOT stopping the box: the run was not uploaded, so its only copy is on this disk." | tee -a "$LOG"
fi
exit "$FAILED"
