#!/usr/bin/env bash
# Upload one finished run directory to a private Hugging Face dataset repo, under stage1/<run_id>/.
#   bash scripts/upload_run.sh --run-dir data/stage1/<run_id> [--repo <user/name>]
# Needs HF_TOKEN and HF_UPLOAD_REPO, from the environment or from ~/.config/rhablation/upload.env
# (override the path with RHABLATION_UPLOAD_ENV). On the GPU box run it as root, never through
# rhbench-run: the token must stay out of reach of the user that runs model-written code.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

RUN_DIR=""
REPO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-dir) RUN_DIR="${2:-}"; shift 2 ;;
    --repo) REPO="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$RUN_DIR" ] || die "--run-dir is required"
[ -d "$RUN_DIR" ] || die "no such run directory: $RUN_DIR"

UPLOAD_ENV="${RHABLATION_UPLOAD_ENV:-$HOME/.config/rhablation/upload.env}"
if [ -f "$UPLOAD_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$UPLOAD_ENV"
  set +a
fi
REPO="${REPO:-${HF_UPLOAD_REPO:-}}"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN not set (environment or $UPLOAD_ENV); see setup.md 'Off-box storage'"
[ -n "$REPO" ] || die "no target repo: pass --repo or set HF_UPLOAD_REPO (environment or $UPLOAD_ENV)"
export HF_TOKEN

# The run directory is writable by the benchmark user; never follow a link out of it.
links=$(find "$RUN_DIR" -type l | head -5)
[ -z "$links" ] || die "symlinks inside $RUN_DIR, refusing to upload: $links"

RUN_ID="$(basename "$(cd "$RUN_DIR" && pwd)")"
DEST="stage1/$RUN_ID"
echo "uploading $RUN_DIR ($(du -sh "$RUN_DIR" | cut -f1)) -> $REPO (dataset) : $DEST"

for attempt in 1 2 3; do
  if uvx --from huggingface_hub hf upload "$REPO" "$RUN_DIR" "$DEST" --repo-type dataset \
       --commit-message "stage1 run $RUN_ID"; then
    echo "uploaded. Download with:"
    echo "  uvx --from huggingface_hub hf download $REPO --repo-type dataset --include \"$DEST/*\" --local-dir data/"
    exit 0
  fi
  echo "upload attempt $attempt failed" >&2
  [ "$attempt" = 3 ] || sleep 60
done
die "upload failed after 3 attempts; the run is still in $RUN_DIR"
