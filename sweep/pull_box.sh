#!/usr/bin/env bash
# Laptop side: fetch the unattended run's reports and code from the GPU box without letting anything
# from the box land where it could be executed.
#   bash sweep/pull_box.sh [--host gpubox] [--remote-dir /workspace/llm-reward-hacking-ablation]
#   bash sweep/pull_box.sh --from-hf <user/name>     # box already stopped: code saved by sweep/upload_code.sh
# Reports (markdown only) go to data/box_report/<UTC time>/ (git-ignored, never synced back to the box).
# The box's code tree goes to a quarantine directory OUTSIDE this repo, with symlinks skipped and
# execute bits stripped, and is compared with this working tree. Read the diff and apply changes by hand;
# never rsync the box's tree over this one. Transcripts come from the HF dataset repo, not from here.
# --from-hf fetches sweep/code/ from the private HF dataset repo instead (needs HF_TOKEN, e.g. from
# ~/.config/rhablation/upload.env); it is the same untrusted box output and gets the same quarantine.
set -euo pipefail

HOST=gpubox
REMOTE_DIR=/workspace/llm-reward-hacking-ablation
FROM_HF=""
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --remote-dir) REMOTE_DIR="${2:-}"; shift 2 ;;
    --from-hf) FROM_HF="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%MZ)"
REPORT_DIR="$REPO_DIR/data/box_report/$STAMP"
QUARANTINE="${RHABLATION_QUARANTINE:-$HOME/rhablation-box-quarantine}/$STAMP"
# Never fetched, never compared: environments, data, and anything an editor or git would act on.
SKIP=(.git .venv data logs .inspect .claude .vscode __pycache__ box_report)

case "$QUARANTINE/" in "$REPO_DIR"/*) echo "ERROR: the quarantine must be outside the repo: $QUARANTINE" >&2; exit 1 ;; esac
mkdir -p "$QUARANTINE"

if [ -n "$FROM_HF" ]; then
  UPLOAD_ENV="${RHABLATION_UPLOAD_ENV:-$HOME/.config/rhablation/upload.env}"
  if [ -z "${HF_TOKEN:-}" ] && [ -f "$UPLOAD_ENV" ]; then
    HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$UPLOAD_ENV" | tail -1)"
  fi
  [ -n "${HF_TOKEN:-}" ] || { echo "ERROR: HF_TOKEN not set (environment or $UPLOAD_ENV)" >&2; exit 1; }
  export HF_TOKEN
  DOWNLOAD="$QUARANTINE.hf"
  echo "== code from $FROM_HF : sweep/code -> $QUARANTINE (quarantine: read it, do not run it)"
  uvx --from huggingface_hub hf download "$FROM_HF" --repo-type dataset --include "sweep/code/*" --local-dir "$DOWNLOAD" >/dev/null
  [ -d "$DOWNLOAD/sweep/code" ] || { echo "ERROR: no sweep/code/ in $FROM_HF yet" >&2; exit 1; }
  rsync -rt --no-links --chmod=Fa-x "$DOWNLOAD/sweep/code/" "$QUARANTINE/"
  rm -rf "$DOWNLOAD"
else
  mkdir -p "$REPORT_DIR"
  echo "== reports -> $REPORT_DIR"
  rsync -rt --no-links --chmod=Fa-x --prune-empty-dirs --include '*/' --include '*.md' --exclude '*' \
    "$HOST:$REMOTE_DIR/box_report/" "$REPORT_DIR/" || echo "no box_report/ on the box yet"
  ls -l "$REPORT_DIR"

  echo
  echo "== code -> $QUARANTINE (quarantine: read it, do not run it)"
  excludes=()
  for name in "${SKIP[@]}"; do excludes+=(--exclude "$name"); done
  rsync -rt --no-links --chmod=Fa-x --max-size=2m "${excludes[@]}" "$HOST:$REMOTE_DIR/" "$QUARANTINE/"
fi

echo
echo "== files that differ from this working tree"
diff_excludes=()
for name in "${SKIP[@]}"; do diff_excludes+=(-x "$name"); done
diff -rq "${diff_excludes[@]}" "$REPO_DIR" "$QUARANTINE" || true
cat <<EOF

Read one change:   git diff --no-index -- <file> "$QUARANTINE/<file>"
Read everything:   diff -ru $(printf -- '-x %s ' "${SKIP[@]}")"$REPO_DIR" "$QUARANTINE" | less
Transcripts:       uvx --from huggingface_hub hf download <repo> --repo-type dataset --include "sweep/*" --exclude "sweep/code/*" --local-dir data/
EOF
