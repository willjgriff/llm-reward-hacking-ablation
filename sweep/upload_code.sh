#!/usr/bin/env bash
# Save the box's code tree to the private Hugging Face dataset repo, under sweep/code/. Run as root on the box:
#   bash sweep/upload_code.sh [--repo <user/name>] [--message "evilgenie runner passes smoke"] [--dest <name>/code]
# --dest (default sweep/code) gives a box its own mirror, so two boxes never overwrite each other (e.g. capability_kto/code).
# This is the only way code leaves the box. It never touches GitHub. sweep/code/ mirrors the tree (files
# deleted on the box are deleted there); every upload is a commit, so earlier versions stay in the history.
# Needs HF_TOKEN and HF_UPLOAD_REPO like scripts/upload_run.sh (~/.config/rhablation/upload.env).
# Refuses to upload if any staged file contains a credential. Get it back on the laptop with:
#   bash sweep/pull_box.sh --from-hf <user/name>
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

REPO=""
MESSAGE=""
DEST="sweep/code"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="${2:-}"; shift 2 ;;
    --message) MESSAGE="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Deletions are limited to this folder of the dataset repo; it must never be empty or the repo root.
case "$DEST" in [A-Za-z0-9_-]*/code) ;; *) die "--dest must be <name>/code, not $DEST" ;; esac
case "$DEST" in *..*|*//*) die "--dest must be a plain path" ;; esac

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

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Code only: no environments, data, reports, editor/agent settings, links or large files.
rsync -rt --no-links --max-size=2m \
  --exclude .venv --exclude data --exclude logs --exclude .inspect --exclude .git --exclude .claude \
  --exclude .vscode --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store --exclude box_report \
  --exclude .env --exclude '*.env' \
  "$REPO_DIR/" "$STAGE/"

# Secret guard. Report file names only, never the matching text.
env_value() { grep -E "^(export )?$1=" /etc/environment 2>/dev/null | tail -1 | sed -E "s/^(export )?$1=//; s/^[\"']//; s/[\"']\$//"; }
bad=""
GITHUB_ENV_FILE="${RHABLATION_GITHUB_ENV:-$HOME/.config/rhablation/github.env}"
GH_TOKEN_VALUE="$(sed -n 's/^GH_TOKEN=//p' "$GITHUB_ENV_FILE" 2>/dev/null | tail -1)"
for value in "$HF_TOKEN" "$GH_TOKEN_VALUE" "$(env_value CONTAINER_API_KEY)" "$(env_value JUPYTER_TOKEN)"; do
  [ "${#value}" -ge 8 ] || continue
  bad+="$(grep -rlF -- "$value" "$STAGE" || true)"$'\n'
done
bad+="$(grep -rlE 'hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}' "$STAGE" || true)"$'\n'
bad+="$(find "$STAGE" -type f \( -name '.credentials.json' -o -name 'id_rsa*' -o -name 'id_ed25519*' \))"
bad="$(printf '%s\n' "$bad" | sed "s|^$STAGE/||" | sort -u | sed '/^$/d')"
[ -z "$bad" ] || die "credential found, nothing uploaded. Remove it from: $(echo "$bad" | tr '\n' ' ')"

STAMP="$(date -u +%Y-%m-%dT%H:%MZ)"
echo "uploading $(find "$STAGE" -type f | wc -l | tr -d ' ') files ($(du -sh "$STAGE" | cut -f1)) -> $REPO (dataset) : $DEST"
for attempt in 1 2 3; do
  if uvx --from huggingface_hub hf upload "$REPO" "$STAGE" "$DEST" --repo-type dataset --delete "*" \
       --commit-message "sweep code $STAMP${MESSAGE:+: $MESSAGE}"; then
    echo "uploaded. On the laptop: bash sweep/pull_box.sh --from-hf $REPO"
    exit 0
  fi
  echo "upload attempt $attempt failed" >&2
  [ "$attempt" = 3 ] || sleep 60
done
die "code upload failed after 3 attempts; the code is still in $REPO_DIR"
