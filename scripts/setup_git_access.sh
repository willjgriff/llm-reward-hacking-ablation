#!/usr/bin/env bash
# Give root on a box or VM git access to the GitHub repos a fine-grained token covers.
#   bash scripts/setup_git_access.sh        # as root, after github.env was copied over
# Reads GH_TOKEN (and optionally GIT_USER_NAME / GIT_USER_EMAIL) from ~/.config/rhablation/github.env
# (override with RHABLATION_GITHUB_ENV). The credential helper reads the file each time git asks, so the
# token never lands in a .gitconfig, a remote URL or a clone's .git/config. Root only: the unprivileged
# benchmark user must never run git with it.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root"
GITHUB_ENV_FILE="${RHABLATION_GITHUB_ENV:-$HOME/.config/rhablation/github.env}"
[ -f "$GITHUB_ENV_FILE" ] || die "no $GITHUB_ENV_FILE; copy it from the laptop (setup.md 'Git access from boxes')"
chmod 600 "$GITHUB_ENV_FILE"
grep -q '^GH_TOKEN=' "$GITHUB_ENV_FILE" || die "$GITHUB_ENV_FILE must define GH_TOKEN"
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }

git config --global --unset-all credential.https://github.com.helper 2>/dev/null || true
git config --global credential.https://github.com.helper \
  "!f() { test \"\$1\" = get || exit 0; . '$GITHUB_ENV_FILE'; echo username=x-access-token; echo \"password=\$GH_TOKEN\"; }; f"

set -a
# shellcheck disable=SC1090
. "$GITHUB_ENV_FILE"
set +a
git config --global user.name "${GIT_USER_NAME:-rhablation box}"
git config --global user.email "${GIT_USER_EMAIL:-box@rhablation.invalid}"
echo "git access configured for root ($(git config --global user.name) <$(git config --global user.email)>)"
