#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_NAME="active-bytes-law"

cd "${REPO_ROOT}"

command -v gh >/dev/null 2>&1 || {
  printf 'GitHub CLI is required: https://cli.github.com/\n' >&2
  exit 2
}

gh auth status >/dev/null
OWNER="$(gh api user --jq .login)"

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Refusing to publish a dirty working tree. Commit or stash changes first.\n' >&2
  exit 2
fi

python3 scripts/check_publication_safety.py
python3 -m unittest discover -s tests

if git remote get-url origin >/dev/null 2>&1; then
  printf 'Remote origin already exists: %s\n' "$(git remote get-url origin)" >&2
  exit 2
fi

if gh repo view "${OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
  printf 'Refusing to reuse existing repository %s/%s without manual review.\n' \
    "${OWNER}" "${REPO_NAME}" >&2
  exit 2
fi

gh repo create "${OWNER}/${REPO_NAME}" \
  --public \
  --description "Auditable experiments and artifacts for the Active-Bytes Law of LLM inference energy" \
  --source "${REPO_ROOT}" \
  --remote origin \
  --push

gh repo view "${OWNER}/${REPO_NAME}" --json nameWithOwner,visibility,url
