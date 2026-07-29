#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 RIT_USERNAME [BRANCH] [REMOTE_ROOT]" >&2
    exit 2
fi

RIT_USERNAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BRANCH="${2:-$(git -C "${REPOSITORY_ROOT}" branch --show-current)}"
REMOTE_ROOT="${3:-masters-art-repro}"
REPOSITORY_URL="https://github.com/hil-se/comparative_painting.git"
TIGRIS_LOGIN="${RIT_USERNAME}@tigris.rc.rit.edu"

if [[ ! "${RIT_USERNAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid RIT username: ${RIT_USERNAME}" >&2
    exit 2
fi
if [[ ! "${REMOTE_ROOT}" =~ ^[A-Za-z0-9._/-]+$ ]] ||
    [[ "${REMOTE_ROOT}" = /* || "${REMOTE_ROOT}" = *..* ]]; then
    echo "REMOTE_ROOT must be a safe path relative to the TIGRIS home directory." >&2
    exit 2
fi
if ! git check-ref-format --branch "${BRANCH}" >/dev/null; then
    echo "Invalid Git branch: ${BRANCH}" >&2
    exit 2
fi
if ! git ls-remote --exit-code --heads "${REPOSITORY_URL}" \
    "refs/heads/${BRANCH}" >/dev/null; then
    echo "Branch is not available on GitHub: ${BRANCH}" >&2
    echo "Push it before syncing TIGRIS." >&2
    exit 2
fi

ssh -o BatchMode=yes "${TIGRIS_LOGIN}" \
    bash -s -- \
    "${REMOTE_ROOT}" \
    "${BRANCH}" \
    "${REPOSITORY_URL}" <<'REMOTE_SCRIPT'
set -euo pipefail

remote_root="$1"
branch="$2"
repository_url="$3"
workspace="${HOME}/${remote_root}"
repository="${workspace}/repo"

mkdir -p "${workspace}/logs" "${workspace}/runs"

if [[ ! -d "${repository}/.git" ]]; then
    if [[ -e "${repository}" ]]; then
        echo "Refusing to replace non-Git path: ${repository}" >&2
        exit 2
    fi
    git clone \
        --filter=blob:none \
        --single-branch \
        --branch "${branch}" \
        "${repository_url}" \
        "${repository}"
else
    cd "${repository}"
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "TIGRIS checkout has uncommitted changes; refusing to overwrite them." >&2
        git status --short >&2
        exit 2
    fi
    git fetch --prune origin
    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        git switch "${branch}"
    else
        git switch --track -c "${branch}" "origin/${branch}"
    fi
    git pull --ff-only origin "${branch}"
fi

cd "${repository}"
echo "TIGRIS repository: ${repository}"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"
REMOTE_SCRIPT
