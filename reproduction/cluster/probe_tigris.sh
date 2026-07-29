#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RIT_USERNAME" >&2
    exit 2
fi

RIT_USERNAME="$1"
TIGRIS_HOST="tigris.rc.rit.edu"
TIGRIS_LOGIN="${RIT_USERNAME}@${TIGRIS_HOST}"

echo "== Login host =="
ssh -o BatchMode=yes "${TIGRIS_LOGIN}" hostname

echo
echo "== Architecture =="
ssh -o BatchMode=yes "${TIGRIS_LOGIN}" uname -m

echo
echo "== Slurm accounts =="
ssh -o BatchMode=yes "${TIGRIS_LOGIN}" 'bash -lc "my-accounts"'

echo
echo "== TIGRIS partitions and GPU resources =="
ssh -o BatchMode=yes "${TIGRIS_LOGIN}" \
    'bash -lc "sinfo --partition=tigris --format=%P,%a,%l,%D,%G"'

echo
echo "== Available Spack environments =="
ssh -o BatchMode=yes "${TIGRIS_LOGIN}" 'bash -lc "spack env list"'
