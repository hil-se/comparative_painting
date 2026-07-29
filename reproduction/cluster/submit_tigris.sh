#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: $0 RIT_USERNAME SLURM_ACCOUNT SPACK_ML_ENV [BRANCH] [REMOTE_ROOT]" >&2
    exit 2
fi

RIT_USERNAME="$1"
SLURM_ACCOUNT="$2"
SPACK_ML_ENV="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BRANCH="${4:-$(git -C "${REPOSITORY_ROOT}" branch --show-current)}"
REMOTE_ROOT="${5:-masters-art-repro}"
TIGRIS_HOST="tigris.rc.rit.edu"
TIGRIS_LOGIN="${RIT_USERNAME}@${TIGRIS_HOST}"

for value in "${RIT_USERNAME}" "${SLURM_ACCOUNT}" "${SPACK_ML_ENV}" "${REMOTE_ROOT}"; do
    if [[ ! "${value}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
        echo "Unsafe or unsupported argument: ${value}" >&2
        exit 2
    fi
done

if [[ "${REMOTE_ROOT}" = /* || "${REMOTE_ROOT}" = *..* ]]; then
    echo "REMOTE_ROOT must be a safe path relative to the TIGRIS home directory." >&2
    exit 2
fi

if ! git check-ref-format --branch "${BRANCH}" >/dev/null; then
    echo "Invalid Git branch: ${BRANCH}" >&2
    exit 2
fi

"${SCRIPT_DIR}/sync_tigris.sh" \
    "${RIT_USERNAME}" \
    "${BRANCH}" \
    "${REMOTE_ROOT}"

ssh -o BatchMode=yes "${TIGRIS_LOGIN}" \
    bash -s -- \
    "${REMOTE_ROOT}" \
    "${BRANCH}" \
    "${SLURM_ACCOUNT}" \
    "${SPACK_ML_ENV}" <<'REMOTE_SCRIPT'
set -euo pipefail

remote_root="$1"
branch="$2"
slurm_account="$3"
spack_ml_env="$4"
repository="${HOME}/${remote_root}/repo"
log_directory="${HOME}/${remote_root}/logs"
output_root="${HOME}/${remote_root}/runs"

cd "${repository}"
if [[ "$(git branch --show-current)" != "${branch}" ]]; then
    echo "TIGRIS checkout is not on the requested branch: ${branch}" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "TIGRIS checkout has uncommitted changes; refusing to submit." >&2
    git status --short >&2
    exit 2
fi

mkdir -p "${log_directory}" "${output_root}"
commit="$(git rev-parse HEAD)"
run_id="${commit:0:12}-$(date -u +%Y%m%dT%H%M%SZ)"

smoke_job_id="$(
    sbatch \
        --parsable \
        --account="${slurm_account}" \
        --output="${log_directory}/%x_%j.out" \
        --error="${log_directory}/%x_%j.err" \
        --export="ALL,TIGRIS_REPO_ROOT=${repository},TIGRIS_OUTPUT_ROOT=${output_root},TIGRIS_RUN_ID=${run_id},TIGRIS_ML_ENV=${spack_ml_env}" \
        reproduction/cluster/tigris_gpu_smoke.sbatch
)"

job_id="$(
    sbatch \
        --parsable \
        --dependency="afterok:${smoke_job_id}" \
        --account="${slurm_account}" \
        --output="${log_directory}/%x_%j.out" \
        --error="${log_directory}/%x_%j.err" \
        --export="ALL,TIGRIS_REPO_ROOT=${repository},TIGRIS_OUTPUT_ROOT=${output_root},TIGRIS_RUN_ID=${run_id},TIGRIS_ML_ENV=${spack_ml_env}" \
        reproduction/cluster/tigris_art_average_baseline.sbatch
)"

echo "Commit: ${commit}"
echo "Run ID: ${run_id}"
echo "Submitted smoke job ${smoke_job_id}."
echo "Submitted dependent full job ${job_id}."
echo "Monitor: squeue --job ${smoke_job_id},${job_id}"
echo "Logs: ${log_directory}/"
echo "Results: ${output_root}/${run_id}/"
REMOTE_SCRIPT
