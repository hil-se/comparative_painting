#!/bin/bash -l
#SBATCH --job-name=art-gpu-smoke
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -z "${TIGRIS_REPO_ROOT:-}" || -z "${TIGRIS_ML_ENV:-}" ]]; then
    echo "TIGRIS_REPO_ROOT and TIGRIS_ML_ENV must be exported." >&2
    exit 2
fi

OUTPUT_ROOT="${TIGRIS_OUTPUT_ROOT:-${TIGRIS_REPO_ROOT}/reproduction/results}"
RUN_ID="${TIGRIS_RUN_ID:-manual-${SLURM_JOB_ID}}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${RUN_ID}/tigris_gpu_smoke"
CACHE_DIRECTORY="${OUTPUT_ROOT}/${RUN_ID}/.cache/matplotlib"

cd "${TIGRIS_REPO_ROOT}"
mkdir -p "${OUTPUT_DIRECTORY}" "${CACHE_DIRECTORY}"
spack env activate "${TIGRIS_ML_ENV}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

export MPLCONFIGDIR="${CACHE_DIRECTORY}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Architecture: $(uname -m)"
echo "Python: ${PYTHON_BIN}"
echo "Spack environment: ${TIGRIS_ML_ENV}"
nvidia-smi

"${PYTHON_BIN}" -c \
    'import numpy, pandas, scipy, sklearn, tensorflow as tf; devices=tf.config.list_physical_devices("GPU"); print("TensorFlow", tf.__version__, "GPUs", devices); assert devices, "TensorFlow did not detect the allocated GPU"'

"${PYTHON_BIN}" reproduction/run_art_average_baseline.py \
    --repo "${TIGRIS_REPO_ROOT}" \
    --output "${OUTPUT_DIRECTORY}" \
    --methods regression comparative \
    --paintings abstract \
    --rating-types beauty \
    --seeds 0 \
    --n-values 10 \
    --workers 1 \
    --threads-per-worker "${SLURM_CPUS_PER_TASK}"

echo "TIGRIS GPU smoke test completed successfully."
