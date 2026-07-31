#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=art-validate-apdd-memory
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-validate-apdd-memory-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-validate-apdd-memory-%j.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
export TF_GPU_MEMORY_LIMIT_MB=24576
export PYTHONUNBUFFERED=1
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/run_art_extensions.py \
    --manifest "${artifact_root}/manifests/apddv2.csv" \
    --features "${artifact_root}/features/apddv2-resnet50.npz" \
    --dataset apddv2 \
    --representation resnet50 \
    --target "Total aesthetic score" \
    --objectives regression \
    --seeds 0 \
    --epochs-regression 200 \
    --batch-size 256 \
    --output "${artifact_root}/results/apddv2-memory-validation.csv" \
    --checkpoint-dir "${artifact_root}/checkpoints/apddv2-memory-validation"
