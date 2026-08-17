#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=art-apdd-reg-smoke
#SBATCH --gres=shard:a100:32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-apdd-reg-smoke-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-apdd-reg-smoke-%j.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
export TF_GPU_MEMORY_LIMIT_MB=24576
export PYTHONUNBUFFERED=1
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting
output_root="${artifact_root}/tuning/apddv2-regression/smoke"
mkdir -p "${output_root}"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/tune_apddv2_regression.py \
    --phase screen \
    --manifest "${artifact_root}/manifests/apddv2.csv" \
    --features "${artifact_root}/features/apddv2-clip-vit-b32.npz" \
    --target Color \
    --config-ids deep-mae-current-spearstop,ridge-z-a1 \
    --seeds 0 \
    --epochs 2 \
    --patience 1 \
    --output "${output_root}/smoke.csv"
