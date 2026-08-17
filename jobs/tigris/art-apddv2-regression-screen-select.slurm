#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=art-apdd-reg-screen-select
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-apdd-reg-screen-select-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-apdd-reg-screen-select-%j.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting
phase_root="${artifact_root}/tuning/apddv2-regression/screen"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/summarize_apddv2_regression_tuning.py \
    --phase screen \
    --input-dir "${phase_root}/raw" \
    --output-dir "${phase_root}/summary" \
    --seeds 0-2 \
    --top-k 3
