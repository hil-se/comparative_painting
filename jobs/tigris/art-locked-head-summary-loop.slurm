#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=art-locked-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL
#SBATCH --output=/scratch/xx4455/paper-projects/logs/art-locked-summary-%j.out
#SBATCH --error=/scratch/xx4455/paper-projects/logs/art-locked-summary-%j.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/scratch/xx4455/.cache
locked_root=/scratch/xx4455/paper-projects/artifacts/comparative_painting/locked_head
mkdir -p "${XDG_CACHE_HOME}"
cd /home/xx4455/paper-projects/comparative_painting

python code/extensions/merge_art_locked_head_results.py \
    --mode sidhu \
    --input-dir "${locked_root}/aggregate_split/sidhu" \
    --output-dir "${locked_root}/aggregate_merged/sidhu"

python code/extensions/merge_art_locked_head_results.py \
    --mode apddv2 \
    --input-dir "${locked_root}/aggregate_split/apddv2" \
    --output-dir "${locked_root}/aggregate_merged/apddv2"

python code/extensions/merge_art_locked_head_results.py \
    --mode rater \
    --input-dir "${locked_root}/rater_split" \
    --output-dir "${locked_root}/rater_merged"

python code/extensions/summarize_sidhu_extensions.py \
    --raw-dir "${locked_root}/aggregate_merged/sidhu" \
    --output-dir "${locked_root}/summary/sidhu"

python code/extensions/summarize_apddv2_extensions.py \
    --raw-dir "${locked_root}/aggregate_merged/apddv2" \
    --output-dir "${locked_root}/summary/apddv2"

python code/extensions/summarize_sidhu_rater_extensions.py \
    --raw-dir "${locked_root}/rater_merged" \
    --output-dir "${locked_root}/summary/sidhu_rater"
