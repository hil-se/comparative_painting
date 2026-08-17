#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=art-locked-smoke
#SBATCH --gres=shard:a100:32
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL
#SBATCH --output=/scratch/xx4455/paper-projects/logs/art-locked-smoke-%j.out
#SBATCH --error=/scratch/xx4455/paper-projects/logs/art-locked-smoke-%j.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
export TF_GPU_MEMORY_LIMIT_MB="${TF_GPU_MEMORY_LIMIT_MB:-16384}"
export PYTHONDONTWRITEBYTECODE=1
export KERAS_HOME=/scratch/xx4455/.keras
export XDG_CACHE_HOME=/scratch/xx4455/.cache
input_root=/home/xx4455/paper-projects/artifacts/comparative_painting
locked_root=/scratch/xx4455/paper-projects/artifacts/comparative_painting/locked_head
output_root="${locked_root}/smoke"
output="${output_root}/sidhu-abstract-beauty-clip-vit-b32-all-s0.csv"
mkdir -p "${output_root}" "${KERAS_HOME}" "${XDG_CACHE_HOME}"
cd /home/xx4455/paper-projects/comparative_painting

python code/extensions/run_art_locked_head.py \
    --manifest "${input_root}/manifests/sidhu.csv" \
    --features "${input_root}/features/sidhu-clip-vit-b32.npz" \
    --dataset sidhu \
    --representation clip-vit-b32 \
    --category abstract \
    --target beauty \
    --objectives regression,hinge,bradley_terry \
    --n-values 1 \
    --seeds 0 \
    --epochs-regression 200 \
    --epochs-pairwise 200 \
    --patience 20 \
    --output "${output}"

python code/extensions/validate_art_result.py "${output}" \
    --expected-rows 3 \
    --expected-dataset sidhu \
    --expected-representation clip-vit-b32 \
    --expected-category abstract \
    --expected-target beauty \
    --expected-seed 0 \
    --manifest "${input_root}/manifests/sidhu.csv" \
    --features "${input_root}/features/sidhu-clip-vit-b32.npz"

jq -e '.method_id == "shallow-mse-z-gelu-ln-rawclip"' \
    "${output%.csv}.metadata.json" >/dev/null
