#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=art-locked-apdd
#SBATCH --array=0-659%8
#SBATCH --gres=shard:a100:32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL
#SBATCH --output=/scratch/xx4455/paper-projects/logs/art-locked-apdd-%A_%a.out
#SBATCH --error=/scratch/xx4455/paper-projects/logs/art-locked-apdd-%A_%a.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
export TF_GPU_MEMORY_LIMIT_MB="${TF_GPU_MEMORY_LIMIT_MB:-16384}"
export PYTHONDONTWRITEBYTECODE=1
export KERAS_HOME=/scratch/xx4455/.keras
export XDG_CACHE_HOME=/scratch/xx4455/.cache
input_root=/home/xx4455/paper-projects/artifacts/comparative_painting
locked_root=/scratch/xx4455/paper-projects/artifacts/comparative_painting/locked_head
output_root="${locked_root}/aggregate_split/apddv2"
targets=(
    "Total aesthetic score"
    "Theme and logic"
    "Creativity"
    "Layout and composition"
    "Space and perspective"
    "The sense of order"
    "Light and shadow"
    "Color"
    "Details and texture"
    "The overall"
    "Mood"
)
slugs=(total theme-logic creativity layout-composition space-perspective order light-shadow color details-texture overall mood)
representations=(resnet50 clip-vit-b32)
objectives=(regression hinge bradley_terry)

seed=$((SLURM_ARRAY_TASK_ID % 10))
objective_index=$(((SLURM_ARRAY_TASK_ID / 10) % 3))
case_index=$((SLURM_ARRAY_TASK_ID / 30))
target_index=$((case_index / 2))
representation_index=$((case_index % 2))
target="${targets[${target_index}]}"
slug="${slugs[${target_index}]}"
representation="${representations[${representation_index}]}"
objective="${objectives[${objective_index}]}"
output="${output_root}/apddv2-${slug}-${representation}-${objective}-s${seed}.csv"
expected_rows=10
if [[ "${objective}" == regression ]]; then
    expected_rows=1
fi

mkdir -p "${output_root}" "${KERAS_HOME}" "${XDG_CACHE_HOME}"
cd /home/xx4455/paper-projects/comparative_painting
sleep $((SLURM_ARRAY_TASK_ID % 4 * 5))

if [[ -f "${output}" && -f "${output%.csv}.metadata.json" ]] \
    && jq -e '.method_id == "shallow-mse-z-gelu-ln-rawclip"' \
        "${output%.csv}.metadata.json" >/dev/null \
    && python code/extensions/validate_art_result.py "${output}" \
        --expected-rows "${expected_rows}" \
        --expected-dataset apddv2 \
        --expected-representation "${representation}" \
        --expected-category all \
        --expected-target "${target}" \
        --expected-objective "${objective}" \
        --expected-seed "${seed}" \
        --manifest "${input_root}/manifests/apddv2.csv" \
        --features "${input_root}/features/apddv2-${representation}.npz"; then
    echo "validated existing ${output}"
    exit 0
fi

python code/extensions/run_art_locked_head.py \
    --manifest "${input_root}/manifests/apddv2.csv" \
    --features "${input_root}/features/apddv2-${representation}.npz" \
    --dataset apddv2 \
    --representation "${representation}" \
    --target "${target}" \
    --objectives "${objective}" \
    --n-values 1-10 \
    --seeds "${seed}" \
    --epochs-regression 200 \
    --epochs-pairwise 200 \
    --patience 20 \
    --output "${output}"

python code/extensions/validate_art_result.py "${output}" \
    --expected-rows "${expected_rows}" \
    --expected-dataset apddv2 \
    --expected-representation "${representation}" \
    --expected-category all \
    --expected-target "${target}" \
    --expected-objective "${objective}" \
    --expected-seed "${seed}" \
    --manifest "${input_root}/manifests/apddv2.csv" \
    --features "${input_root}/features/apddv2-${representation}.npz"
