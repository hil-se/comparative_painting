#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=art-locked-sidhu
#SBATCH --array=0-239%8
#SBATCH --gres=shard:a100:32
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL
#SBATCH --output=/scratch/xx4455/paper-projects/logs/art-locked-sidhu-%A_%a.out
#SBATCH --error=/scratch/xx4455/paper-projects/logs/art-locked-sidhu-%A_%a.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
export TF_GPU_MEMORY_LIMIT_MB="${TF_GPU_MEMORY_LIMIT_MB:-16384}"
export PYTHONDONTWRITEBYTECODE=1
export KERAS_HOME=/scratch/xx4455/.keras
export XDG_CACHE_HOME=/scratch/xx4455/.cache
input_root=/home/xx4455/paper-projects/artifacts/comparative_painting
locked_root=/scratch/xx4455/paper-projects/artifacts/comparative_painting/locked_head
output_root="${locked_root}/aggregate_split/sidhu"
categories=(abstract abstract abstract abstract representational representational representational representational)
targets=(beauty beauty liking liking beauty beauty liking liking)
representations=(resnet50 clip-vit-b32 resnet50 clip-vit-b32 resnet50 clip-vit-b32 resnet50 clip-vit-b32)
objectives=(regression hinge bradley_terry)

seed=$((SLURM_ARRAY_TASK_ID % 10))
objective_index=$(((SLURM_ARRAY_TASK_ID / 10) % 3))
case_index=$((SLURM_ARRAY_TASK_ID / 30))
category="${categories[${case_index}]}"
target="${targets[${case_index}]}"
representation="${representations[${case_index}]}"
objective="${objectives[${objective_index}]}"
output="${output_root}/sidhu-${category}-${target}-${representation}-${objective}-s${seed}.csv"
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
        --expected-dataset sidhu \
        --expected-representation "${representation}" \
        --expected-category "${category}" \
        --expected-target "${target}" \
        --expected-objective "${objective}" \
        --expected-seed "${seed}" \
        --manifest "${input_root}/manifests/sidhu.csv" \
        --features "${input_root}/features/sidhu-${representation}.npz"; then
    echo "validated existing ${output}"
    exit 0
fi

python code/extensions/run_art_locked_head.py \
    --manifest "${input_root}/manifests/sidhu.csv" \
    --features "${input_root}/features/sidhu-${representation}.npz" \
    --dataset sidhu \
    --representation "${representation}" \
    --category "${category}" \
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
    --expected-dataset sidhu \
    --expected-representation "${representation}" \
    --expected-category "${category}" \
    --expected-target "${target}" \
    --expected-objective "${objective}" \
    --expected-seed "${seed}" \
    --manifest "${input_root}/manifests/sidhu.csv" \
    --features "${input_root}/features/sidhu-${representation}.npz"
