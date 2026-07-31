#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=art-prepare-apddv2
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-prepare-apddv2-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-prepare-apddv2-%j.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting
dataset_root="${artifact_root}/datasets/apddv2"
export HF_HOME=/home/xx4455/paper-projects/artifacts/comparative_image_caption/huggingface

mkdir -p "${artifact_root}/manifests" "${artifact_root}/features"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/build_art_manifests.py \
    --dataset apddv2 \
    --annotations "${dataset_root}/APDDv2-10023.csv" \
    --images "${dataset_root}/images" \
    --max-missing-images 1 \
    --output "${artifact_root}/manifests/apddv2.csv"

python code/extensions/extract_art_features.py \
    --manifest "${artifact_root}/manifests/apddv2.csv" \
    --representation resnet50 \
    --output "${artifact_root}/features/apddv2-resnet50.npz" \
    --batch-size 256

python code/extensions/extract_art_features.py \
    --manifest "${artifact_root}/manifests/apddv2.csv" \
    --representation clip-vit-b32 \
    --output "${artifact_root}/features/apddv2-clip-vit-b32.npz" \
    --cache-dir "${HF_HOME}" \
    --batch-size 256
