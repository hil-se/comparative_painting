#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=art-prepare-sidhu
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-prepare-sidhu-%j.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-prepare-sidhu-%j.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting
export HF_HOME=/home/xx4455/paper-projects/artifacts/comparative_image_caption/huggingface

mkdir -p "${artifact_root}/manifests" "${artifact_root}/features"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/build_art_manifests.py \
    --dataset sidhu \
    --output "${artifact_root}/manifests/sidhu.csv" \
    --resnet-output "${artifact_root}/features/sidhu-resnet50.npz"

python code/extensions/extract_art_features.py \
    --manifest "${artifact_root}/manifests/sidhu.csv" \
    --representation clip-vit-b32 \
    --output "${artifact_root}/features/sidhu-clip-vit-b32.npz" \
    --cache-dir "${HF_HOME}" \
    --batch-size 256
