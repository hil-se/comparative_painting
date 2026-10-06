# Reproducing the Paper

## Paper Tables and Figures

All paths below are relative to the repository root.

| Paper result | Experiment data | Regenerated output |
|--------------|-----------------|--------------------|
| RQ1: OLS baseline and sensitivity | `Data/*_Data.csv`, raw rater tables, painting images | `results/baseline/heldout_ols/` |
| RQ1: ResNet/CLIP regression | `results/deep_learning/aggregate/{sidhu,apddv2}/` | `results/deep_learning/paper/rq1_*` |
| RQ2: focal budgets and full sweep | The same aggregate CSVs | `results/deep_learning/paper/rq2_*` |
| RQ3: aggregate/within/cross | Aggregate CSVs and `results/deep_learning/rater/` | `results/deep_learning/paper/rq3_*` |
| RQ4: annotation time and agreement | Qualtrics export and `results/human_survey/survey_data/` | `results/human_survey/paper/` |
| Aggregate and rater budget plots | Seed-level mean Spearman | `figures/art_clip_n_sweep.png`, `figures/art_rater_n_sweep.png` |
| Prediction head selection | `results/deep_learning/head_selection/{screen,confirm}/raw/` | `results/deep_learning/paper/head_*` |

The published AANSPS, SAAN, and ArtCLIP rows in the paper are contextual
values from Jin et al.'s APDDv2 paper, rather than experiments run here.

Run the [repository README](../../README.md) commands to regenerate the
OLS results, tables, and figures. The scripts also accept output-directory
arguments. `statistical_analysis.py` implements exact paired Wilcoxon
signed-rank inference with average tied ranks and the paper's Holm families:
nine RQ1 contrasts, three contrasts per dataset/focal budget, and twenty
contrasts per dataset over the full N=1–10 sweep. RQ3 is descriptive.

## Prepare Features

Install `requirements-training.txt` before feature extraction or training.
The default Python commands run from the repository root.

### Sidhu

```bash
python code/deep_learning/build_art_manifests.py --dataset sidhu \
  --output build/manifests/sidhu.csv \
  --resnet-output build/features/sidhu-resnet50.npz
```

Use `Data/fixed_features/sidhu-clip-vit-b32.npz` for CLIP. The builder
packages the released original-size ResNet arrays under
`code/deep_learning/feature/`. `data_origin.py` reproduces that extraction
without the standard ResNet input transform, following the paper's Sidhu
pipeline. To extract fresh CLIP vectors:

```bash
python code/deep_learning/extract_art_features.py \
  --manifest build/manifests/sidhu.csv --representation clip-vit-b32 \
  --output build/features/sidhu-clip-vit-b32.npz --device cuda
```

### APDDv2

Download the annotations and images from the
[official source](https://github.com/BestiVictory/APDDv2), which specifies
CC BY-NC-ND 4.0, and prepare them locally:

```bash
python code/deep_learning/build_art_manifests.py --dataset apddv2 \
  --annotations /path/to/APDDv2-10023.csv --images /path/to/images \
  --output build/manifests/apddv2.csv
python code/deep_learning/extract_art_features.py \
  --manifest build/manifests/apddv2.csv --representation clip-vit-b32 \
  --output build/features/apddv2-clip-vit-b32.npz --device cuda
python code/deep_learning/extract_art_features.py \
  --manifest build/manifests/apddv2.csv --representation resnet50 \
  --output build/features/apddv2-resnet50.npz
```

CLIP uses its standard image processor and L2-normalized 512-dimensional
image embeddings. APDDv2 ResNet uses standard Keras preprocessing and
2048-dimensional vectors. Each APDDv2 target uses its finite, nonempty labels;
the valid population ranges from 3,818 to 10,022 images.

## Train the Models

The repository README gives an aggregate Sidhu run. For APDDv2:

```bash
python code/deep_learning/run_art_locked_head.py \
  --manifest build/manifests/apddv2.csv \
  --features build/features/apddv2-clip-vit-b32.npz \
  --dataset apddv2 --representation clip-vit-b32 --category all \
  --target "Total aesthetic score" \
  --objectives regression,hinge,bradley_terry --n-values 1-10 --seeds 0-9 \
  --output build/reruns/apddv2-total-clip.csv
```

Repeat for the 11 targets in `build_art_manifests.APDD_TARGETS` and both
representations. Each target is modeled separately; APDDv2 headline results
are unweighted means across target-level metrics.

Rater-level Sidhu example:

```bash
python code/deep_learning/run_sidhu_rater_locked_head.py \
  --features Data/fixed_features/sidhu-clip-vit-b32.npz --data-dir Data \
  --category abstract --target beauty --mode within --rater 1 \
  --objectives regression,hinge,bradley_terry --n-values 1-10 --seeds 0-9 \
  --output build/reruns/sidhu-abstract-beauty-within-r1.csv
```

Repeat for both categories, both rating targets, modes `within` and `cross`,
and raters 1–5. Within-rater labels come from the target rater; cross-rater
training labels average the other four raters, with evaluation against the
target rater. The selected prediction head remains fixed in all runs.

## Select the Prediction Head

Selection uses APDDv2 CLIP regression and validation Spearman. For each of
its 11 targets, screen the 22 configurations with seeds 0–2:

```bash
python code/deep_learning/tune_apddv2_regression.py --phase screen \
  --manifest build/manifests/apddv2.csv \
  --features build/features/apddv2-clip-vit-b32.npz \
  --target "Total aesthetic score" --seeds 0-2 \
  --output build/selection/screen/raw/total.csv
```

After running all targets, select the top three:

```bash
python code/deep_learning/summarize_apddv2_regression_tuning.py \
  --phase screen --input-dir build/selection/screen/raw \
  --output-dir build/selection/screen/summary --seeds 0-2
```

Confirm those configurations for every target using seeds 0–9, then rank
validation results:

```bash
python code/deep_learning/tune_apddv2_regression.py --phase confirm \
  --manifest build/manifests/apddv2.csv \
  --features build/features/apddv2-clip-vit-b32.npz \
  --target "Total aesthetic score" --seeds 0-9 \
  --config-ids-file build/selection/screen/summary/selected.json \
  --output build/selection/confirm/raw/total.csv
python code/deep_learning/summarize_apddv2_regression_tuning.py \
  --phase confirm --input-dir build/selection/confirm/raw \
  --selection build/selection/screen/summary/selected.json \
  --output-dir build/selection/confirm/summary --seeds 0-9
```

The released winner is `shallow-mse-z-gelu-ln-rawclip`, with validation macro
Spearman 0.77719858. The canonical experiment CSVs use that configuration.
