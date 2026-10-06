# Comparative Learning for Art Aesthetics

Replication package for *Comparative Learning for Art Aesthetics:
Representation, Scale, and Annotation Efficiency* (IEEE Access revision,
October 2026).

Authors: Manoj Reddy Bethi, Xiaoyin Xi, Sai Rupa Jhade, Pravallika Yaganti,
Monoshiz Mahbub Khan, and Zhe Yu.

## Reproduce the paper analyses on a CPU

Use Python 3.12 from a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python code/extensions/run_sidhu_heldout_ols.py
python code/extensions/reproduce_art_paper.py
python -m unittest discover -s tests -v
```

The OLS runner audits 474 predictor/image joins, restores original painting
IDs, and reproduces the held-out baseline and complete-case sensitivity.
The analysis runner verifies all 78 final neural-result CSVs and their
metadata (14,700 fits), validates the recovered head-selection evidence,
and writes RQ1--RQ4 tables, all 61 statistical contrasts, and budget figures
to `results/paper/`. It requires no cluster access or APDDv2 image download.
TensorFlow-specific tests run when the training dependencies are installed.

See [the replication guide](docs/replication.md) for the table-to-source map,
external data preparation, full training commands, and provenance limits.

## Controlled study

The final experiments compare fixed ResNet-50 and CLIP ViT-B/32 visual
representations using one downstream method selected on APDDv2 validation
data and then locked for every dataset, target, objective, rater, and seed.

- Datasets: four Sidhu conditions and all 11 APDDv2 targets.
- Splits: Sidhu uses 140 training images, 20 validation images, and the
  remainder for testing; APDDv2 uses 70/15/15 splits re-randomized per seed.
- Head: `256 -> 64 -> 1`, GELU, LayerNorm, dropout 0.1, and L2 weight decay
  `1e-5`.
- Features: raw fixed extractor outputs; no additional feature
  standardization.
- Regression: MSE on training-standardized targets.
- Pairwise objectives: hinge ranking and Bradley--Terry logistic loss.
- Pair budget: `N = M / n_train`, swept from `N=1` through `N=10`.
- Optimization: Adam at `1e-3`, batch size 128, at most 200 epochs,
  validation-Spearman early stopping after at least 25 epochs with patience
  20, and learning-rate halving after 10 plateaus.
- Selection: the locked head was chosen from 22 configurations using APDDv2
  validation macro Spearman. Test data were not used for selection.

Recovered screening and confirmation CSVs, sidecars, rankings, and selection
records are in `results/extensions/head_selection/`. The screen used seeds
0--2; the top three configurations were confirmed using seeds 0--9. The
winner's validation macro Spearman is 0.77719858.

The authoritative implementation is
`code/extensions/run_art_locked_head.py`, with rater-level evaluation in
`code/extensions/run_sidhu_rater_locked_head.py`. Older scripts and result
directories are retained as historical artifacts and are not authoritative
for the final paper.

## Repository layout

```text
comparative_painting/
├── Data/                              # source data and provenance notes
├── code/extensions/
│   ├── build_art_manifests.py
│   ├── extract_art_features.py
│   ├── tune_apddv2_regression.py
│   ├── run_art_locked_head.py
│   ├── run_sidhu_rater_locked_head.py
│   └── manuscript_statistical_tests.py
├── jobs/tigris/                       # loop-account Slurm jobs
├── results/extensions/locked_head/    # final CSVs, metadata, and summaries
├── docs/                              # controlled protocol and results
└── tests/                             # manifest, feature, and result checks
```

## Reproducing one locked-head run

Install `requirements-training.txt` first. Prepare Sidhu from the bundled
images, released ResNet arrays, and recovered CLIP feature bundle:

```bash
python code/extensions/build_art_manifests.py \
  --dataset sidhu \
  --output build/manifests/sidhu.csv \
  --resnet-output build/features/sidhu-resnet50.npz
```

For example:

```bash
python code/extensions/run_art_locked_head.py \
  --manifest build/manifests/sidhu.csv \
  --features Data/fixed_features/sidhu-clip-vit-b32.npz \
  --dataset sidhu \
  --representation clip-vit-b32 \
  --category abstract \
  --target beauty \
  --objectives regression,hinge,bradley_terry \
  --n-values 1-10 \
  --seeds 0-9 \
  --epochs-regression 200 \
  --epochs-pairwise 200 \
  --patience 20 \
  --output results/extensions/locked_head/example.csv
```

Cluster launch scripts use the `loop` account on SPORC's `onboard`
partition, including fractional A100 shards for the locked-head matrix.

## Main results

Representation comparison under regression:

| Dataset | ResNet-50 Spearman | CLIP Spearman |
|---|---:|---:|
| Sidhu, mean over four conditions | 0.534 | 0.732 |
| APDDv2, mean over 11 targets | 0.701 | 0.775 |

CLIP results at `N=10`:

| Dataset | Regression | Hinge | Bradley--Terry |
|---|---:|---:|---:|
| Sidhu | 0.732 | 0.767 | 0.772 |
| APDDv2 | 0.775 | 0.764 | 0.764 |

On Sidhu, both pairwise gains over regression are significant after Holm
correction; hinge and Bradley--Terry are not significantly different. On
APDDv2, the pairwise objectives are approximately 0.011 below regression and
are indistinguishable from each other. Hinge and Bradley--Terry should
therefore be treated as practically similar overall.

At `N=1`, rater-level Spearman ranges from 0.208 to 0.413 within rater and
from 0.011 to 0.391 across raters. Within-rater prediction remains stronger,
and cross-rater Abstract Liking is near zero.

The released raw survey export has seven finished entries, including one
preview, leaving six completed non-preview responses. The timing filter
removes one constant-response rater and retains five. The manuscript's
agreement panel uses R1, R3, R4, R5, and R6; the repository columns are P1,
P3, P4, P5, and P6. Human-to-human agreement averages ten unique pairs and
excludes GT: 0.725 for direct ratings and 0.520 for comparisons. Timing is
about 10.71 seconds for comparisons versus 27.28 seconds for direct ratings
when reproducing the paper's averaging of rounded condition means. Full
precision timing values are also exported.

## Data

The Sidhu dataset contains 240 abstract and 240 representational paintings.
The controlled manifest aligns 477 available images and ratings by explicit
item ID. APDDv2 contributes 10,022 matched images and 11 aggregate aesthetic
targets. APDDv2 does not provide individual-rater labels, so within-rater and
cross-rater analyses are limited to Sidhu. See `Data/README.md` and
`docs/art_extension_experiments.md` for provenance details.

## License

This project is maintained by the HiL-SE Lab at Rochester Institute of
Technology.
