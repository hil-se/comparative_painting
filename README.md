# Modeling Art Evaluations from Comparative Judgments

Replication package for *Modeling Art Evaluations from Comparative
Judgments: A Deep Learning Approach to Predicting Aesthetic Preferences*.

Authors: Manoj Reddy Bethi, Xiaoyin Xi, Sai Rupa Jhade, Pravallika Yaganti,
Monoshiz Mahbub Khan, and Zhe Yu.

## Controlled study

The final experiments compare fixed ResNet-50 and CLIP ViT-B/32 visual
representations using one downstream method selected on APDDv2 validation
data and then locked for every dataset, target, objective, rater, and seed.

- Datasets: four Sidhu conditions and all 11 APDDv2 targets.
- Splits: Sidhu uses 140 training images, 20 validation images, and the
  remainder for testing; APDDv2 uses deterministic 70/15/15 splits.
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

Prepared manifests and fixed-feature files are required. For example:

```bash
python code/extensions/run_art_locked_head.py \
  --manifest /path/to/manifests/sidhu.csv \
  --features /path/to/features/sidhu-clip-vit-b32.npz \
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

The human study retained five raters after seven completed responses; two
were excluded for insufficient response variance. Comparative judgments took
about 10.71 seconds per item versus 27.28 seconds for direct ratings, a 60%
average reduction.

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
