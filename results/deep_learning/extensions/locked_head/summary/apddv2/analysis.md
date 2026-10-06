# APDDv2 extension results

## Validation and analysis unit

All 22 result files passed SHA-256 verification against their metadata. Each file contains 210 conditions (10 regression, 100 hinge, and 100 Bradley-Terry), for 4,620 fitted models total. Comparisons are paired on target, seed, split, and N where applicable. Confidence intervals are nonparametric 95% bootstrap intervals over the 11 target-level mean differences (20,000 resamples); seeds and N values are repeated measures, not independent samples.

## CLIP versus ResNet-50

Positive differences favor CLIP for the correlation and pair-accuracy metrics. Pairwise rows average N=1 through N=10 before targets are compared.

| Objective | Metric | ResNet | CLIP | CLIP − ResNet (95% CI) | CLIP wins |
|---|---:|---:|---:|---:|---:|
| Regression | spearman | 0.701 | 0.775 | 0.074 [0.068, 0.079] | 11/11 |
| Regression | pair accuracy | 0.762 | 0.798 | 0.036 [0.033, 0.038] | 11/11 |
| Hinge | spearman | 0.685 | 0.761 | 0.076 [0.071, 0.080] | 11/11 |
| Hinge | pair accuracy | 0.755 | 0.790 | 0.035 [0.033, 0.037] | 11/11 |
| Bradley Terry | spearman | 0.686 | 0.762 | 0.076 [0.071, 0.081] | 11/11 |
| Bradley Terry | pair accuracy | 0.755 | 0.790 | 0.035 [0.033, 0.037] | 11/11 |

Regression Spearman by target:

| Target | ResNet | CLIP | Difference |
|---|---:|---:|---:|
| Creativity | 0.623 | 0.709 | 0.086 |
| The sense of order | 0.678 | 0.763 | 0.084 |
| The overall | 0.704 | 0.786 | 0.081 |
| Details and texture | 0.742 | 0.819 | 0.077 |
| Theme and logic | 0.672 | 0.748 | 0.076 |
| Space and perspective | 0.714 | 0.789 | 0.075 |
| Total aesthetic score | 0.737 | 0.812 | 0.075 |
| Layout and composition | 0.704 | 0.776 | 0.072 |
| Light and shadow | 0.741 | 0.811 | 0.070 |
| Mood | 0.677 | 0.744 | 0.067 |
| Color | 0.714 | 0.766 | 0.052 |

## Bradley-Terry versus hinge

Positive differences favor Bradley-Terry. Results average N=1 through N=10 within each target before target-level comparison.

| Representation | Metric | Hinge | Bradley-Terry | BT − hinge (95% CI) | BT wins |
|---|---:|---:|---:|---:|---:|
| resnet50 | spearman | 0.685 | 0.686 | 0.001 [-0.000, 0.002] | 7/11 |
| resnet50 | pair accuracy | 0.755 | 0.755 | 0.000 [0.000, 0.001] | 8/11 |
| clip-vit-b32 | spearman | 0.761 | 0.762 | 0.001 [0.001, 0.002] | 11/11 |
| clip-vit-b32 | pair accuracy | 0.790 | 0.790 | 0.001 [0.001, 0.001] | 11/11 |

## Pairwise training versus regression

N=10 is shown because it is the largest comparison budget tested. Positive differences favor pairwise training.

| Representation | Pairwise loss | Metric | Regression | Pairwise N=10 | Difference (95% CI) | Pairwise wins |
|---|---|---:|---:|---:|---:|---:|
| resnet50 | hinge | spearman | 0.701 | 0.696 | -0.005 [-0.008, -0.003] | 2/11 |
| resnet50 | hinge | pair accuracy | 0.762 | 0.760 | -0.002 [-0.003, -0.001] | 1/11 |
| resnet50 | bradley terry | spearman | 0.701 | 0.697 | -0.004 [-0.007, -0.001] | 3/11 |
| resnet50 | bradley terry | pair accuracy | 0.762 | 0.761 | -0.001 [-0.003, 0.000] | 3/11 |
| clip-vit-b32 | hinge | spearman | 0.775 | 0.764 | -0.011 [-0.013, -0.009] | 0/11 |
| clip-vit-b32 | hinge | pair accuracy | 0.798 | 0.792 | -0.006 [-0.007, -0.004] | 1/11 |
| clip-vit-b32 | bradley terry | spearman | 0.775 | 0.764 | -0.011 [-0.014, -0.009] | 0/11 |
| clip-vit-b32 | bradley terry | pair accuracy | 0.798 | 0.792 | -0.006 [-0.007, -0.004] | 0/11 |

## Effect of comparisons per item

These are descriptive grand means over targets and seeds.

| Representation | Objective | Spearman N=1 → N=10 | Pair accuracy N=1 → N=10 |
|---|---|---:|---:|
| resnet50 | hinge | 0.660 → 0.696 | 0.741 → 0.760 |
| resnet50 | bradley terry | 0.659 → 0.697 | 0.741 → 0.761 |
| clip-vit-b32 | hinge | 0.750 → 0.764 | 0.783 → 0.792 |
| clip-vit-b32 | bradley terry | 0.751 → 0.764 | 0.784 → 0.792 |

## Interpretation

The primary evidence is the paired target-level difference, not the raw count of 4,620 fitted models. Intervals crossing zero indicate that the direction is not consistent enough across the 11 APDDv2 attributes to claim a general advantage. MAE should not be averaged as a cross-target headline because APDDv2 attributes use different numeric scales; rank correlation and pair accuracy are comparable across targets.
