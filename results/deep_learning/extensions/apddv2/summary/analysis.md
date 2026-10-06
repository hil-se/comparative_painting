# APDDv2 extension results

## Validation and analysis unit

All 22 result files passed SHA-256 verification against their metadata. Each file contains 210 conditions (10 regression, 100 hinge, and 100 Bradley-Terry), for 4,620 fitted models total. Comparisons are paired on target, seed, split, and N where applicable. Confidence intervals are nonparametric 95% bootstrap intervals over the 11 target-level mean differences (20,000 resamples); seeds and N values are repeated measures, not independent samples.

## CLIP versus ResNet-50

Positive differences favor CLIP for the correlation and pair-accuracy metrics. Pairwise rows average N=1 through N=10 before targets are compared.

| Objective | Metric | ResNet | CLIP | CLIP − ResNet (95% CI) | CLIP wins |
|---|---:|---:|---:|---:|---:|
| Regression | spearman | 0.685 | 0.766 | 0.081 [0.074, 0.087] | 11/11 |
| Regression | pair accuracy | 0.755 | 0.794 | 0.039 [0.035, 0.041] | 11/11 |
| Hinge | spearman | 0.671 | 0.758 | 0.087 [0.081, 0.093] | 11/11 |
| Hinge | pair accuracy | 0.748 | 0.788 | 0.041 [0.038, 0.043] | 11/11 |
| Bradley Terry | spearman | 0.673 | 0.762 | 0.090 [0.084, 0.096] | 11/11 |
| Bradley Terry | pair accuracy | 0.748 | 0.790 | 0.042 [0.039, 0.044] | 11/11 |

Regression Spearman by target:

| Target | ResNet | CLIP | Difference |
|---|---:|---:|---:|
| The sense of order | 0.658 | 0.752 | 0.094 |
| The overall | 0.686 | 0.778 | 0.092 |
| Creativity | 0.616 | 0.708 | 0.092 |
| Details and texture | 0.728 | 0.815 | 0.087 |
| Theme and logic | 0.647 | 0.732 | 0.085 |
| Layout and composition | 0.685 | 0.769 | 0.084 |
| Total aesthetic score | 0.724 | 0.807 | 0.082 |
| Mood | 0.664 | 0.740 | 0.076 |
| Light and shadow | 0.724 | 0.799 | 0.075 |
| Space and perspective | 0.706 | 0.780 | 0.074 |
| Color | 0.696 | 0.750 | 0.053 |

## Bradley-Terry versus hinge

Positive differences favor Bradley-Terry. Results average N=1 through N=10 within each target before target-level comparison.

| Representation | Metric | Hinge | Bradley-Terry | BT − hinge (95% CI) | BT wins |
|---|---:|---:|---:|---:|---:|
| resnet50 | spearman | 0.671 | 0.673 | 0.001 [0.000, 0.002] | 9/11 |
| resnet50 | pair accuracy | 0.748 | 0.748 | 0.001 [0.000, 0.001] | 9/11 |
| clip-vit-b32 | spearman | 0.758 | 0.762 | 0.004 [0.003, 0.005] | 11/11 |
| clip-vit-b32 | pair accuracy | 0.788 | 0.790 | 0.002 [0.002, 0.002] | 11/11 |

## Pairwise training versus regression

N=10 is shown because it is the largest comparison budget tested. Positive differences favor pairwise training.

| Representation | Pairwise loss | Metric | Regression | Pairwise N=10 | Difference (95% CI) | Pairwise wins |
|---|---|---:|---:|---:|---:|---:|
| resnet50 | hinge | spearman | 0.685 | 0.685 | -0.000 [-0.005, 0.003] | 6/11 |
| resnet50 | hinge | pair accuracy | 0.755 | 0.754 | -0.001 [-0.003, 0.001] | 5/11 |
| resnet50 | bradley terry | spearman | 0.685 | 0.690 | 0.005 [0.002, 0.008] | 9/11 |
| resnet50 | bradley terry | pair accuracy | 0.755 | 0.757 | 0.002 [0.000, 0.003] | 10/11 |
| clip-vit-b32 | hinge | spearman | 0.766 | 0.767 | 0.000 [-0.005, 0.005] | 7/11 |
| clip-vit-b32 | hinge | pair accuracy | 0.794 | 0.793 | -0.000 [-0.003, 0.002] | 4/11 |
| clip-vit-b32 | bradley terry | spearman | 0.766 | 0.769 | 0.003 [-0.001, 0.007] | 9/11 |
| clip-vit-b32 | bradley terry | pair accuracy | 0.794 | 0.795 | 0.001 [-0.001, 0.004] | 6/11 |

## Effect of comparisons per item

These are descriptive grand means over targets and seeds.

| Representation | Objective | Spearman N=1 → N=10 | Pair accuracy N=1 → N=10 |
|---|---|---:|---:|
| resnet50 | hinge | 0.625 → 0.685 | 0.727 → 0.754 |
| resnet50 | bradley terry | 0.623 → 0.690 | 0.726 → 0.757 |
| clip-vit-b32 | hinge | 0.735 → 0.767 | 0.776 → 0.793 |
| clip-vit-b32 | bradley terry | 0.744 → 0.769 | 0.781 → 0.795 |

## Interpretation

The primary evidence is the paired target-level difference, not the raw count of 4,620 fitted models. Intervals crossing zero indicate that the direction is not consistent enough across the 11 APDDv2 attributes to claim a general advantage. MAE should not be averaged as a cross-target headline because APDDv2 attributes use different numeric scales; rank correlation and pair accuracy are comparable across targets.
