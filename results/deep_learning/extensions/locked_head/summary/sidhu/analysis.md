# Corrected Sidhu extension results

## Validation and analysis unit

All 16 source files passed SHA-256 verification. The analysis uses 80 regression fits and 1,600 pairwise fits generated from the corrected manifest. Comparisons are paired on condition, seed, split, and N. Confidence intervals bootstrap the four condition-level mean differences (20,000 resamples).

## CLIP versus ResNet-50

| Objective | Metric | ResNet | CLIP | CLIP − ResNet (95% CI) | CLIP wins |
|---|---:|---:|---:|---:|---:|
| Regression | spearman | 0.534 | 0.732 | 0.198 [0.166, 0.226] | 4/4 |
| Regression | pair accuracy | 0.692 | 0.776 | 0.084 [0.073, 0.095] | 4/4 |
| Hinge | spearman | 0.620 | 0.752 | 0.132 [0.117, 0.146] | 4/4 |
| Hinge | pair accuracy | 0.723 | 0.785 | 0.062 [0.060, 0.065] | 4/4 |
| Bradley Terry | spearman | 0.621 | 0.756 | 0.135 [0.121, 0.150] | 4/4 |
| Bradley Terry | pair accuracy | 0.724 | 0.788 | 0.064 [0.062, 0.065] | 4/4 |

Regression Spearman by condition:

| Condition | ResNet | CLIP | Difference |
|---|---:|---:|---:|
| Representational Liking | 0.561 | 0.793 | 0.232 |
| Abstract Liking | 0.409 | 0.630 | 0.221 |
| Representational Beauty | 0.563 | 0.754 | 0.191 |
| Abstract Beauty | 0.602 | 0.750 | 0.147 |

## Bradley-Terry versus hinge

| Representation | Metric | Hinge | Bradley-Terry | BT − hinge (95% CI) | BT wins |
|---|---:|---:|---:|---:|---:|
| resnet50 | spearman | 0.620 | 0.621 | 0.001 [-0.001, 0.003] | 3/4 |
| resnet50 | pair accuracy | 0.723 | 0.724 | 0.001 [-0.000, 0.002] | 3/4 |
| clip-vit-b32 | spearman | 0.752 | 0.756 | 0.005 [0.003, 0.007] | 4/4 |
| clip-vit-b32 | pair accuracy | 0.785 | 0.788 | 0.003 [0.002, 0.003] | 4/4 |

## Pairwise training versus regression at N=10

| Representation | Pairwise loss | Metric | Regression | Pairwise | Difference (95% CI) | Pairwise wins |
|---|---|---:|---:|---:|---:|---:|
| resnet50 | hinge | spearman | 0.534 | 0.642 | 0.108 [0.040, 0.153] | 4/4 |
| resnet50 | hinge | pair accuracy | 0.692 | 0.734 | 0.042 [0.021, 0.057] | 4/4 |
| resnet50 | bradley terry | spearman | 0.534 | 0.642 | 0.108 [0.042, 0.149] | 4/4 |
| resnet50 | bradley terry | pair accuracy | 0.692 | 0.735 | 0.043 [0.025, 0.056] | 4/4 |
| clip-vit-b32 | hinge | spearman | 0.732 | 0.767 | 0.035 [0.010, 0.056] | 3/4 |
| clip-vit-b32 | hinge | pair accuracy | 0.776 | 0.792 | 0.015 [0.004, 0.024] | 3/4 |
| clip-vit-b32 | bradley terry | spearman | 0.732 | 0.772 | 0.040 [0.018, 0.061] | 4/4 |
| clip-vit-b32 | bradley terry | pair accuracy | 0.776 | 0.795 | 0.018 [0.008, 0.028] | 4/4 |

## Effect of comparisons per item

| Representation | Objective | Spearman N=1 → N=10 | Pair accuracy N=1 → N=10 |
|---|---|---:|---:|
| resnet50 | hinge | 0.523 → 0.642 | 0.683 → 0.734 |
| resnet50 | bradley terry | 0.529 → 0.642 | 0.684 → 0.735 |
| clip-vit-b32 | hinge | 0.682 → 0.767 | 0.751 → 0.792 |
| clip-vit-b32 | bradley terry | 0.690 → 0.772 | 0.755 → 0.795 |
