# Corrected Sidhu extension results

## Validation and analysis unit

All 16 source files passed SHA-256 verification. The analysis uses the 80 unaffected regression fits from the first run and the 1,600 corrected pairwise fits from the v2 rerun. The 1,600 pairwise rows from the first run are excluded because the Keras label/prediction rank mismatch caused cross-pair broadcasting. Comparisons are paired on condition, seed, split, and N. Confidence intervals bootstrap the four condition-level mean differences (20,000 resamples).

## CLIP versus ResNet-50

| Objective | Metric | ResNet | CLIP | CLIP − ResNet (95% CI) | CLIP wins |
|---|---:|---:|---:|---:|---:|
| Regression | spearman | 0.205 | 0.371 | 0.166 [0.106, 0.227] | 4/4 |
| Regression | pair accuracy | 0.566 | 0.628 | 0.063 [0.037, 0.090] | 4/4 |
| Hinge | spearman | 0.348 | 0.530 | 0.182 [0.160, 0.202] | 4/4 |
| Hinge | pair accuracy | 0.621 | 0.689 | 0.068 [0.058, 0.077] | 4/4 |
| Bradley Terry | spearman | 0.344 | 0.526 | 0.182 [0.161, 0.203] | 4/4 |
| Bradley Terry | pair accuracy | 0.620 | 0.687 | 0.067 [0.057, 0.074] | 4/4 |

Regression Spearman by condition:

| Condition | ResNet | CLIP | Difference |
|---|---:|---:|---:|
| Representational Beauty | 0.200 | 0.457 | 0.257 |
| Abstract Liking | 0.143 | 0.340 | 0.197 |
| Abstract Beauty | 0.194 | 0.323 | 0.128 |
| Representational Liking | 0.281 | 0.365 | 0.083 |

## Bradley-Terry versus hinge

| Representation | Metric | Hinge | Bradley-Terry | BT − hinge (95% CI) | BT wins |
|---|---:|---:|---:|---:|---:|
| resnet50 | spearman | 0.348 | 0.344 | -0.004 [-0.014, 0.005] | 1/4 |
| resnet50 | pair accuracy | 0.621 | 0.620 | -0.001 [-0.005, 0.002] | 2/4 |
| clip-vit-b32 | spearman | 0.530 | 0.526 | -0.003 [-0.006, -0.001] | 0/4 |
| clip-vit-b32 | pair accuracy | 0.689 | 0.687 | -0.002 [-0.003, -0.001] | 0/4 |

## Pairwise training versus regression at N=10

| Representation | Pairwise loss | Metric | Regression | Pairwise | Difference (95% CI) | Pairwise wins |
|---|---|---:|---:|---:|---:|---:|
| resnet50 | hinge | spearman | 0.205 | 0.377 | 0.173 [0.146, 0.204] | 4/4 |
| resnet50 | hinge | pair accuracy | 0.566 | 0.636 | 0.070 [0.060, 0.089] | 4/4 |
| resnet50 | bradley terry | spearman | 0.205 | 0.386 | 0.181 [0.149, 0.222] | 4/4 |
| resnet50 | bradley terry | pair accuracy | 0.566 | 0.635 | 0.069 [0.056, 0.086] | 4/4 |
| clip-vit-b32 | hinge | spearman | 0.371 | 0.561 | 0.190 [0.153, 0.225] | 4/4 |
| clip-vit-b32 | hinge | pair accuracy | 0.628 | 0.704 | 0.076 [0.060, 0.094] | 4/4 |
| clip-vit-b32 | bradley terry | spearman | 0.371 | 0.552 | 0.181 [0.149, 0.208] | 4/4 |
| clip-vit-b32 | bradley terry | pair accuracy | 0.628 | 0.699 | 0.071 [0.054, 0.088] | 4/4 |

## Effect of comparisons per item

| Representation | Objective | Spearman N=1 → N=10 | Pair accuracy N=1 → N=10 |
|---|---|---:|---:|
| resnet50 | hinge | 0.268 → 0.377 | 0.594 → 0.636 |
| resnet50 | bradley terry | 0.260 → 0.386 | 0.591 → 0.635 |
| clip-vit-b32 | hinge | 0.459 → 0.561 | 0.659 → 0.704 |
| clip-vit-b32 | bradley terry | 0.457 → 0.552 | 0.660 → 0.699 |
