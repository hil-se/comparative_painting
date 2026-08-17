# Controlled art-extension protocol

The final extension compares ResNet-50 with CLIP ViT-B/32, hinge ranking with
Bradley--Terry loss, and the Sidhu conditions with all 11 APDDv2 targets.

## Locked downstream method

Twenty-two candidate configurations were screened and confirmed using APDDv2
validation macro Spearman. The selected method,
`shallow-mse-z-gelu-ln-rawclip`, was locked before test evaluation and reused
without retuning across datasets, representations, objectives, targets,
raters, and seeds.

- Architecture: `256 -> 64 -> 1`.
- Activation and normalization: GELU and LayerNorm.
- Regularization: dropout 0.1 and L2 `1e-5` on the first hidden layer.
- Fixed visual features: no additional feature standardization.
- Regression targets: standardized from the training partition; predictions
  are transformed back to the original target scale.
- Regression loss: MSE.
- Pairwise losses: hinge and Bradley--Terry.
- Pairwise scores: affine-calibrated on validation data only for MAE and
  R-squared; pair accuracy and rank correlations use raw scores.
- Optimization: Adam at `1e-3`, batch size 128, maximum 200 epochs.
- Model selection during training: validation Spearman, minimum 25 epochs,
  patience 20, best-weight restoration, and learning-rate halving after 10
  plateaus to a minimum of `1e-6`.

## Data and splits

- Sidhu: 140 training examples, 20 validation examples, and the remainder for
  testing for each of the four category/target conditions.
- APDDv2: deterministic 70/15/15 train/validation/test splits for all 11
  aggregate targets.
- Pair budget: `N = M / n_train`; pairwise runs use `N=1` through `N=10`.
- Repetitions: ten matched seeds.

The corrected Sidhu manifest joins ratings, images, and features by explicit
item ID. It accounts for the three paintings omitted from the released
feature arrays and retains 477 aligned items.

The official APDDv2 archive has one missing annotated image and six
byte-identical duplicate files. The controlled manifest records the missing
item, ignores unreferenced duplicates, and retains 10,022 matched images.
APDDv2 publishes aggregate scores only, so rater-level experiments cannot be
constructed for that dataset.

## Execution and integrity

The authoritative runners are `run_art_locked_head.py` and
`run_sidhu_rater_locked_head.py`. Each result has a metadata sidecar with the
locked configuration, source hashes, result hash, split, targets, objectives,
budgets, and seeds. `validate_art_result.py` verifies the row keys and SHA-256
digests before merged summaries are produced.

Cluster jobs use the `loop` account on SPORC's `onboard` partition. Generated
images, feature arrays, and intermediate checkpoints remain outside Git;
final result CSVs, metadata, and summaries are under
`results/extensions/locked_head/`.
