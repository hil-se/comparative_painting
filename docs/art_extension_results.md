# Consolidated art-extension results

This note consolidates Dr. Yu's requested art-paper experiments on the
corrected Sidhu dataset and APDDv2. Detailed, machine-generated tables are in:

- `results/extensions/sidhu/summary/analysis.md`
- `results/extensions/apddv2/summary/analysis.md`

## Result integrity

- Sidhu: 80 valid regression fits from the initial run plus 1,600 corrected
  pairwise fits from the v2 rerun. The initial 1,600 pairwise fits are excluded
  because a Keras tensor-rank mismatch broadcast labels and predictions across
  unrelated pairs.
- APDDv2: 22 targets/representation files, 210 fits per file, and 4,620 fits
  total.
- Every included CSV matches the SHA-256 digest in its metadata.
- Comparisons are paired on the same target/condition, seed, split, and N.
- Confidence intervals bootstrap target/condition-level mean differences, so
  repeated seeds and N values are not treated as independent evidence.

## Main findings

### 1. CLIP consistently outperforms ResNet-50

CLIP improves both rank correlation and pair accuracy for every Sidhu
condition and every APDDv2 attribute.

| Dataset | Objective | Spearman: ResNet | Spearman: CLIP | Difference (95% CI) | CLIP wins |
|---|---|---:|---:|---:|---:|
| Sidhu | Regression | 0.205 | 0.371 | +0.166 [0.106, 0.227] | 4/4 |
| Sidhu | Hinge | 0.348 | 0.530 | +0.182 [0.160, 0.202] | 4/4 |
| Sidhu | Bradley-Terry | 0.344 | 0.526 | +0.182 [0.161, 0.203] | 4/4 |
| APDDv2 | Regression | 0.685 | 0.766 | +0.081 [0.074, 0.087] | 11/11 |
| APDDv2 | Hinge | 0.671 | 0.758 | +0.087 [0.081, 0.093] | 11/11 |
| APDDv2 | Bradley-Terry | 0.673 | 0.762 | +0.090 [0.084, 0.096] | 11/11 |

The representation result is the strongest and most stable new finding.

### 2. Bradley-Terry and hinge are practically equivalent

The loss comparison changes direction across datasets and its magnitude is
small:

| Dataset | Representation | Spearman: hinge | Spearman: BT | BT − hinge (95% CI) |
|---|---|---:|---:|---:|
| Sidhu | ResNet-50 | 0.348 | 0.344 | −0.004 [−0.014, 0.005] |
| Sidhu | CLIP | 0.530 | 0.526 | −0.003 [−0.006, −0.001] |
| APDDv2 | ResNet-50 | 0.671 | 0.673 | +0.001 [0.000, 0.002] |
| APDDv2 | CLIP | 0.758 | 0.762 | +0.004 [0.003, 0.005] |

This does not support a robust general advantage for either loss. A fair
summary is that Bradley-Terry and hinge reach nearly the same predictive
quality under the controlled protocol.

### 3. Pairwise training helps much more on Sidhu than APDDv2

At N=10, pairwise training substantially improves Sidhu performance relative
to regression. The gain appears for both representations and both losses; for
example, CLIP Spearman rises from 0.371 with regression to 0.561 with hinge
(+0.190) and 0.552 with Bradley-Terry (+0.181).

On APDDv2, N=10 pairwise training approximately matches regression. The
largest mean gain is ResNet-50 with Bradley-Terry: 0.685 to 0.690 Spearman
(+0.005). The CLIP Bradley-Terry change is 0.766 to 0.769 (+0.003), with its
target-level interval crossing zero.

This difference should be interpreted in context: APDDv2 comparisons are
derived from aggregate per-image scores rather than collected pairwise human
judgments, and its training sets are much larger than Sidhu's 140-image
training split.

### 4. More comparisons generally improve pairwise models

From N=1 to N=10, mean Spearman increases in all eight
dataset/representation/loss combinations. The increase is larger on Sidhu
(roughly +0.095 to +0.126) than APDDv2 (+0.025 to +0.067), consistent with
diminishing returns on the larger dataset.

## Suggested message to Dr. Yu

The experiments support a clear recommendation to use CLIP rather than the
released ResNet-50 features. Bradley-Terry is not consistently better than
hinge: it is marginally better on APDDv2 and marginally worse on Sidhu. The
pairwise-training advantage is strong on Sidhu but becomes parity on APDDv2,
where pairs are synthesized from aggregate scores. These distinctions should
be preserved instead of summarizing all three changes as uniformly positive.
