# Locked-head art-extension results

## Integrity

The final locked-head package contains 78 source result CSVs with 14,700
model-fit rows: Sidhu aggregate runs, APDDv2 aggregate runs, and Sidhu
rater-level runs. All source result hashes, row counts, and condition keys
match their metadata. Summary CSVs and Markdown files are generated from
these validated sources.

## Representation comparison

| Dataset | Objective | ResNet-50 Spearman | CLIP Spearman | Difference |
|---|---|---:|---:|---:|
| Sidhu | Regression | 0.534 | 0.732 | +0.198 |
| Sidhu | Hinge, all N | 0.620 | 0.752 | +0.132 |
| Sidhu | Bradley--Terry, all N | 0.621 | 0.756 | +0.135 |
| APDDv2 | Regression | 0.701 | 0.775 | +0.074 |
| APDDv2 | Hinge, all N | 0.685 | 0.761 | +0.076 |
| APDDv2 | Bradley--Terry, all N | 0.686 | 0.762 | +0.076 |

CLIP is higher for every Sidhu condition and every APDDv2 target under the
regression comparison.

## Comparative learning at N=10

| Dataset | Regression | Hinge | Bradley--Terry |
|---|---:|---:|---:|
| Sidhu | 0.732 | 0.767 | 0.772 |
| APDDv2 | 0.775 | 0.764 | 0.764 |

For Sidhu, paired two-sided Wilcoxon tests across ten matched seeds give
Holm-adjusted `p=0.0117` for hinge versus regression, `p=0.0059` for
Bradley--Terry versus regression, and `p=0.1055` for Bradley--Terry versus
hinge. For APDDv2 the corresponding values are `0.0059`, `0.0059`, and
`0.9219`. Pairwise learning helps modestly on Sidhu but is slightly lower on
APDDv2, whose preferences are derived from aggregate scores.

Across the full budget sweep, objective differences are small. Sidhu CLIP
mean Spearman is 0.752 for hinge and 0.756 for Bradley--Terry; APDDv2 CLIP is
0.761 and 0.762. Neither objective is uniformly preferable.

## Rater-level results at N=1

| Task | Objective | Within Spearman | Cross Spearman |
|---|---|---:|---:|
| Abstract Beauty | Regression | 0.386 | 0.343 |
| Abstract Beauty | Hinge | 0.378 | 0.327 |
| Abstract Beauty | Bradley--Terry | 0.387 | 0.328 |
| Abstract Liking | Regression | 0.220 | 0.014 |
| Abstract Liking | Hinge | 0.208 | 0.011 |
| Abstract Liking | Bradley--Terry | 0.215 | 0.017 |
| Representational Beauty | Regression | 0.411 | 0.404 |
| Representational Beauty | Hinge | 0.413 | 0.388 |
| Representational Beauty | Bradley--Terry | 0.412 | 0.391 |
| Representational Liking | Regression | 0.208 | 0.196 |
| Representational Liking | Hinge | 0.223 | 0.196 |
| Representational Liking | Bradley--Terry | 0.217 | 0.200 |

Within-rater performance exceeds cross-rater performance for every task and
objective. Cross-rater Abstract Liking remains near zero.

## Human study

Seven participants completed the survey; five remained after excluding two
participants with insufficient response variance. Mean time per item was
27.28 seconds for direct ratings and 10.71 seconds for comparative judgments,
an approximately 60% reduction. This result pertains to the retained sample.
