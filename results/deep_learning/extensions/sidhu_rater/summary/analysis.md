# Sidhu CLIP rater-extension analysis

## Integrity

- TIGRIS job 37786 completed all 40 array tasks with exit code 0.
- The result set contains 40 CSV files and 40 metadata files.
- Each CSV contains 210 unique objective/N/seed rows: 10 regression, 100
  hinge, and 100 Bradley-Terry fits.
- All 8,400 rows match the SHA-256 digests and row counts recorded in their
  metadata.
- The design covers four tasks, within-rater and cross-rater protocols, five
  target raters, N=1 through N=10, and ten matched seeds.

## N=10 results

Values pool five target raters and ten seeds (50 evaluations per entry).

| Task | Objective | Within Pearson | Within Spearman | Cross Pearson | Cross Spearman |
|---|---|---:|---:|---:|---:|
| Abstract Beauty | Regression | 0.370 | 0.345 | 0.335 | 0.297 |
| Abstract Beauty | Hinge | 0.433 | 0.410 | 0.395 | 0.355 |
| Abstract Beauty | Bradley-Terry | 0.439 | 0.411 | 0.392 | 0.350 |
| Abstract Liking | Regression | 0.244 | 0.240 | 0.058 | 0.044 |
| Abstract Liking | Hinge | 0.273 | 0.263 | 0.025 | 0.012 |
| Abstract Liking | Bradley-Terry | 0.267 | 0.261 | 0.029 | 0.015 |
| Representational Beauty | Regression | 0.366 | 0.349 | 0.354 | 0.339 |
| Representational Beauty | Hinge | 0.439 | 0.420 | 0.413 | 0.394 |
| Representational Beauty | Bradley-Terry | 0.444 | 0.426 | 0.410 | 0.391 |
| Representational Liking | Regression | 0.201 | 0.192 | 0.135 | 0.127 |
| Representational Liking | Hinge | 0.247 | 0.237 | 0.200 | 0.194 |
| Representational Liking | Bradley-Terry | 0.254 | 0.247 | 0.202 | 0.199 |

## Interpretation

- Within-rater performance is stronger than cross-rater performance in every
  task and objective.
- Pairwise training improves regression for Abstract Beauty,
  Representational Beauty, and Representational Liking.
- Cross-rater Abstract Liking is the exception: regression Spearman is 0.044,
  versus 0.012 for hinge and 0.015 for Bradley-Terry.
- Hinge and Bradley-Terry are practically equivalent at N=10; their Spearman
  values differ by at most 0.006.
- APDDv2 cannot support an analogous rater experiment because it exposes
  aggregate attribute scores rather than individual-rater labels.
