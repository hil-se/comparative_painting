# Sidhu CLIP rater-extension analysis

## Integrity

- The result set contains 40 CSV files and 40 metadata files.
- Each CSV contains 210 unique objective/N/seed rows: 10 regression, 100 hinge, and 100 Bradley-Terry fits.
- All 8,400 rows match the SHA-256 digests recorded in their metadata.
- The design covers four tasks, within-rater and cross-rater protocols, five target raters, N=1 through N=10, and ten matched seeds.

## N=1 results

Values pool five target raters and ten seeds (50 evaluations per entry).

| Task | Objective | Within Pearson | Within Spearman | Cross Pearson | Cross Spearman |
|---|---|---:|---:|---:|---:|
| Abstract Beauty | Regression | 0.408 | 0.386 | 0.385 | 0.343 |
| Abstract Beauty | Hinge | 0.398 | 0.378 | 0.362 | 0.327 |
| Abstract Beauty | Bradley Terry | 0.406 | 0.387 | 0.356 | 0.328 |
| Abstract Liking | Regression | 0.226 | 0.220 | 0.029 | 0.014 |
| Abstract Liking | Hinge | 0.215 | 0.208 | 0.020 | 0.011 |
| Abstract Liking | Bradley Terry | 0.224 | 0.215 | 0.021 | 0.017 |
| Representational Beauty | Regression | 0.423 | 0.411 | 0.406 | 0.404 |
| Representational Beauty | Hinge | 0.424 | 0.413 | 0.397 | 0.388 |
| Representational Beauty | Bradley Terry | 0.425 | 0.412 | 0.398 | 0.391 |
| Representational Liking | Regression | 0.212 | 0.208 | 0.200 | 0.196 |
| Representational Liking | Hinge | 0.220 | 0.223 | 0.201 | 0.196 |
| Representational Liking | Bradley Terry | 0.218 | 0.217 | 0.203 | 0.200 |
