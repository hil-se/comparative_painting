# Data

## Original Human Survey

`RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv` is the raw
Qualtrics export. Seven finished entries include one preview. Six non-preview
responses become five after excluding the constant-response participant P2.
The agreement panel uses P1, P3, P4, P5, and P6.

Question-level ratings, comparisons, and ground-truth scores are in
`../results/human_survey/survey_data/`. The corresponding scripts are in
`../code/human_survey/`.

## Sidhu et al. (2018)

> Sidhu, D. M., McDougall, K. H., Jalava, S. T., & Bodner, G. E. (2018). Prediction of beauty and liking ratings for abstract and representational paintings using subjective and objective measures. *PLOS ONE*, 13(7), 1–15. https://doi.org/10.1371/journal.pone.0200431

Source: [OSF](https://osf.io/2sy4f/).

| File/Directory | Description |
|----------------|-------------|
| `Abstract_Images/` | 239 available images; painting 173 is absent |
| `Representational_Images/` | 238 available images; paintings 90 and 157 are absent |
| `*_All_Raters.csv` | Beauty or liking scores from the source raters |
| `Abstract_Data.csv`, `Representational_Data.csv` | The 11 objective predictors for OLS |
| `PaintingDataMeans.csv` | Source mean ratings |
| `fixed_features/sidhu-clip-vit-b32.npz` | Fixed CLIP embeddings for the 477 available images, with original item IDs |

Aggregate targets average all available source raters. Within- and cross-rater
experiments use raters 1–5. The original-size ResNet feature arrays are in
`../code/deep_learning/feature/`.

## APDDv2

Obtain annotations and images from the
[official repository](https://github.com/BestiVictory/APDDv2), under its
CC BY-NC-ND 4.0 terms. The paper uses 10,022 available images and 11 aggregate
aesthetic targets. See the [replication guide](../code/deep_learning/README.md)
for preparation and feature extraction.
