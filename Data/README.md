# Data

This directory contains two categories of data used in the paper.

## Original Data (Ours)

| File | Description |
|------|-------------|
| `RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv` | Raw Qualtrics export from the human study (RQ4). Seven participants completed the survey; five were retained after excluding two for insufficient response variance. |

Analysis code for these data is in `../code/human_survey/`.

## External Data (Sidhu et al., 2018)

The painting images and aesthetic ratings below are from:

> Sidhu, D. M., McDougall, K. H., Jalava, S. T., & Bodner, G. E. (2018). Prediction of beauty and liking ratings for abstract and representational paintings using subjective and objective measures. *PLOS ONE*, 13(7), 1-15. https://doi.org/10.1371/journal.pone.0200431

Data source: https://osf.io/2sy4f/

| File/Directory | Description |
|----------------|-------------|
| `Abstract_Images/` | 239 available abstract paintings; original ID 173 is absent |
| `Representational_Images/` | 238 available representational paintings; original IDs 90 and 157 are absent |
| `PaintingDataMeans.csv` | Mean beauty and liking ratings across all raters |
| `Abstract_All_Raters.csv` | Per-rater beauty ratings for abstract paintings |
| `Abstract_Liking_All_Raters.csv` | Per-rater liking ratings for abstract paintings |
| `Representational_All_Raters.csv` | Per-rater beauty ratings for representational paintings |
| `Representational_Liking_All_Raters.csv` | Per-rater liking ratings for representational paintings |
| `Abstract_Data.csv` | Objective features for abstract paintings |
| `Representational_Data.csv` | Objective features for representational paintings |

Beauty and liking use the integer scale 1--9. The four raw rater tables
contain 43--49 raters depending on the condition. Controlled extension
experiments build aggregate targets directly from these raw tables and retain
the original painting numbers when joining ratings to the 477 available
images. This avoids the row shift that occurs if compacted positions after the
three missing images are treated as original painting IDs.
