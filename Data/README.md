# Data

This directory contains two categories of data used in the paper.

## Original Data (Ours)

| File | Description |
|------|-------------|
| `RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv` | Raw Qualtrics export from our human subjective experiment (RQ4). 6 respondents rated paintings using both direct rating and comparative judgment methods. |

Analysis code for this data is in `../human_survey/`.

## External Data (Sidhu et al., 2018)

The painting images and aesthetic ratings below are from:

> Sidhu, D. M., McDougall, K. H., Jalava, S. T., & Bodner, G. E. (2018). Prediction of beauty and liking ratings for abstract and representational paintings using subjective and objective measures. *PLOS ONE*, 13(7), 1-15. https://doi.org/10.1371/journal.pone.0200431

Data source: https://osf.io/2sy4f/

| File/Directory | Description |
|----------------|-------------|
| `Abstract_Images/` | 240 abstract paintings |
| `Representational_Images/` | 240 representational paintings |
| `PaintingDataMeans.csv` | Mean beauty and liking ratings across all raters |
| `Abstract_All_Raters.csv` | Per-rater beauty ratings for abstract paintings |
| `Abstract_Liking_All_Raters.csv` | Per-rater liking ratings for abstract paintings |
| `Representational_All_Raters.csv` | Per-rater beauty ratings for representational paintings |
| `Representational_Liking_All_Raters.csv` | Per-rater liking ratings for representational paintings |
| `Abstract_Data.csv` | Objective features for abstract paintings |
| `Representational_Data.csv` | Objective features for representational paintings |
