"""
Baseline OLS regression model for predicting average beauty/liking ratings.

This script implements the baseline model described in the IEEE Access paper
"Modeling Art Evaluations from Comparative Judgments." It fits an Ordinary
Least Squares (OLS) regression using 11 handcrafted objective image features
from Sidhu et al. (2018) to predict mean beauty and liking ratings for both
abstract and representational paintings.

The script produces the baseline values reported in Table 1 of the paper.
No train/test split is used here; the model is fit on the full dataset and
evaluated in-sample to establish upper-bound performance metrics (adj. R²,
R², Pearson r, Spearman rho).

Output:
    results/baseline/average_ratings/average_ratings.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import os

# The 11 objective image features from Sidhu et al. (2018), used as predictors
# in all baseline models throughout the paper.
objective_predictors = [
    "HueSD", "Saturation", "SaturationSD", "Brightness", "BrightnessSD",
    "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
    "Vertical_Symmetry", "Horizontal_Symmetry", "ColourComponent"
]

# Each case is a (data_path, representational_filter, rating_column, label) tuple.
# rep_filter=0 selects abstract paintings; rep_filter=1 selects representational.
# This covers all four conditions in Table 1: Abstract/Representational x Beauty/Liking.
cases = [
    ("../../Data/Abstract_Data.csv", 0, "BeautyRating_M", "Abstract_Beauty"),
    ("../../Data/Abstract_Data.csv", 0, "Liking_M", "Abstract_Liking"),
    ("../../Data/Representational_Data.csv", 1, "BeautyRating_M", "Representational_Beauty"),
    ("../../Data/Representational_Data.csv", 1, "Liking_M", "Representational_Liking")
]

all_results = []

# Run each case (DRY - no code repetition)
for data_path, rep_filter, rating_var, case_name in cases:
    print(f"\n--- {case_name} ---")

    # Load and clean data (as in the paper; no standardization)
    data = pd.read_csv(data_path)
    data = data[data["Representational"] == rep_filter].copy()
    # Replace Excel null markers and infinities with NaN for clean numeric handling
    data.replace(["#NULL!", np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Force numeric conversion: Abstract_Data.csv may contain string-typed
    # numeric columns due to "#NULL!" entries; coerce them to float.
    for col in objective_predictors + [rating_var]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(inplace=True)  # Drop any new NaNs from conversion

    X = data[objective_predictors]
    y = data[rating_var]

    # Fit OLS with all 11 features (no stepwise selection) plus an intercept
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # Collect in-sample performance metrics for Table 1
    results = {
        "Case": case_name,
        "adj_r2": model.rsquared_adj,
        "r2": model.rsquared,
        "pearson_r": pearsonr(y, model.fittedvalues)[0],
        "spearman_rho": spearmanr(y, model.fittedvalues)[0],
    }
    all_results.append(results)

    print(f"✓ adj R² = {model.rsquared_adj:.3f}, N = {len(y)}")

# Save all results in one CSV
OUT_DIR = "../../results/baseline/average_ratings"
os.makedirs(OUT_DIR, exist_ok=True)
results_df = pd.DataFrame(all_results)
out_path = os.path.join(OUT_DIR, "average_ratings.csv")
results_df.to_csv(out_path, index=False)

print("\n" + "="*60)
print("ALL RESULTS SAVED:")
print(results_df.to_string(index=False))
print(f"\n✓ Saved to: {out_path}")
