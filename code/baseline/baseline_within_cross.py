"""
Baseline OLS regression with within-rater and cross-rater train/test splits.

This script implements the baseline experiments described in the IEEE Access
paper "Modeling Art Evaluations from Comparative Judgments." It extends the
baseline OLS model (11 objective features from Sidhu et al. 2018) to three
experimental settings:

1. Average-rater: Train/test split on mean ratings (BeautyRating_M / Liking_M).
2. Within-rater: For each of 5 raters, train and test on that rater's own
   ratings using a 140/remaining split.
3. Cross-rater: For each target rater, train on the average of the other 4
   raters' ratings and test on the target rater's ratings.

Each experiment is repeated 10 times with different random splits to produce
stable estimates. Results are averaged across runs (and across raters for the
summary files). These values populate the "Baseline" rows of Tables 2-5 in
the paper.

Output directories:
    results/baseline/average_ratings/  -- average-rater per-case CSVs + summary
    results/baseline/within_rater/     -- within-rater per-case CSVs + summary
    results/baseline/cross_rater/      -- cross-rater per-case CSVs + summary
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import os
from multiprocessing import Pool, cpu_count


# The 11 objective image features from Sidhu et al. (2018), used as predictors
# in all baseline models throughout the paper.
objective_predictors = [
    "HueSD", "Saturation", "SaturationSD", "Brightness", "BrightnessSD",
    "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
    "Vertical_Symmetry", "Horizontal_Symmetry", "ColourComponent"
]


# Average rater cases: use pre-computed mean columns from the dataset.
# Tuple format: (data_path, representational_filter, rating_column, case_label)
avg_cases = [
    ("../../Data/Abstract_Data.csv", 0, "BeautyRating_M", "abstract_beauty"),
    ("../../Data/Abstract_Data.csv", 0, "Liking_M", "abstract_liking"),
    ("../../Data/Representational_Data.csv", 1, "BeautyRating_M", "representational_beauty"),
    ("../../Data/Representational_Data.csv", 1, "Liking_M", "representational_liking")
]


# Within/Cross rater cases: use individual rater response files.
# Tuple format: (rater_data_path, feature_data_path, rep_filter, rating_col, case_label)
rater_cases = [
    ("../../Data/Abstract_All_Raters.csv", "../../Data/Abstract_Data.csv", 0, "Beauty", "abstract_beauty"),
    ("../../Data/Abstract_Liking_All_Raters.csv", "../../Data/Abstract_Data.csv", 0, "Liking", "abstract_liking"),
    ("../../Data/Representational_All_Raters.csv", "../../Data/Representational_Data.csv", 1, "Beauty", "representational_beauty"),
    ("../../Data/Representational_Liking_All_Raters.csv", "../../Data/Representational_Data.csv", 1, "Liking", "representational_liking")
]


# The 5 individual raters used in within- and cross-rater experiments
target_raters = [1, 2, 3, 4, 5]
# Number of random train/test splits per experiment for stability
NUM_RUNS = 10

# Output directory structure
RESULTS_BASE = "../../results/baseline"
AVG_DIR = os.path.join(RESULTS_BASE, "average_ratings")
WITHIN_DIR = os.path.join(RESULTS_BASE, "within_rater")
CROSS_DIR = os.path.join(RESULTS_BASE, "cross_rater")
for d in [AVG_DIR, WITHIN_DIR, CROSS_DIR]:
    os.makedirs(d, exist_ok=True)



def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics comparing true and predicted ratings.

    These are the four metrics reported in Tables 2-5 of the paper for
    evaluating model performance on held-out test data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth ratings from the test set.
    y_pred : np.ndarray
        Model-predicted ratings for the test set.

    Returns
    -------
    tuple of (float, float, float, float)
        mae : Mean Absolute Error.
        r2 : Coefficient of determination (R²), computed manually to handle
             out-of-sample predictions (which can yield negative R²).
        rho : Pearson correlation coefficient between true and predicted.
        rs : Spearman rank correlation between true and predicted.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Guard against zero variance in test labels (would cause division by zero)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rho = pearsonr(y_true, y_pred)[0]
    rs = spearmanr(y_true, y_pred)[0]
    return mae, r2, rho, rs



# ==========================================================================
# AVERAGE-RATER: Single run (using BeautyRating_M / Liking_M columns)
# ==========================================================================
def run_average_single(args):
    """
    Execute one train/test split for the average-rater baseline experiment.

    Trains an OLS model on 140 paintings (or 80% if fewer than 150 available)
    using mean ratings, then evaluates on the remaining paintings.

    Parameters
    ----------
    args : tuple
        (run_id, data_path, rep_filter, rating_col, case_name) where:
        - run_id (int): Index of this run, used to seed the random split.
        - data_path (str): Path to the CSV containing features and mean ratings.
        - rep_filter (int): 0 for abstract, 1 for representational paintings.
        - rating_col (str): Column name for the target variable (e.g., "BeautyRating_M").
        - case_name (str): Label for this experimental condition.

    Returns
    -------
    dict or None
        Dictionary with keys "mae", "r2", "rho", "rs" if successful,
        or None if there is insufficient data.
    """
    run_id, data_path, rep_filter, rating_col, case_name = args

    # Load and clean data
    data = pd.read_csv(data_path)
    data = data[data["Representational"] == rep_filter].copy()
    data.replace(["#NULL!", np.inf, -np.inf], np.nan, inplace=True)

    # Force numeric conversion (handles string-typed columns from Excel nulls)
    for col in objective_predictors + [rating_col]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(subset=objective_predictors + [rating_col], inplace=True)

    # Skip if too few observations for a meaningful split
    if len(data) < 10:
        return None

    X = data[objective_predictors].to_numpy()
    y = data[rating_col].to_numpy()

    n = len(y)
    # Use 140 as training size (matching the paper's protocol) when enough
    # data is available; otherwise fall back to 80/20 split
    train_size = 140 if n >= 150 else int(0.8 * n)
    if train_size <= 1 or n - train_size <= 1:
        return None

    # Deterministic seed per run_id ensures reproducibility across runs
    rng = np.random.default_rng(seed=3000 + run_id)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:train_size], idx[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit OLS with intercept on training data
    X_train_c = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_c).fit()

    # Predict on held-out test set
    X_test_c = sm.add_constant(X_test)
    y_pred = model.predict(X_test_c)

    mae, r2, rho, rs = compute_metrics(y_test, y_pred)
    return {"mae": mae, "r2": r2, "rho": rho, "rs": rs}



# ==========================================================================
# WITHIN-RATER: Single run
# ==========================================================================
def run_within_single(args):
    """
    Execute one train/test split for the within-rater baseline experiment.

    For a given rater, trains an OLS model on that rater's own ratings for
    140 paintings and evaluates on the remaining paintings rated by the same
    rater. This tests how well objective features predict an individual's
    aesthetic judgments (Tables 2-3 in the paper).

    Parameters
    ----------
    args : tuple
        (run_id, rater_file, feature_file, rep_filter, rating_col, rater) where:
        - run_id (int): Index of this run, used to seed the random split.
        - rater_file (str): Path to CSV with individual rater responses (long format).
        - feature_file (str): Path to CSV with objective image features.
        - rep_filter (int): 0 for abstract, 1 for representational.
        - rating_col (str): Column name in rater_file (e.g., "Beauty" or "Liking").
        - rater (int): Rater ID (1-5).

    Returns
    -------
    dict or None
        Dictionary with keys "mae", "r2", "rho", "rs" if successful,
        or None if insufficient data.
    """
    run_id, rater_file, feature_file, rep_filter, rating_col, rater = args

    # Load objective features from the main dataset
    feat = pd.read_csv(feature_file)
    feat = feat[feat["Representational"] == rep_filter].copy()
    for c in objective_predictors:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    feat.dropna(subset=objective_predictors, inplace=True)
    # Strip .jpg extension so painting names match between feature and rater files
    feat["painting_clean"] = feat["Painting"].str.replace(".jpg", "", regex=False)


    # Load individual rater responses (long format: one row per rater-painting pair)
    ratings_long = pd.read_csv(rater_file)
    ratings_long["painting_clean"] = ratings_long["Painting"].str.replace(".jpg", "", regex=False)
    ratings_long = ratings_long[ratings_long["Rater"].isin(target_raters)]


    # Pivot to wide format: rows = paintings, columns = rater IDs, values = ratings
    pivot = ratings_long.pivot(index="painting_clean", columns="Rater", values=rating_col)
    merged = feat.merge(pivot, left_on="painting_clean", right_index=True, how="inner")


    # Extract this rater's data, dropping paintings they did not rate
    mask = merged[rater].notna()
    X = merged.loc[mask, objective_predictors].to_numpy()
    y = merged.loc[mask, rater].to_numpy()


    n = len(y)
    if n < 10:
        return None


    # 140/remaining split as described in the paper
    train_size = 140 if n >= 150 else int(0.8 * n)
    if train_size <= 1 or n - train_size <= 1:
        return None


    # Seed combines run_id and rater for unique-per-combination reproducibility
    rng = np.random.default_rng(seed=1000 * run_id + rater)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:train_size], idx[train_size:]


    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]


    X_train_c = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_c).fit()


    X_test_c = sm.add_constant(X_test)
    y_pred = model.predict(X_test_c)


    mae, r2, rho, rs = compute_metrics(y_test, y_pred)
    return {"mae": mae, "r2": r2, "rho": rho, "rs": rs}



# ==========================================================================
# CROSS-RATER: Single run
# ==========================================================================
def run_cross_single(args):
    """
    Execute one train/test split for the cross-rater baseline experiment.

    For a given target rater, trains an OLS model on the mean ratings of the
    other 4 raters and evaluates predictions against the target rater's own
    ratings. This tests how well a model trained on other people's preferences
    can predict a new individual's judgments (Tables 4-5 in the paper).

    Parameters
    ----------
    args : tuple
        (run_id, rater_file, feature_file, rep_filter, rating_col, rater) where:
        - run_id (int): Index of this run, used to seed the random split.
        - rater_file (str): Path to CSV with individual rater responses (long format).
        - feature_file (str): Path to CSV with objective image features.
        - rep_filter (int): 0 for abstract, 1 for representational.
        - rating_col (str): Column name in rater_file (e.g., "Beauty" or "Liking").
        - rater (int): Target rater ID (1-5) whose ratings are used for testing.

    Returns
    -------
    dict or None
        Dictionary with keys "mae", "r2", "rho", "rs" if successful,
        or None if insufficient data.
    """
    run_id, rater_file, feature_file, rep_filter, rating_col, rater = args

    # Load objective features
    feat = pd.read_csv(feature_file)
    feat = feat[feat["Representational"] == rep_filter].copy()
    for c in objective_predictors:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
    feat.dropna(subset=objective_predictors, inplace=True)
    feat["painting_clean"] = feat["Painting"].str.replace(".jpg", "", regex=False)


    # Load individual rater responses
    ratings_long = pd.read_csv(rater_file)
    ratings_long["painting_clean"] = ratings_long["Painting"].str.replace(".jpg", "", regex=False)
    ratings_long = ratings_long[ratings_long["Rater"].isin(target_raters)]


    # Pivot to wide format: rows = paintings, columns = rater IDs
    pivot = ratings_long.pivot(index="painting_clean", columns="Rater", values=rating_col)
    merged = feat.merge(pivot, left_on="painting_clean", right_index=True, how="inner")


    # Compute training labels: mean of the OTHER 4 raters (excluding target)
    others = [o for o in target_raters if o != rater]
    if len(others) == 0:
        return None


    y_other = merged[others].mean(axis=1)


    # Keep only paintings where both target rater and other-rater mean are available
    mask = merged[rater].notna() & y_other.notna()
    X = merged.loc[mask, objective_predictors].to_numpy()
    y_train_label = y_other.loc[mask].to_numpy()      # Train on others' mean
    y_test_label = merged.loc[mask, rater].to_numpy()  # Evaluate on target rater


    n = len(y_test_label)
    if n < 10:
        return None


    train_size = 140 if n >= 150 else int(0.8 * n)
    if train_size <= 1 or n - train_size <= 1:
        return None


    # Seed uses different offset (2000) to ensure cross-rater splits differ
    # from within-rater splits (which use 1000)
    rng = np.random.default_rng(seed=2000 * run_id + rater)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:train_size], idx[train_size:]


    # Train on other raters' mean, test against target rater's actual ratings
    X_train, y_train = X[train_idx], y_train_label[train_idx]
    X_test, y_test = X[test_idx], y_test_label[test_idx]


    X_train_c = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_c).fit()


    X_test_c = sm.add_constant(X_test)
    y_pred = model.predict(X_test_c)


    mae, r2, rho, rs = compute_metrics(y_test, y_pred)
    return {"mae": mae, "r2": r2, "rho": rho, "rs": rs}



# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    num_processes = min(cpu_count(), 8)
    print(f"Using {num_processes} processes")
    print(f"Running {NUM_RUNS} runs per rater/analysis\n")


    # ===== AVERAGE-RATER (using mean columns) =====
    print("="*70)
    print("AVERAGE-RATER EXPERIMENTS (10 runs)")
    print("="*70)

    for data_path, rep_filter, rating_col, case_name in avg_cases:
        print(f"\n[AVERAGE] {case_name}...")

        # Build argument list: one entry per run for parallel execution
        avg_task_args = [
            (run_id, data_path, rep_filter, rating_col, case_name)
            for run_id in range(NUM_RUNS)
        ]


        with Pool(processes=num_processes) as pool:
            avg_results = pool.map(run_average_single, avg_task_args)


        # Filter out None values (from runs with insufficient data) and average
        avg_results = [r for r in avg_results if r is not None]
        if avg_results:
            arr = np.array([[m["mae"], m["r2"], m["rho"], m["rs"]] for m in avg_results])
            mae, r2, rho, rs = arr.mean(axis=0)
            avg_df = pd.DataFrame([{"mae": mae, "r2": r2, "rho": rho, "rs": rs}])
            avg_path = os.path.join(AVG_DIR, f"{case_name}_average.csv")
            avg_df.to_csv(avg_path, index=False)
            print(f"   ✓ Saved: {avg_path}")


    # ===== WITHIN & CROSS-RATER (using individual rater data) =====
    for rater_file, feature_file, rep_filter, rating_col, case_name in rater_cases:
        print(f"\n{'='*70}")
        print(f"{case_name}")
        print(f"{'='*70}")


        # ===== WITHIN-RATER =====
        print("\n[1/2] WITHIN-RATER (10 runs × 5 raters)...")
        # Cartesian product of runs and raters for parallel dispatch
        within_task_args = [
            (run_id, rater_file, feature_file, rep_filter, rating_col, rater)
            for run_id in range(NUM_RUNS)
            for rater in target_raters
        ]


        with Pool(processes=num_processes) as pool:
            within_results_flat = pool.map(run_within_single, within_task_args)


        # Reorganize flat results list back into per-rater groups.
        # The flat list order matches the nested loop: outer=run_id, inner=rater.
        within_by_rater = {r: [] for r in target_raters}
        idx = 0
        for run_id in range(NUM_RUNS):
            for rater in target_raters:
                if within_results_flat[idx] is not None:
                    within_by_rater[rater].append(within_results_flat[idx])
                idx += 1


        # Average metrics across runs for each rater
        within_rows = []
        for rater in target_raters:
            if within_by_rater[rater]:
                arr = np.array([[m["mae"], m["r2"], m["rho"], m["rs"]] for m in within_by_rater[rater]])
                mae, r2, rho, rs = arr.mean(axis=0)
                within_rows.append({"Rater": rater, "mae": mae, "r2": r2, "rho": rho, "rs": rs})


        within_df = pd.DataFrame(within_rows)
        within_path = os.path.join(WITHIN_DIR, f"{case_name}_within_rater.csv")
        within_df.to_csv(within_path, index=False)
        print(f"   ✓ Saved: {within_path}")


        # ===== CROSS-RATER =====
        print("[2/2] CROSS-RATER (10 runs × 5 raters)...")
        cross_task_args = [
            (run_id, rater_file, feature_file, rep_filter, rating_col, rater)
            for run_id in range(NUM_RUNS)
            for rater in target_raters
        ]


        with Pool(processes=num_processes) as pool:
            cross_results_flat = pool.map(run_cross_single, cross_task_args)


        # Reorganize and group by rater (same structure as within-rater above)
        cross_by_rater = {r: [] for r in target_raters}
        idx = 0
        for run_id in range(NUM_RUNS):
            for rater in target_raters:
                if cross_results_flat[idx] is not None:
                    cross_by_rater[rater].append(cross_results_flat[idx])
                idx += 1


        # Average metrics across runs for each rater
        cross_rows = []
        for rater in target_raters:
            if cross_by_rater[rater]:
                arr = np.array([[m["mae"], m["r2"], m["rho"], m["rs"]] for m in cross_by_rater[rater]])
                mae, r2, rho, rs = arr.mean(axis=0)
                cross_rows.append({"Rater": rater, "mae": mae, "r2": r2, "rho": rho, "rs": rs})


        cross_df = pd.DataFrame(cross_rows)
        cross_path = os.path.join(CROSS_DIR, f"{case_name}_cross_rater.csv")
        cross_df.to_csv(cross_path, index=False)
        print(f"   ✓ Saved: {cross_path}")


    print(f"\n{'='*70}")
    print("✓ All experiments completed!")
    print(f"{'='*70}\n")

    # ===== GENERATE SUMMARY (AVERAGE OF RATERS) =====
    # After all per-rater results are saved, compute per-case averages
    # across all 5 raters. These summaries provide the single "Baseline"
    # row values reported in Tables 2-5.
    print("="*70)
    print("CALCULATING PER-CASE AVERAGES")
    print("="*70)

    case_names = [
        "abstract_beauty",
        "abstract_liking",
        "representational_beauty",
        "representational_liking"
    ]

    dir_map = {
        "within": (WITHIN_DIR, "within_rater"),
        "cross": (CROSS_DIR, "cross_rater"),
        "average": (AVG_DIR, "average")
    }

    for analysis_type in ["within", "cross", "average"]:
        out_dir, suffix = dir_map[analysis_type]
        print(f"\n[SUMMARY] {analysis_type.upper()} - Per-case averages...")

        summary_rows = []

        for case_name in case_names:
            file_path = os.path.join(out_dir, f"{case_name}_{suffix}.csv")

            if not os.path.exists(file_path):
                print(f"  ⚠ File not found: {file_path}")
                continue

            df = pd.read_csv(file_path)

            # Average across all raters (within/cross) or just take the
            # single row (average-rater) to get one summary value per case
            mean_mae = df["mae"].mean()
            mean_r2 = df["r2"].mean()
            mean_rho = df["rho"].mean()
            mean_rs = df["rs"].mean()

            summary_rows.append({
                "Case": case_name,
                "mae": mean_mae,
                "r2": mean_r2,
                "rho": mean_rho,
                "rs": mean_rs
            })

            print(f"  {case_name}:")
            print(f"    mae={mean_mae:.6f}, r2={mean_r2:.6f}, rho={mean_rho:.6f}, rs={mean_rs:.6f}")

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_dir, f"summary_{suffix}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"  ✓ Saved: {summary_path}")

    print(f"\n{'='*70}")
    print("✓ All summaries completed!")
    print(f"{'='*70}")
