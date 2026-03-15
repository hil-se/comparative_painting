"""
Orchestrator for regression-based aesthetic prediction experiments.

This module runs the regression experiments described in the IEEE Access paper,
executing all combinations of painting type, rating type, and rater across
multiple runs using multiprocessing for parallelism. Three experiment types
are supported:

- Average rating: Train/test using mean ratings across all raters.
- Within-rater: Train and test on a single rater's ratings (same rater).
- Cross-rater: Train on leave-one-out average (excluding target rater),
  test on the target rater's ratings.

Each experiment is repeated 10 times with random train/test splits, and
results are averaged. The 10 raters and 10 runs yield 100 total runs per
painting/type combination for within and cross experiments.

Results (MAE, R2, Pearson rho, Spearman rs) are saved as CSV files under
``../../results/deep_learning/regression/``.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from regression_modified import within, cross, build_model
from metrics import Metrics
from multiprocessing import Pool, cpu_count
import os
from functools import partial
import time
import random


def average_rating(painting, type, origin=True, process_id=None):
    """
    Train and evaluate an MLP regression model using average ratings.

    Loads pre-extracted image features and average human ratings, splits into
    140 training / remaining test samples, trains the MLP from build_model(),
    and evaluates on the held-out test set.

    Parameters
    ----------
    painting : str
        Painting category: "abstract" or "representational".
    type : str
        Rating dimension: "beauty" or "liking".
    origin : bool, optional
        If True, use original (non-augmented) features. Default is True.
    process_id : str or None, optional
        Unique identifier for this process, used to create a unique checkpoint
        file path so that parallel workers do not overwrite each other's
        model weights. If None, a unique ID is generated automatically.

    Returns
    -------
    dict
        Dictionary with keys "mae", "r2", "rho" (Pearson), "rs" (Spearman).
    """
    # Load features and ratings based on painting type and rating dimension
    if painting == "abstract":
        if origin:
            X = np.load('feature/abstract_feature_origin.npy')
        else:
            X = np.load('feature/abstract_feature.npy')
        if type == "beauty":
            rate = pd.read_csv("feature/abstract_beauty.csv")
        else:
            rate = pd.read_csv("feature/abstract_liking.csv")
        y = np.array(rate["Average"])
    else:
        if origin:
            X = np.load('feature/representational_feature_origin.npy')
        else:
            X = np.load('feature/representational_feature.npy')
        if type == "beauty":
            rate = pd.read_csv("feature/representational_beauty.csv")
        else:
            rate = pd.read_csv("feature/representational_liking.csv")
        y = np.array(rate["Average"])

    # Random 140/remaining train/test split (no stratification)
    train = np.random.choice(len(y), 140, replace=False)
    # Compute test indices as the set difference of all indices minus train
    test = np.array(list(set(list(range(len(y))))-set(list(train))))

    model = build_model((X[train].shape[1]))

    # Generate a unique checkpoint path to avoid file collisions in multiprocessing
    if process_id is None:
        process_id = f"{os.getpid()}_{int(time.time()*1000000)}_{random.randint(0,999999)}"
    checkpoint_path = f"checkpoint/mlp_avg_{process_id}.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=0)

    model.fit(
        X[train], y[train],
        validation_split=0.0,
        batch_size=10,
        epochs=200,
        callbacks=[reduce_lr, checkpoint],
        verbose=0
    )

    # Restore best weights found during training
    model.load_weights(checkpoint_path)
    preds_test = model.predict(X[test], verbose=0).flatten()
    m = Metrics(y[test], preds_test)
    result = {"mae": m.mae(), "r2": m.r2(), "rho": m.pearsonr().statistic, "rs": m.spearmanr().statistic}

    # Clean up checkpoint file to avoid accumulating disk usage
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    except:
        pass

    return result


def run_average_experiment(args):
    """
    Wrapper that unpacks arguments and calls average_rating() for pool.map().

    Parameters
    ----------
    args : tuple
        (painting, type, origin, run_id) -- run_id is the repeat index (0-9).

    Returns
    -------
    dict
        Metrics dictionary from average_rating().
    """
    painting, type, origin, run_id = args
    print(f"  → Starting average rating run {run_id + 1}/10...")
    process_id = f"avg_{run_id}"
    result = average_rating(painting, type, origin, process_id)
    print(f"  ✓ Completed average rating run {run_id + 1}/10")
    return result


def run_within_experiment(args):
    """
    Wrapper that unpacks arguments and calls within() for pool.map().

    Parameters
    ----------
    args : tuple
        (painting, type, rater, origin, run_id).

    Returns
    -------
    dict
        Metrics dictionary from within().
    """
    painting, type, rater, origin, run_id = args
    print(f"  → Starting within-rater {rater} run {run_id + 1}/10...")
    process_id = f"within_r{rater}_{run_id}"
    result = within(painting, type, rater, origin, process_id)
    print(f"  ✓ Completed within-rater {rater} run {run_id + 1}/10")
    return result


def run_cross_experiment(args):
    """
    Wrapper that unpacks arguments and calls cross() for pool.map().

    Parameters
    ----------
    args : tuple
        (painting, type, rater, origin, run_id).

    Returns
    -------
    dict
        Metrics dictionary from cross().
    """
    painting, type, rater, origin, run_id = args
    print(f"  → Starting cross-rater {rater} run {run_id + 1}/10...")
    result = cross(painting, type, rater, origin)
    print(f"  ✓ Completed cross-rater {rater} run {run_id + 1}/10")
    return result


def average_results(results_list):
    """
    Compute the mean of each metric across a list of experiment results.

    Parameters
    ----------
    results_list : list of dict
        Each dict has keys "mae", "r2", "rho", "rs".

    Returns
    -------
    dict
        Mean values for each metric.
    """
    mae_list = [r["mae"] for r in results_list]
    r2_list = [r["r2"] for r in results_list]
    rho_list = [r["rho"] for r in results_list]
    rs_list = [r["rs"] for r in results_list]

    return {
        "mae": np.mean(mae_list),
        "r2": np.mean(r2_list),
        "rho": np.mean(rho_list),
        "rs": np.mean(rs_list)
    }


def main():
    """
    Main entry point: configure and run all regression experiments.

    Iterates over painting types, runs average/within/cross experiments
    in parallel using multiprocessing.Pool, averages metrics across 10
    runs, and saves results to CSV.
    """
    # ==================== CONFIGURATION ====================
    paintings = ["abstract", "representational"]  # Options: ["abstract"], ["representational"], or both
    type = "liking"  # Options: "beauty" or "liking"
    origin = False
    num_runs = 10
    num_raters = 10

    # Choose which experiments to run
    run_average = True    # Set to False to skip average rating experiments
    run_within = True     # Set to False to skip within-rater experiments
    run_cross = True      # Set to False to skip cross-rater experiments
    # ======================================================

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    # Create result directory
    os.makedirs("../../results/deep_learning/regression", exist_ok=True)

    # Determine number of processes (cap at 8 to avoid excessive memory usage)
    num_processes = min(cpu_count(), 8)  # Use up to 8 cores
    print(f"Using {num_processes} processes for parallel execution")
    print(f"Configuration: type={type}, origin={origin}, runs_per_experiment={num_runs}")
    print(f"Experiments to run: Average={run_average}, Within={run_within}, Cross={run_cross}\n")

    for painting in paintings:
        print("\n" + "="*70)
        print(f"PROCESSING PAINTING TYPE: {painting.upper()}")
        print("="*70)

        # ========== Average Rating Experiments ==========
        if run_average:
            print("\n" + "="*50)
            print("Running Average Rating Experiments")
            print("="*50)

            avg_args = [(painting, type, origin, i) for i in range(num_runs)]

            # Run all 10 repeats in parallel
            with Pool(processes=num_processes) as pool:
                avg_results = pool.map(run_average_experiment, avg_args)

            avg_mean = average_results(avg_results)
            print(f"\n✓ Average Rating Mean Results: {avg_mean}")

            # Save to CSV
            pd.DataFrame([avg_mean]).to_csv(f"../../results/deep_learning/regression/{painting}_{type}_average.csv", index=False)
            print(f"✓ Saved to result/{painting}_{type}_average.csv")
        else:
            print("\n⊗ Skipping Average Rating Experiments")

        # ========== Within-Rater Experiments ==========
        if run_within:
            print("\n" + "="*50)
            print("Running Within-Rater Experiments")
            print("="*50)

            within_results_all = []

            # Process each rater sequentially; runs within a rater are parallelized
            for rater in range(1, num_raters + 1):
                print(f"\n→ Processing rater {rater}/{num_raters}")
                within_args = [(painting, type, rater, origin, i) for i in range(num_runs)]

                with Pool(processes=num_processes) as pool:
                    within_results = pool.map(run_within_experiment, within_args)

                within_mean = average_results(within_results)
                within_mean["Rater"] = rater
                within_results_all.append(within_mean)
                print(f"✓ Rater {rater} completed - Mean: mae={within_mean['mae']:.4f}, r2={within_mean['r2']:.4f}")

            # Save to CSV with Rater column first
            within_df = pd.DataFrame(within_results_all)
            within_df = within_df[["Rater", "mae", "r2", "rho", "rs"]]  # Reorder columns
            within_df.to_csv(f"../../results/deep_learning/regression/{painting}_{type}_within.csv", index=False)
            print(f"\n✓ Saved to result/{painting}_{type}_within.csv")
        else:
            print("\n⊗ Skipping Within-Rater Experiments")

        # ========== Cross-Rater Experiments ==========
        if run_cross:
            print("\n" + "="*50)
            print("Running Cross-Rater Experiments")
            print("="*50)

            cross_results_all = []

            for rater in range(1, num_raters + 1):
                print(f"\n→ Processing rater {rater}/{num_raters}")
                cross_args = [(painting, type, rater, origin, i) for i in range(num_runs)]

                with Pool(processes=num_processes) as pool:
                    cross_results = pool.map(run_cross_experiment, cross_args)

                cross_mean = average_results(cross_results)
                cross_mean["Rater"] = rater
                cross_results_all.append(cross_mean)
                print(f"✓ Rater {rater} completed - Mean: mae={cross_mean['mae']:.4f}, r2={cross_mean['r2']:.4f}")

            # Save to CSV with Rater column first
            cross_df = pd.DataFrame(cross_results_all)
            cross_df = cross_df[["Rater", "mae", "r2", "rho", "rs"]]  # Reorder columns
            cross_df.to_csv(f"../../results/deep_learning/regression/{painting}_{type}_cross.csv", index=False)
            print(f"\n✓ Saved to result/{painting}_{type}_cross.csv")
        else:
            print("\n⊗ Skipping Cross-Rater Experiments")

    print("\n" + "="*50)
    print("All experiments completed!")
    print("="*50)


if __name__ == '__main__':
    main()
