"""
Orchestrator for comparative (pairwise) aesthetic prediction experiments.

This module runs the comparative learning experiments described in the IEEE
Access paper. Unlike the regression approach (see experiment_runner.py),
comparative learning trains on pairwise preference judgments generated from
rating data, using a Siamese-style architecture (ComparativeModel).

The key experimental variable is N -- the number of comparative judgments
generated per training item. Experiments sweep N from 1 to 10 to study
how the amount of pairwise supervision affects prediction quality.

Three experiment types are supported:
- Average rating: Pairwise judgments derived from mean ratings across raters.
- Within-rater: Pairwise judgments from a single rater's ratings.
- Cross-rater: Pairwise judgments from leave-one-out average, tested on
  the held-out rater.

Each experiment is repeated 10 times with random train/test splits across
5 raters and N=1..10, using multiprocessing for parallelism.

Results (MAE, R2, Pearson rho, Spearman rs) are saved as CSV files under
``../../results/deep_learning/comparative/``.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
# Import from comparative_modified.py (or rename it to comparative.py)
from comparative_modified import within, cross, build_model, ComparativeModel, generate_comparative_judgments
from metrics import Metrics
from multiprocessing import Pool, cpu_count
import os
from functools import partial
import time
import random


def average_rating(painting, type, origin=True, N=1, process_id=None):
    """
    Train and evaluate a comparative model using average ratings.

    Generates N pairwise comparative judgments per training item from the
    average ratings, trains a ComparativeModel (Siamese MLP with hinge loss),
    and evaluates the encoder's scalar predictions on the held-out test set.

    Parameters
    ----------
    painting : str
        Painting category: "abstract" or "representational".
    type : str
        Rating dimension: "beauty" or "liking".
    origin : bool, optional
        If True, use original (non-augmented) features. Default is True.
    N : int, optional
        Number of comparative judgments to generate per training item.
        Higher N means more pairwise training data. Default is 1.
    process_id : str or None, optional
        Unique identifier for checkpoint file paths in multiprocessing.
        If None, a unique ID is generated automatically.

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

    # Random 140/remaining train/test split
    train = np.random.choice(len(y), 140, replace=False)
    test = np.array(list(set(list(range(len(y))))-set(list(train))))

    # Generate pairwise training data: N comparisons per training item
    features = generate_comparative_judgments(X[train], y[train], N=N)

    # Build Siamese architecture: shared encoder wrapped in ComparativeModel
    encoder = build_model((X[train].shape[1]))
    model = ComparativeModel(encoder)

    # Generate a unique checkpoint path to avoid file collisions in multiprocessing
    if process_id is None:
        process_id = f"{os.getpid()}_{int(time.time()*1000000)}_{random.randint(0,999999)}"
    checkpoint_path = f"checkpoint/comparative_avg_{process_id}.weights.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model.compile(optimizer="adam")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=0)

    # Train model on pairwise comparisons with hinge loss
    model.fit(features, features["Label"], epochs=100, batch_size=10, callbacks=[reduce_lr, checkpoint],
              verbose=0)

    # Restore best weights and predict using the encoder directly
    model.load_weights(checkpoint_path)

    preds_test = model.predict(X[test]).flatten()
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
        (painting, type, origin, N, run_id).

    Returns
    -------
    dict
        Metrics dictionary from average_rating().
    """
    painting, type, origin, N, run_id = args
    print(f"  → Starting average rating run {run_id + 1}/10...")
    process_id = f"avg_{run_id}"
    result = average_rating(painting, type, origin, N, process_id)
    print(f"  ✓ Completed average rating run {run_id + 1}/10")
    return result


def run_within_experiment(args):
    """
    Wrapper that unpacks arguments and calls within() for pool.map().

    Parameters
    ----------
    args : tuple
        (painting, type, rater, origin, N, run_id).

    Returns
    -------
    dict
        Metrics dictionary from within().
    """
    painting, type, rater, origin, N, run_id = args
    print(f"  → Starting within-rater {rater} run {run_id + 1}/10...")
    process_id = f"within_r{rater}_{run_id}"
    result = within(painting, type, rater, origin, N, process_id)
    print(f"  ✓ Completed within-rater {rater} run {run_id + 1}/10")
    return result


def run_cross_experiment(args):
    """
    Wrapper that unpacks arguments and calls cross() for pool.map().

    Parameters
    ----------
    args : tuple
        (painting, type, rater, origin, N, run_id).

    Returns
    -------
    dict
        Metrics dictionary from cross().
    """
    painting, type, rater, origin, N, run_id = args
    print(f"  → Starting cross-rater {rater} run {run_id + 1}/10...")
    process_id = f"cross_r{rater}_{run_id}"
    result = cross(painting, type, rater, origin, N, process_id)
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
    Main entry point: configure and run all comparative experiments.

    Iterates over painting types, rating types, and N values (1..10),
    runs average/within/cross comparative experiments in parallel using
    multiprocessing.Pool, averages metrics across 10 runs per setting,
    and saves results to CSV.
    """
    # ==================== CONFIGURATION ====================
    paintings = ["abstract", "representational"]  # Options: ["abstract"], ["representational"], or both
    types = ["beauty", "liking"]  # Options: ["beauty"], ["liking"], or both
    origin = False
    N_values = list(range(1, 11))  # N = 1, 2, 3, ..., 10 (number of comparative judgments per item)
    num_runs = 10
    num_raters = 5

    # Choose which experiments to run
    run_average = True    # Set to False to skip average rating experiments
    run_within = True     # Set to False to skip within-rater experiments
    run_cross = True      # Set to False to skip cross-rater experiments
    # ======================================================

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    # Create result directory
    os.makedirs("../../results/deep_learning/comparative", exist_ok=True)

    # Determine number of processes (cap at 8 to avoid excessive memory usage)
    num_processes = min(cpu_count(), 8)  # Use up to 8 cores
    print(f"Using {num_processes} processes for parallel execution")
    print(f"Configuration: types={types}, origin={origin}, N_values={N_values}, runs_per_experiment={num_runs}")
    print(f"Experiments to run: Average={run_average}, Within={run_within}, Cross={run_cross}\n")

    for painting in paintings:
        for type in types:
            print("\n" + "="*70)
            print(f"PROCESSING: {painting.upper()} - {type.upper()}")
            print("="*70)

            # ========== Average Rating Experiments ==========
            if run_average:
                print("\n" + "="*50)
                print("Running Average Rating Experiments (Comparative)")
                print("="*50)

                avg_results_all_N = []

                # Sweep over N values to study effect of pairwise data quantity
                for N in N_values:
                    print(f"\n→ Processing N={N} (comparative judgments per item)")
                    avg_args = [(painting, type, origin, N, i) for i in range(num_runs)]

                    with Pool(processes=num_processes) as pool:
                        avg_results = pool.map(run_average_experiment, avg_args)

                    avg_mean = average_results(avg_results)
                    avg_mean["N"] = N
                    avg_results_all_N.append(avg_mean)
                    print(f"✓ N={N} completed - Mean: mae={avg_mean['mae']:.4f}, r2={avg_mean['r2']:.4f}")

                # Save to CSV with N column first
                avg_df = pd.DataFrame(avg_results_all_N)
                avg_df = avg_df[["N", "mae", "r2", "rho", "rs"]]  # Reorder columns
                avg_df.to_csv(f"../../results/deep_learning/comparative/{painting}_{type}_average_comparative.csv", index=False)
                print(f"\n✓ Saved to result/{painting}_{type}_average_comparative.csv")
            else:
                print("\n⊗ Skipping Average Rating Experiments")

            # ========== Within-Rater Experiments ==========
            if run_within:
                print("\n" + "="*50)
                print("Running Within-Rater Experiments (Comparative)")
                print("="*50)

                within_results_all_N = []

                for N in N_values:
                    print(f"\n→ Processing N={N} (comparative judgments per item)")

                    for rater in range(1, num_raters + 1):
                        print(f"  → Processing rater {rater}/{num_raters}")
                        within_args = [(painting, type, rater, origin, N, i) for i in range(num_runs)]

                        with Pool(processes=num_processes) as pool:
                            within_results = pool.map(run_within_experiment, within_args)

                        within_mean = average_results(within_results)
                        within_mean["N"] = N
                        within_mean["Rater"] = rater
                        within_results_all_N.append(within_mean)
                        print(f"  ✓ Rater {rater} completed - Mean: mae={within_mean['mae']:.4f}, r2={within_mean['r2']:.4f}")

                # Save to CSV with N and Rater columns first
                within_df = pd.DataFrame(within_results_all_N)
                within_df = within_df[["N", "Rater", "mae", "r2", "rho", "rs"]]  # Reorder columns
                within_df.to_csv(f"../../results/deep_learning/comparative/{painting}_{type}_within_comparative.csv", index=False)
                print(f"\n✓ Saved to result/{painting}_{type}_within_comparative.csv")
            else:
                print("\n⊗ Skipping Within-Rater Experiments")

            # ========== Cross-Rater Experiments ==========
            if run_cross:
                print("\n" + "="*50)
                print("Running Cross-Rater Experiments (Comparative)")
                print("="*50)

                cross_results_all_N = []

                for N in N_values:
                    print(f"\n→ Processing N={N} (comparative judgments per item)")

                    for rater in range(1, num_raters + 1):
                        print(f"  → Processing rater {rater}/{num_raters}")
                        cross_args = [(painting, type, rater, origin, N, i) for i in range(num_runs)]

                        with Pool(processes=num_processes) as pool:
                            cross_results = pool.map(run_cross_experiment, cross_args)

                        cross_mean = average_results(cross_results)
                        cross_mean["N"] = N
                        cross_mean["Rater"] = rater
                        cross_results_all_N.append(cross_mean)
                        print(f"  ✓ Rater {rater} completed - Mean: mae={cross_mean['mae']:.4f}, r2={cross_mean['r2']:.4f}")

                # Save to CSV with N and Rater columns first
                cross_df = pd.DataFrame(cross_results_all_N)
                cross_df = cross_df[["N", "Rater", "mae", "r2", "rho", "rs"]]  # Reorder columns
                cross_df.to_csv(f"../../results/deep_learning/comparative/{painting}_{type}_cross_comparative.csv", index=False)
                print(f"\n✓ Saved to result/{painting}_{type}_cross_comparative.csv")
            else:
                print("\n⊗ Skipping Cross-Rater Experiments")

    print("\n" + "="*50)
    print("All comparative experiments completed!")
    print("="*50)


if __name__ == '__main__':
    main()
