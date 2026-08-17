"""Run the APDDv2-locked shallow head on Sidhu rater-level protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_art_extensions import parse_range
from run_art_locked_head import run_locked_arrays, write_results
from run_sidhu_rater_extensions import load_rater_data, split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--category", choices=("abstract", "representational"), required=True
    )
    parser.add_argument("--target", choices=("beauty", "liking"), required=True)
    parser.add_argument("--mode", choices=("within", "cross"), required=True)
    parser.add_argument("--rater", type=int, choices=range(1, 6), required=True)
    parser.add_argument(
        "--objectives", default="regression,hinge,bradley_terry"
    )
    parser.add_argument("--n-values", default="1-10")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--epochs-regression", type=int, default=200)
    parser.add_argument("--epochs-pairwise", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objectives = args.objectives.split(",")
    n_values = parse_range(args.n_values)
    recorded_n_values = (
        n_values if any(objective != "regression" for objective in objectives) else []
    )
    features, training_ratings, evaluation_ratings = load_rater_data(
        args.features,
        args.data_dir,
        args.category,
        args.target,
        args.rater,
        args.mode,
    )
    rows = run_locked_arrays(
        features=features,
        training_ratings=training_ratings,
        evaluation_ratings=evaluation_ratings,
        split_function=lambda seed: split_indices(len(training_ratings), seed),
        dataset_name="sidhu",
        representation="clip-vit-b32",
        category=args.category,
        target=args.target,
        objectives=objectives,
        n_values=n_values,
        seeds=parse_range(args.seeds),
        epochs_regression=args.epochs_regression,
        epochs_pairwise=args.epochs_pairwise,
        patience=args.patience,
        extra_fields={"mode": args.mode, "rater": args.rater},
    )
    write_results(
        output=args.output,
        rows=rows,
        features_path=args.features,
        metadata={
            "dataset": "sidhu",
            "representation": "clip-vit-b32",
            "category": args.category,
            "target": args.target,
            "mode": args.mode,
            "rater": args.rater,
            "objectives": objectives,
            "n_values": recorded_n_values,
            "seeds": parse_range(args.seeds),
            "split": "140 train / 20 validation / remainder test",
            "cross_training_labels": "mean of the other four raters",
            "evaluation_labels": "target rater",
        },
    )


if __name__ == "__main__":
    main()
