"""Run CLIP within-rater and cross-rater Sidhu experiments.

The paper's rater-level tables use two protocols:

* within-rater: train and evaluate on the same rater;
* cross-rater: train on the mean of the other four raters and evaluate on the
  held-out target rater.

This runner mirrors the controlled aggregate experiments in
``run_art_extensions.py`` while keeping the training labels and evaluation
labels separate for the cross-rater protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_art_extensions import (
    calibrate,
    metrics,
    pair_accuracy,
    parse_range,
    seed_everything,
    standardize,
    train_pairwise,
    train_regression,
)


RATER_FILES = {
    ("abstract", "beauty"): ("Abstract_All_Raters.csv", "Beauty"),
    ("abstract", "liking"): ("Abstract_Liking_All_Raters.csv", "Liking"),
    ("representational", "beauty"): (
        "Representational_All_Raters.csv",
        "Beauty",
    ),
    ("representational", "liking"): (
        "Representational_Liking_All_Raters.csv",
        "Liking",
    ),
}


def load_rater_data(
    features_path: Path,
    data_dir: Path,
    category: str,
    target: str,
    rater: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filename, rating_column = RATER_FILES[(category, target)]
    ratings_by_item: dict[str, dict[int, float]] = {}
    with (data_dir / filename).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            current_rater = int(row["Rater"])
            if current_rater not in range(1, 6):
                continue
            painting_number = int(Path(row["Painting"]).stem)
            item_id = f"{category}-{painting_number - 1:03d}"
            ratings_by_item.setdefault(item_id, {})[current_rater] = float(
                row[rating_column]
            )

    with np.load(features_path) as feature_data:
        feature_ids = [str(value) for value in feature_data["item_ids"]]
        feature_matrix = np.asarray(feature_data["features"], dtype=np.float32)

    rows: list[np.ndarray] = []
    training_ratings: list[float] = []
    evaluation_ratings: list[float] = []
    other_raters = [candidate for candidate in range(1, 6) if candidate != rater]
    for item_id, feature in zip(feature_ids, feature_matrix, strict=True):
        item_ratings = ratings_by_item.get(item_id, {})
        if rater not in item_ratings:
            continue
        if mode == "within":
            training_rating = item_ratings[rater]
        else:
            available = [
                item_ratings[candidate]
                for candidate in other_raters
                if candidate in item_ratings
            ]
            if not available:
                continue
            training_rating = float(np.mean(available))
        rows.append(feature)
        training_ratings.append(training_rating)
        evaluation_ratings.append(item_ratings[rater])

    if len(rows) < 200:
        raise ValueError(
            f"{category}/{target}/{mode}/rater-{rater} has only {len(rows)} rows"
        )
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(training_ratings, dtype=np.float32),
        np.asarray(evaluation_ratings, dtype=np.float32),
    )


def split_indices(item_count: int, seed: int) -> tuple[np.ndarray, ...]:
    indices = np.random.default_rng(seed).permutation(item_count)
    return indices[:140], indices[140:160], indices[160:]


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    features, training_ratings, evaluation_ratings = load_rater_data(
        args.features,
        args.data_dir,
        args.category,
        args.target,
        args.rater,
        args.mode,
    )
    objectives = args.objectives.split(",")
    n_values = parse_range(args.n_values)
    seeds = parse_range(args.seeds)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        train, validation, test = split_indices(len(training_ratings), seed)
        normalized_features = standardize(features, train)
        for objective in objectives:
            settings = (None,) if objective == "regression" else n_values
            for n in settings:
                seed_everything(seed)
                tag = (
                    f"{args.category}-{args.target}-{args.mode}-r{args.rater}-"
                    f"{objective}-n{n}-s{seed}"
                )
                checkpoint = args.checkpoint_dir / f"{tag}.weights.h5"
                if objective == "regression":
                    scores, epochs = train_regression(
                        normalized_features,
                        training_ratings,
                        train,
                        validation,
                        checkpoint,
                        args.epochs_regression,
                        args.batch_size,
                    )
                    train_pairs = validation_pairs = 0
                    calibrated = scores
                else:
                    scores, epochs, train_pairs, validation_pairs = train_pairwise(
                        normalized_features,
                        training_ratings,
                        train,
                        validation,
                        objective,
                        int(n),
                        seed,
                        checkpoint,
                        args.epochs_pairwise,
                        args.batch_size,
                    )
                    slope, intercept = calibrate(
                        scores[validation], training_ratings[validation]
                    )
                    calibrated = scores * slope + intercept

                result = metrics(
                    evaluation_ratings[test], scores[test], calibrated[test]
                )
                accuracy, test_pairs = pair_accuracy(
                    scores,
                    test,
                    evaluation_ratings,
                    seed=seed * 1019 + (0 if n is None else int(n)),
                )
                row: dict[str, object] = {
                    "dataset": "sidhu",
                    "representation": "clip-vit-b32",
                    "category": args.category,
                    "target": args.target,
                    "mode": args.mode,
                    "rater": args.rater,
                    "objective": objective,
                    "N": "" if n is None else int(n),
                    "seed": seed,
                    "train_examples": len(train),
                    "validation_examples": len(validation),
                    "test_examples": len(test),
                    "train_pairs": train_pairs,
                    "validation_pairs": validation_pairs,
                    "test_pairs": test_pairs,
                    "epochs_trained": epochs,
                    "pair_accuracy": accuracy,
                    **result,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True))
                checkpoint.unlink(missing_ok=True)
    return rows


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
    parser.add_argument("--epochs-pairwise", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "dataset": "sidhu",
        "representation": "clip-vit-b32",
        "category": args.category,
        "target": args.target,
        "mode": args.mode,
        "rater": args.rater,
        "objectives": args.objectives.split(","),
        "n_values": parse_range(args.n_values),
        "seeds": parse_range(args.seeds),
        "rows": len(rows),
        "sha256": digest,
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "split": "140 train / 20 validation / remainder test",
        "training_only_standardization": True,
        "cross_training_labels": "mean of the other four raters",
        "evaluation_labels": "target rater",
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
