"""Evaluate the APDDv2-locked tuned CLIP regressor on Sidhu.

This is a transfer evaluation, not a Sidhu tuning run. The configuration was
selected using APDDv2 validation data and is applied unchanged to all four
Sidhu category/target conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import r2_score

from run_art_extensions import load_dataset, pair_accuracy, seed_everything, split_indices
from tune_apddv2_regression import (
    CONFIG_BY_ID,
    fit_predict,
    parse_range,
    regression_metrics,
    scale_features,
)


LOCKED_CONFIG_ID = "shallow-mse-z-gelu-ln-rawclip"


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.config_id != LOCKED_CONFIG_ID:
        raise ValueError(
            f"Sidhu transfer must use the APDDv2-locked config {LOCKED_CONFIG_ID!r}"
        )
    config = CONFIG_BY_ID[args.config_id]
    data = load_dataset(
        args.manifest,
        args.features,
        args.target,
        category_filter=args.category,
    )
    rows: list[dict[str, object]] = []

    for seed in parse_range(args.seeds):
        train, validation, test = split_indices(len(data.ratings), "sidhu", seed)
        train_features, validation_features, test_features = scale_features(
            data.features[train],
            data.features[validation],
            data.features[test],
            enabled=config.feature_standardization,
        )
        if test_features is None:
            raise AssertionError("Sidhu transfer test features are missing")
        seed_everything(seed)
        validation_prediction, test_prediction, epochs_trained, best_epoch = fit_predict(
            config,
            train_features,
            data.ratings[train],
            validation_features,
            data.ratings[validation],
            test_features,
            epochs=args.epochs,
            patience=args.patience,
        )
        if test_prediction is None:
            raise AssertionError("Sidhu transfer test prediction is missing")

        full_scores = np.zeros(len(data.ratings), dtype=np.float32)
        full_scores[test] = test_prediction
        accuracy, pair_count = pair_accuracy(
            full_scores,
            test,
            data.ratings,
            seed=seed * 1019,
        )
        row: dict[str, object] = {
            "phase": "locked_transfer_test",
            "dataset": "sidhu",
            "representation": "clip-vit-b32",
            "category": args.category,
            "target": args.target,
            "config_id": args.config_id,
            "seed": seed,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "test_examples": len(test),
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "test_pairs": pair_count,
            "test_pair_accuracy": accuracy,
            **regression_metrics(
                data.ratings[validation],
                validation_prediction,
                "validation",
            ),
            **regression_metrics(data.ratings[test], test_prediction, "test"),
            "test_r2": float(r2_score(data.ratings[test], test_prediction)),
            "test_kendall": float(
                kendalltau(data.ratings[test], test_prediction).statistic
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--category", choices=("abstract", "representational"), required=True
    )
    parser.add_argument("--target", choices=("beauty", "liking"), required=True)
    parser.add_argument("--config-id", default=LOCKED_CONFIG_ID)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config = CONFIG_BY_ID[args.config_id]
    metadata = {
        "phase": "locked_transfer_test",
        "dataset": "sidhu",
        "category": args.category,
        "target": args.target,
        "config": asdict(config),
        "configuration_selected_on": "APDDv2 validation macro Spearman",
        "sidhu_hyperparameter_selection": False,
        "seeds": parse_range(args.seeds),
        "split": "140 train / 20 validation / 40 test",
        "rows": len(rows),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
