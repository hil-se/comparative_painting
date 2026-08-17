"""Evaluate the APDDv2-locked shallow head across the art experiments.

The configuration is selected once on APDDv2 validation macro Spearman and is
then kept fixed for every dataset, representation, objective, target, rater,
and random seed.  Pretrained visual encoders remain frozen; this runner trains
only the shared downstream scalar prediction head.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.stats import spearmanr

from run_art_extensions import (
    Dataset,
    bradley_terry_pairwise_loss,
    calibrate,
    generate_pairs,
    hinge_pairwise_loss,
    load_dataset,
    metrics,
    pair_accuracy,
    parse_range,
    seed_everything,
    split_indices,
)
from tune_apddv2_regression import (
    CONFIG_BY_ID,
    build_model,
    fit_predict,
)


LOCKED_METHOD_ID = "shallow-mse-z-gelu-ln-rawclip"
LOCKED_CONFIG = CONFIG_BY_ID[LOCKED_METHOD_ID]
OBJECTIVES = ("regression", "hinge", "bradley_terry")


def validation_spearman_callback(
    encoder,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    patience: int,
    minimum_epochs: int,
):
    """Stop pairwise training using item-level validation Spearman."""

    import tensorflow as tf

    class ValidationSpearman(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best = -np.inf
            self.best_epoch = 0
            self.best_weights: list[np.ndarray] | None = None
            self.wait = 0

        def on_epoch_end(self, epoch, logs=None) -> None:
            logs = {} if logs is None else logs
            prediction = np.asarray(
                encoder(validation_features, training=False)
            ).ravel()
            score = float(
                spearmanr(validation_targets, prediction).statistic
            )
            logs["val_spearman"] = score
            if np.isfinite(score) and score > self.best + 1e-4:
                self.best = score
                self.best_epoch = epoch + 1
                self.best_weights = encoder.get_weights()
                self.wait = 0
            else:
                self.wait += 1
            if epoch + 1 >= minimum_epochs and self.wait >= patience:
                self.model.stop_training = True

        def on_train_end(self, logs=None) -> None:
            if self.best_weights is None:
                raise RuntimeError("No finite validation Spearman was observed")
            encoder.set_weights(self.best_weights)

    return ValidationSpearman()


def train_locked_pairwise(
    features: np.ndarray,
    ratings: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    objective: str,
    n: int,
    seed: int,
    *,
    epochs: int,
    patience: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """Train one pairwise model with the locked shallow scalar head."""

    import tensorflow as tf

    train_left, train_right, train_labels = generate_pairs(
        train, ratings, n=n, seed=seed * 1009 + n
    )
    validation_left, validation_right, validation_labels = generate_pairs(
        validation, ratings, n=min(n, 5), seed=seed * 1013 + n
    )
    if not len(train_labels) or not len(validation_labels):
        raise ValueError("Pair generation produced an empty training or validation set")

    encoder = build_model(features.shape[1], LOCKED_CONFIG)
    left_input = tf.keras.Input(shape=(features.shape[1],), name="left")
    right_input = tf.keras.Input(shape=(features.shape[1],), name="right")
    difference = encoder(left_input) - encoder(right_input)
    model = tf.keras.Model([left_input, right_input], difference)
    if objective == "hinge":
        loss = hinge_pairwise_loss
    elif objective == "bradley_terry":
        loss = bradley_terry_pairwise_loss
    else:
        raise ValueError(objective)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LOCKED_CONFIG.learning_rate
        ),
        loss=loss,
        jit_compile=False,
    )
    stopper = validation_spearman_callback(
        encoder,
        features[validation],
        ratings[validation],
        patience=patience,
        minimum_epochs=min(25, epochs),
    )
    callbacks: list[object] = [stopper]
    if LOCKED_CONFIG.reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_spearman",
                mode="max",
                factor=0.5,
                patience=max(4, patience // 2),
                min_delta=1e-4,
                min_lr=1e-6,
                verbose=0,
            )
        )
    history = model.fit(
        [features[train_left], features[train_right]],
        train_labels,
        validation_data=(
            [features[validation_left], features[validation_right]],
            validation_labels,
        ),
        epochs=epochs,
        batch_size=LOCKED_CONFIG.batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    scores = np.asarray(encoder(features, training=False)).ravel()
    epochs_trained = len(history.epoch)
    best_epoch = stopper.best_epoch
    train_pair_count = len(train_labels)
    validation_pair_count = len(validation_labels)
    del (
        history,
        stopper,
        model,
        encoder,
        train_left,
        train_right,
        train_labels,
        validation_left,
        validation_right,
        validation_labels,
    )
    tf.keras.backend.clear_session()
    gc.collect()
    return (
        scores,
        epochs_trained,
        best_epoch,
        train_pair_count,
        validation_pair_count,
    )


def run_locked_arrays(
    *,
    features: np.ndarray,
    training_ratings: np.ndarray,
    evaluation_ratings: np.ndarray,
    split_function: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dataset_name: str,
    representation: str,
    category: str,
    target: str,
    objectives: Iterable[str],
    n_values: Iterable[int],
    seeds: Iterable[int],
    epochs_regression: int,
    epochs_pairwise: int,
    patience: int,
    extra_fields: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Run the locked method on aggregate or rater-level arrays."""

    rows: list[dict[str, object]] = []
    feature_matrix = np.asarray(features, dtype=np.float32)
    training_ratings = np.asarray(training_ratings, dtype=np.float32)
    evaluation_ratings = np.asarray(evaluation_ratings, dtype=np.float32)
    extra_fields = extra_fields or {}

    if LOCKED_CONFIG.feature_standardization:
        raise AssertionError("The locked method must use fixed extractor outputs")
    unknown = sorted(set(objectives) - set(OBJECTIVES))
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")

    for seed in seeds:
        train, validation, test = split_function(seed)
        for objective in objectives:
            settings = (None,) if objective == "regression" else n_values
            for n in settings:
                seed_everything(seed)
                if objective == "regression":
                    validation_prediction, test_prediction, epochs, best_epoch = (
                        fit_predict(
                            LOCKED_CONFIG,
                            feature_matrix[train],
                            training_ratings[train],
                            feature_matrix[validation],
                            training_ratings[validation],
                            feature_matrix[test],
                            epochs=epochs_regression,
                            patience=patience,
                        )
                    )
                    if test_prediction is None:
                        raise AssertionError("Regression test prediction is missing")
                    scores = np.zeros(len(training_ratings), dtype=np.float32)
                    scores[validation] = validation_prediction
                    scores[test] = test_prediction
                    calibrated = scores
                    train_pairs = validation_pairs = 0
                    gc.collect()
                else:
                    (
                        scores,
                        epochs,
                        best_epoch,
                        train_pairs,
                        validation_pairs,
                    ) = train_locked_pairwise(
                        feature_matrix,
                        training_ratings,
                        train,
                        validation,
                        objective,
                        int(n),
                        seed,
                        epochs=epochs_pairwise,
                        patience=patience,
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
                    "phase": "locked_method_evaluation",
                    "method_id": LOCKED_METHOD_ID,
                    "dataset": dataset_name,
                    "representation": representation,
                    "category": category,
                    "target": target,
                    **extra_fields,
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
                    "best_epoch": best_epoch,
                    "pair_accuracy": accuracy,
                    **result,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def write_results(
    *,
    output: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
    features_path: Path,
    manifest_path: Path | None = None,
) -> None:
    if not rows:
        raise ValueError("No result rows were produced")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        **metadata,
        "phase": "locked_method_evaluation",
        "method_id": LOCKED_METHOD_ID,
        "method_selected_on": "APDDv2 validation macro Spearman",
        "method_retuned_on_evaluation_data": False,
        "config": {
            field: getattr(LOCKED_CONFIG, field)
            for field in LOCKED_CONFIG.__dataclass_fields__
        },
        "feature_standardization": False,
        "regression_target_standardization": True,
        "selection_and_early_stopping_metric": "validation Spearman",
        "pairwise_validation_affine_calibration": True,
        "pairwise_loss_elementwise_shape_check": True,
        "rows": len(rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "tensorflow": __import__("tensorflow").__version__,
    }
    if manifest_path is not None:
        payload["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    output.with_suffix(".metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sidhu", "apddv2"), required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--category", default="all")
    parser.add_argument("--target", required=True)
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
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
    data: Dataset = load_dataset(
        args.manifest,
        args.features,
        args.target,
        None if args.category == "all" else args.category,
    )
    rows = run_locked_arrays(
        features=data.features,
        training_ratings=data.ratings,
        evaluation_ratings=data.ratings,
        split_function=lambda seed: split_indices(
            len(data.ratings), args.dataset, seed
        ),
        dataset_name=args.dataset,
        representation=args.representation,
        category=args.category,
        target=args.target,
        objectives=objectives,
        n_values=n_values,
        seeds=parse_range(args.seeds),
        epochs_regression=args.epochs_regression,
        epochs_pairwise=args.epochs_pairwise,
        patience=args.patience,
    )
    write_results(
        output=args.output,
        rows=rows,
        features_path=args.features,
        manifest_path=args.manifest,
        metadata={
            "dataset": args.dataset,
            "representation": args.representation,
            "category": args.category,
            "target": args.target,
            "objectives": objectives,
            "n_values": recorded_n_values,
            "seeds": parse_range(args.seeds),
            "split": (
                "140 train / 20 validation / remainder test"
                if args.dataset == "sidhu"
                else "70% train / 15% validation / 15% test"
            ),
        },
    )


if __name__ == "__main__":
    main()
