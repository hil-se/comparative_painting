"""Controlled regression, hinge, and Bradley-Terry art experiments.

The runner keeps splits, feature standardization, and sampled pairs identical
between representations/objectives. Pairwise scores are affine-calibrated on
validation data before MAE/R2 are computed; rank and pair metrics use the raw
latent scores.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


_TENSORFLOW_DEVICES_CONFIGURED = False


@dataclass(frozen=True)
class Dataset:
    item_ids: np.ndarray
    features: np.ndarray
    ratings: np.ndarray
    categories: np.ndarray


def seed_everything(seed: int) -> None:
    global _TENSORFLOW_DEVICES_CONFIGURED

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    if not _TENSORFLOW_DEVICES_CONFIGURED:
        memory_limit = os.environ.get("TF_GPU_MEMORY_LIMIT_MB")
        for device in tf.config.list_physical_devices("GPU"):
            if memory_limit:
                tf.config.set_logical_device_configuration(
                    device,
                    [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=float(memory_limit)
                        )
                    ],
                )
            else:
                tf.config.experimental.set_memory_growth(device, True)
        _TENSORFLOW_DEVICES_CONFIGURED = True
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_dataset(
    manifest: Path,
    feature_file: Path,
    target: str,
    category_filter: str | None = None,
) -> Dataset:
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    feature_data = np.load(feature_file)
    feature_ids = [str(value) for value in feature_data["item_ids"]]
    feature_map = {
        item_id: feature_data["features"][index]
        for index, item_id in enumerate(feature_ids)
    }

    item_ids: list[str] = []
    features: list[np.ndarray] = []
    ratings: list[float] = []
    categories: list[str] = []
    for row in rows:
        if category_filter is not None and row.get("category") != category_filter:
            continue
        raw_rating = row.get(target, "").strip()
        if not raw_rating:
            continue
        item_id = row["item_id"]
        if item_id not in feature_map:
            raise KeyError(f"No feature found for item {item_id}")
        rating = float(raw_rating)
        if not math.isfinite(rating):
            continue
        item_ids.append(item_id)
        features.append(feature_map[item_id])
        ratings.append(rating)
        categories.append(row.get("category", ""))

    if len(item_ids) < 30:
        raise ValueError(f"Target {target!r} has only {len(item_ids)} valid rows")
    return Dataset(
        item_ids=np.asarray(item_ids),
        features=np.asarray(features, dtype=np.float32),
        ratings=np.asarray(ratings, dtype=np.float32),
        categories=np.asarray(categories),
    )


def split_indices(
    item_count: int, dataset_name: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(item_count)
    if dataset_name == "sidhu":
        if item_count < 200:
            raise ValueError("Sidhu category split expects at least 200 paintings")
        train_count, validation_count = 140, 20
    else:
        train_count = int(round(item_count * 0.70))
        validation_count = int(round(item_count * 0.15))
    return (
        indices[:train_count],
        indices[train_count : train_count + validation_count],
        indices[train_count + validation_count :],
    )


def standardize(
    features: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    mean = features[train_indices].mean(axis=0)
    scale = features[train_indices].std(axis=0)
    scale[scale == 0] = 1.0
    return ((features - mean) / scale).astype(np.float32)


def build_encoder(input_dim: int):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )


def generate_pairs(
    indices: np.ndarray, ratings: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate at most N unique unordered non-tied comparisons per item."""

    rng = np.random.default_rng(seed)
    used: set[tuple[int, int]] = set()
    left: list[int] = []
    right: list[int] = []
    labels: list[float] = []
    order = rng.permutation(indices)

    for first in order:
        candidates = rng.permutation(indices[indices != first])
        added = 0
        for second in candidates:
            pair = tuple(sorted((int(first), int(second))))
            if pair in used or ratings[first] == ratings[second]:
                continue
            used.add(pair)
            left.append(int(first))
            right.append(int(second))
            labels.append(1.0 if ratings[first] > ratings[second] else -1.0)
            added += 1
            if added == n:
                break
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
    )


def pair_accuracy(
    scores: np.ndarray,
    indices: np.ndarray,
    ratings: np.ndarray,
    seed: int,
    n: int = 10,
) -> tuple[float, int]:
    left, right, labels = generate_pairs(indices, ratings, n=n, seed=seed)
    if not len(labels):
        return float("nan"), 0
    prediction = np.where(scores[left] - scores[right] >= 0, 1.0, -1.0)
    return float(np.mean(prediction == labels)), int(len(labels))


def calibrate(
    validation_scores: np.ndarray, validation_ratings: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack(
        [validation_scores, np.ones_like(validation_scores)]
    )
    slope, intercept = np.linalg.lstsq(
        design, validation_ratings, rcond=None
    )[0]
    return float(slope), float(intercept)


def metrics(
    ratings: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(ratings, calibrated_scores)),
        "mse": float(mean_squared_error(ratings, calibrated_scores)),
        "r2": float(r2_score(ratings, calibrated_scores)),
        "pearson": float(pearsonr(ratings, raw_scores).statistic),
        "spearman": float(spearmanr(ratings, raw_scores).statistic),
        "kendall": float(kendalltau(ratings, raw_scores).statistic),
    }


def callbacks(path: Path):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]


def aligned_pairwise_tensors(y_true, y_pred):
    """Flatten pair labels and score differences to matching 1-D tensors.

    Keras expands one-dimensional targets to ``(batch, 1)`` before invoking a
    compiled loss. Flattening only the predictions would therefore broadcast
    ``(batch, 1) * (batch,)`` to ``(batch, batch)`` and mix unrelated pairs.
    """

    import tensorflow as tf

    labels = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    differences = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    tf.debugging.assert_equal(
        tf.shape(labels),
        tf.shape(differences),
        message="Pair labels and score differences must align elementwise",
    )
    return labels, differences


def hinge_pairwise_loss(y_true, y_pred):
    import tensorflow as tf

    labels, differences = aligned_pairwise_tensors(y_true, y_pred)
    return tf.reduce_mean(tf.nn.relu(1.0 - labels * differences))


def bradley_terry_pairwise_loss(y_true, y_pred):
    import tensorflow as tf

    labels, differences = aligned_pairwise_tensors(y_true, y_pred)
    return tf.reduce_mean(tf.nn.softplus(-labels * differences))


def train_regression(
    features: np.ndarray,
    ratings: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    checkpoint: Path,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    import tensorflow as tf

    model = build_encoder(features.shape[1])
    model.compile(optimizer="adam", loss="mae")
    history = model.fit(
        features[train],
        ratings[train],
        validation_data=(features[validation], ratings[validation]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks(checkpoint),
        verbose=0,
    )
    scores = model.predict(features, batch_size=256, verbose=0).ravel()
    epochs_trained = len(history.history["loss"])
    del history, model
    tf.keras.backend.clear_session()
    gc.collect()
    return scores, epochs_trained


def train_pairwise(
    features: np.ndarray,
    ratings: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    objective: str,
    n: int,
    seed: int,
    checkpoint: Path,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, int, int, int]:
    import tensorflow as tf

    train_left, train_right, train_labels = generate_pairs(
        train, ratings, n=n, seed=seed * 1009 + n
    )
    val_left, val_right, val_labels = generate_pairs(
        validation, ratings, n=min(n, 5), seed=seed * 1013 + n
    )
    encoder = build_encoder(features.shape[1])
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

    model.compile(optimizer="adam", loss=loss)
    history = model.fit(
        [features[train_left], features[train_right]],
        train_labels,
        validation_data=(
            [features[val_left], features[val_right]],
            val_labels,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks(checkpoint),
        verbose=0,
    )
    scores = encoder.predict(features, batch_size=256, verbose=0).ravel()
    epochs_trained = len(history.history["loss"])
    train_pair_count = len(train_labels)
    validation_pair_count = len(val_labels)
    del (
        history,
        model,
        encoder,
        train_left,
        train_right,
        train_labels,
        val_left,
        val_right,
        val_labels,
    )
    tf.keras.backend.clear_session()
    gc.collect()
    return (
        scores,
        epochs_trained,
        train_pair_count,
        validation_pair_count,
    )


def run(
    data: Dataset,
    dataset_name: str,
    representation: str,
    category: str,
    target: str,
    objectives: Iterable[str],
    n_values: Iterable[int],
    seeds: Iterable[int],
    epochs_regression: int,
    epochs_pairwise: int,
    batch_size: int,
    checkpoint_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_everything(seed)
        train, validation, test = split_indices(
            len(data.ratings), dataset_name, seed
        )
        features = standardize(data.features, train)
        for objective in objectives:
            settings = (None,) if objective == "regression" else n_values
            for n in settings:
                seed_everything(seed)
                tag = f"{target}-{representation}-{objective}-n{n}-s{seed}"
                checkpoint = checkpoint_dir / f"{tag}.weights.h5"
                if objective == "regression":
                    scores, epochs = train_regression(
                        features,
                        data.ratings,
                        train,
                        validation,
                        checkpoint,
                        epochs_regression,
                        batch_size,
                    )
                    train_pairs = val_pairs = 0
                    calibrated = scores
                else:
                    scores, epochs, train_pairs, val_pairs = train_pairwise(
                        features,
                        data.ratings,
                        train,
                        validation,
                        objective,
                        int(n),
                        seed,
                        checkpoint,
                        epochs_pairwise,
                        batch_size,
                    )
                    slope, intercept = calibrate(
                        scores[validation], data.ratings[validation]
                    )
                    calibrated = scores * slope + intercept

                result = metrics(
                    data.ratings[test], scores[test], calibrated[test]
                )
                accuracy, test_pairs = pair_accuracy(
                    scores,
                    test,
                    data.ratings,
                    seed=seed * 1019 + (0 if n is None else int(n)),
                )
                row: dict[str, object] = {
                    "dataset": dataset_name,
                    "representation": representation,
                    "category": category,
                    "target": target,
                    "objective": objective,
                    "N": "" if n is None else int(n),
                    "seed": seed,
                    "train_examples": len(train),
                    "validation_examples": len(validation),
                    "test_examples": len(test),
                    "train_pairs": train_pairs,
                    "validation_pairs": val_pairs,
                    "test_pairs": test_pairs,
                    "epochs_trained": epochs,
                    "pair_accuracy": accuracy,
                    **result,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True))
                try:
                    checkpoint.unlink()
                except FileNotFoundError:
                    pass
    return rows


def parse_range(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(piece) for piece in part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sidhu", "apddv2"), required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument(
        "--category",
        default="all",
        help="exact manifest category to select; default uses every row",
    )
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--objectives",
        default="regression,hinge,bradley_terry",
        help="comma-separated regression, hinge, bradley_terry",
    )
    parser.add_argument("--n-values", default="1-10")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--epochs-regression", type=int, default=200)
    parser.add_argument("--epochs-pairwise", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(
        args.manifest,
        args.features,
        args.target,
        None if args.category == "all" else args.category,
    )
    rows = run(
        data=data,
        dataset_name=args.dataset,
        representation=args.representation,
        category=args.category,
        target=args.target,
        objectives=args.objectives.split(","),
        n_values=parse_range(args.n_values),
        seeds=parse_range(args.seeds),
        epochs_regression=args.epochs_regression,
        epochs_pairwise=args.epochs_pairwise,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "dataset": args.dataset,
        "representation": args.representation,
        "category": args.category,
        "target": args.target,
        "objectives": args.objectives.split(","),
        "n_values": parse_range(args.n_values),
        "seeds": parse_range(args.seeds),
        "rows": len(rows),
        "sha256": digest,
        "tensorflow": __import__("tensorflow").__version__,
        "split": (
            "140 train / 20 validation / remainder test"
            if args.dataset == "sidhu"
            else "70% train / 15% validation / 15% test"
        ),
        "training_only_standardization": True,
        "pairwise_validation_affine_calibration": True,
        "pairwise_loss_elementwise_shape_check": True,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
