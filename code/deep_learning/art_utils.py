"""Data, splits, pair sampling, and objectives used in the paper."""

from __future__ import annotations
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def seed_everything(seed):
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


@dataclass(frozen=True)
class Dataset:
    item_ids: np.ndarray
    features: np.ndarray
    ratings: np.ndarray
    categories: np.ndarray


def load_dataset(
    manifest: Path, feature_file: Path, target: str, category_filter: str | None = None
) -> Dataset:
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(feature_file) as feature_data:
        feature_ids = [str(value) for value in feature_data["item_ids"]]
        feature_matrix = np.asarray(feature_data["features"], dtype=np.float32)
    feature_index = {item_id: index for index, item_id in enumerate(feature_ids)}
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
        rating = float(raw_rating)
        if not math.isfinite(rating):
            continue
        item_ids.append(item_id)
        features.append(feature_matrix[feature_index[item_id]])
        ratings.append(rating)
        categories.append(row.get("category", ""))
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
        train_count, validation_count = (140, 20)
    else:
        train_count = int(round(item_count * 0.7))
        validation_count = int(round(item_count * 0.15))
    return (
        indices[:train_count],
        indices[train_count : train_count + validation_count],
        indices[train_count + validation_count :],
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


def calibrate(
    validation_scores: np.ndarray, validation_ratings: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack([validation_scores, np.ones_like(validation_scores)])
    slope, intercept = np.linalg.lstsq(design, validation_ratings, rcond=None)[0]
    return (float(slope), float(intercept))


def metrics(ratings, raw_scores, calibrated_scores):
    return {
        "mae": float(mean_absolute_error(ratings, calibrated_scores)),
        "r2": float(r2_score(ratings, calibrated_scores)),
        "spearman": float(spearmanr(ratings, raw_scores).statistic),
    }


def aligned_pairwise_tensors(y_true, y_pred):
    """Flatten pair labels and score differences to matching 1-D tensors.

    Keras expands one-dimensional targets to ``(batch, 1)`` before invoking a
    compiled loss. Flattening only the predictions would therefore broadcast
    ``(batch, 1) * (batch,)`` to ``(batch, batch)`` and mix unrelated pairs.
    """
    import tensorflow as tf

    labels = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    differences = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    return (labels, differences)


def hinge_pairwise_loss(y_true, y_pred):
    import tensorflow as tf

    labels, differences = aligned_pairwise_tensors(y_true, y_pred)
    return tf.reduce_mean(tf.nn.relu(1.0 - labels * differences))


def bradley_terry_pairwise_loss(y_true, y_pred):
    import tensorflow as tf

    labels, differences = aligned_pairwise_tensors(y_true, y_pred)
    return tf.reduce_mean(tf.nn.softplus(-labels * differences))


def parse_range(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(piece) for piece in part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values
