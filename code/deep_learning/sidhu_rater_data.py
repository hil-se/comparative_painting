"""Load the five Sidhu raters for within- and cross-rater prediction."""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

RATER_FILES = {
    ("abstract", "beauty"): ("Abstract_All_Raters.csv", "Beauty"),
    ("abstract", "liking"): ("Abstract_Liking_All_Raters.csv", "Liking"),
    ("representational", "beauty"): ("Representational_All_Raters.csv", "Beauty"),
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
    filename, rating_column = RATER_FILES[category, target]
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
    for item_id, feature in zip(feature_ids, feature_matrix):
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
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(training_ratings, dtype=np.float32),
        np.asarray(evaluation_ratings, dtype=np.float32),
    )


def split_indices(item_count: int, seed: int) -> tuple[np.ndarray, ...]:
    indices = np.random.default_rng(seed).permutation(item_count)
    return (indices[:140], indices[140:160], indices[160:])
