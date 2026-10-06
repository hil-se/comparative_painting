"""Refit the 11-feature Sidhu OLS baseline on the controlled painting splits.

This runner was reconstructed for the replication release. It uses the
released predictors and checks restored painting IDs against image-derived
brightness/saturation before fitting. No validation or test values enter
imputation, and complete-case sensitivity retains the assigned partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import statsmodels.api as sm

from build_art_manifests import resolve_sidhu_image, SIDHU_RATING_FILES
from run_art_extensions import parse_range, split_indices

ROOT = Path(__file__).resolve().parents[2]
PREDICTORS = (
    "HueSD", "Saturation", "SaturationSD", "Brightness", "BrightnessSD",
    "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
    "Vertical_Symmetry", "Horizontal_Symmetry", "ColourComponent",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictors(repository: Path, category: str) -> tuple[pd.DataFrame, list[int]]:
    image_dir = repository / "Data" / f"{category.title()}_Images"
    ids = [i + 1 for i in range(240) if resolve_sidhu_image(image_dir, i)]
    frame = pd.read_csv(repository / "Data" / f"{category.title()}_Data.csv")
    source_ids = frame.Painting.map(lambda name: int(Path(name).stem)).to_numpy()
    if category == "representational":
        # This release renumbered the 238 predictor rows after removing 90/157.
        if ids != [i for i in range(1, 241) if i not in (90, 157)]:
            raise ValueError("Unexpected representational image population")
        if not np.array_equal(source_ids, np.arange(1, 239)):
            raise ValueError("Unexpected representational predictor row numbering")
        frame.index = ids
    else:
        frame.index = source_ids
    if frame.index.duplicated().any():
        raise ValueError("Duplicate predictor painting IDs")
    numeric = frame.loc[:, list(PREDICTORS)].apply(pd.to_numeric, errors="coerce")
    return numeric.reindex(ids), ids


def audit_join(repository: Path, category: str, predictors: pd.DataFrame) -> dict:
    discrepancies = []
    image_dir = repository / "Data" / f"{category.title()}_Images"
    for painting, row in predictors.iterrows():
        if pd.isna(row.Brightness) or pd.isna(row.Saturation):
            continue
        image_path = resolve_sidhu_image(image_dir, int(painting) - 1)
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        maximum, minimum = pixels.max(axis=2), pixels.min(axis=2)
        saturation = np.divide(maximum - minimum, maximum,
                               out=np.zeros_like(maximum), where=maximum != 0)
        discrepancies.append(max(abs(maximum.mean() - row.Brightness),
                                 abs(saturation.mean() - row.Saturation)))
    maximum_error = max(discrepancies)
    if maximum_error > 1e-10:
        raise ValueError(f"{category} predictor/image join failed: {maximum_error}")
    return {"verified_feature_rows": len(discrepancies),
            "maximum_brightness_saturation_error": maximum_error,
            "paintings_with_missing_predictors": [int(i) for i in predictors.index[
                predictors.isna().any(axis=1)]]}


def impute_training_medians(values: np.ndarray, train: np.ndarray) -> np.ndarray:
    medians = np.nanmedian(values[train], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("A predictor has no observed training values")
    return np.where(np.isnan(values), medians, values)


def fit_condition(repository: Path, category: str, target: str,
                  predictors: pd.DataFrame, ids: list[int], seeds: list[int]):
    name, column = SIDHU_RATING_FILES[(category, target)]
    ratings = pd.read_csv(repository / "Data" / name)
    ratings["painting_id"] = ratings.Painting.map(lambda p: int(Path(p).stem))
    # Match the float32 targets consumed by the controlled neural runner.
    y = ratings.groupby("painting_id")[column].mean().reindex(ids).to_numpy(
        dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Missing aggregate rating")
    values = predictors.to_numpy(dtype=np.float64)
    complete = np.isfinite(values).all(axis=1)
    rows, predictions = [], []
    for seed in seeds:
        assigned_train, validation, assigned_test = split_indices(len(ids), "sidhu", seed)
        for protocol in ("training_median", "complete_case"):
            if protocol == "training_median":
                x = impute_training_medians(values, assigned_train)
                train, test = assigned_train, assigned_test
            else:
                x = values
                train = assigned_train[complete[assigned_train]]
                test = assigned_test[complete[assigned_test]]
            model = sm.OLS(y[train], sm.add_constant(x[train], has_constant="add")).fit()
            prediction = model.predict(sm.add_constant(x[test], has_constant="add"))
            error = y[test] - prediction
            rows.append({"dataset": "sidhu", "category": category, "target": target,
                         "objective": "ols", "protocol": protocol, "seed": seed,
                         "train_examples": len(train), "validation_examples": len(validation),
                         "test_examples": len(test), "mae": float(np.abs(error).mean()),
                         "r2": float(1 - (error ** 2).sum() /
                                     ((y[test] - y[test].mean()) ** 2).sum()),
                         "spearman": float(spearmanr(y[test], prediction).statistic)})
            for index, value in zip(test, prediction, strict=True):
                predictions.append({"category": category, "target": target,
                                    "protocol": protocol, "seed": seed,
                                    "painting_id": ids[index],
                                    "observed": float(y[index]), "predicted": float(value)})
    return rows, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/extensions/heldout_ols")
    args = parser.parse_args()
    rows, predictions, audits = [], [], {}
    for category in ("abstract", "representational"):
        predictors, ids = load_predictors(args.repository, category)
        audits[category] = audit_join(args.repository, category, predictors)
        for target in ("beauty", "liking"):
            new_rows, new_predictions = fit_condition(args.repository, category, target,
                                                       predictors, ids, parse_range(args.seeds))
            rows.extend(new_rows)
            predictions.extend(new_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "metrics.csv"
    data = pd.DataFrame(rows)
    data.to_csv(output, index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False)
    data.groupby(["category", "target", "protocol"])[["mae", "r2", "spearman"]].mean().to_csv(
        args.output_dir / "summary.csv")
    sources = {str(p.relative_to(args.repository)): digest(p)
               for p in sorted((args.repository / "Data").glob("*.csv"))
               if p.name.endswith("_Data.csv") or p.name.endswith("_All_Raters.csv")}
    metadata = {"reconstructed_runner": True, "predictors": PREDICTORS,
                "rows": len(rows), "sha256": digest(output), "source_sha256": sources,
                "join_audit": audits, "seeds": parse_range(args.seeds),
                "split": "140 training / 20 validation / remainder test",
                "imputation": "feature median from assigned training paintings only",
                "complete_case": "filter within assigned partitions; never reshuffle"}
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(data.groupby(["category", "target", "protocol"]).spearman.mean().to_string())


if __name__ == "__main__":
    main()
