"""Validate one experiment CSV against its sidecar metadata and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    path: Path,
    expected_rows: int,
    expected_fields: dict[str, str] | None = None,
    manifest: Path | None = None,
    features: Path | None = None,
) -> None:
    expected_fields = expected_fields or {}
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = file_sha256(path)
    if digest != metadata.get("sha256"):
        raise ValueError(f"SHA-256 mismatch for {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows or metadata.get("rows") != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows in {path}, found {len(rows)}"
        )
    keys = ("objective", "N", "seed")
    missing = [key for key in keys if key not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if len({tuple(row[key] for key in keys) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate objective/N/seed rows in {path}")

    for field, expected in expected_fields.items():
        actual = {row.get(field) for row in rows}
        if actual != {expected}:
            raise ValueError(
                f"Expected {field}={expected!r} in {path}, found {sorted(actual)}"
            )

    for field in ("dataset", "representation", "category", "target", "mode", "rater"):
        if field not in rows[0]:
            continue
        actual = {row[field] for row in rows if field in row}
        if len(actual) != 1:
            raise ValueError(f"Expected one {field} value in {path}, found {actual}")
        if field in metadata and str(metadata[field]) not in actual:
            raise ValueError(f"CSV/metadata {field} mismatch for {path}")

    objectives = sorted({row["objective"] for row in rows})
    if "objectives" in metadata and sorted(metadata["objectives"]) != objectives:
        raise ValueError(f"CSV/metadata objective mismatch for {path}")
    seeds = sorted({int(row["seed"]) for row in rows})
    if "seeds" in metadata and sorted(metadata["seeds"]) != seeds:
        raise ValueError(f"CSV/metadata seed mismatch for {path}")
    n_values = sorted({int(float(row["N"])) for row in rows if row["N"]})
    if "n_values" in metadata and sorted(metadata["n_values"]) != n_values:
        raise ValueError(f"CSV/metadata N-value mismatch for {path}")

    for source, field in ((manifest, "manifest_sha256"), (features, "features_sha256")):
        if source is not None:
            expected_digest = file_sha256(source)
            if metadata.get(field) != expected_digest:
                raise ValueError(
                    f"{field} mismatch for {path}: expected {expected_digest}, "
                    f"found {metadata.get(field)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-dataset")
    parser.add_argument("--expected-representation")
    parser.add_argument("--expected-category")
    parser.add_argument("--expected-target")
    parser.add_argument("--expected-mode")
    parser.add_argument("--expected-rater", type=int)
    parser.add_argument("--expected-objective")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    expected_fields = {
        field.removeprefix("expected_"): str(value)
        for field, value in vars(args).items()
        if field.startswith("expected_")
        and field != "expected_rows"
        and value is not None
    }
    validate(
        args.path,
        args.expected_rows,
        expected_fields=expected_fields,
        manifest=args.manifest,
        features=args.features,
    )


if __name__ == "__main__":
    main()
