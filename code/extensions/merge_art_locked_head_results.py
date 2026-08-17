"""Merge and validate split files from the locked-head art reruns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


METHOD_ID = "shallow-mse-z-gelu-ln-rawclip"
OBJECTIVE_ORDER = ("regression", "hinge", "bradley_terry")
APDD_TARGET_SLUGS = {
    "Total aesthetic score": "total",
    "Theme and logic": "theme-logic",
    "Creativity": "creativity",
    "Layout and composition": "layout-composition",
    "Space and perspective": "space-perspective",
    "The sense of order": "order",
    "Light and shadow": "light-shadow",
    "Color": "color",
    "Details and texture": "details-texture",
    "The overall": "overall",
    "Mood": "mood",
}
CONSISTENT_METADATA_FIELDS = (
    "phase",
    "method_id",
    "method_selected_on",
    "method_retuned_on_evaluation_data",
    "config",
    "dataset",
    "representation",
    "category",
    "target",
    "mode",
    "rater",
    "manifest_sha256",
    "features_sha256",
    "tensorflow",
    "split",
    "feature_standardization",
    "regression_target_standardization",
    "selection_and_early_stopping_metric",
    "pairwise_validation_affine_calibration",
    "pairwise_loss_elementwise_shape_check",
)


def validated_source(path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("method_id") != METHOD_ID:
        raise ValueError(f"Unexpected method in {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata.get("sha256"):
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata.get("rows"):
        raise ValueError(f"Row-count mismatch for {path.name}")
    if {row["method_id"] for row in rows} != {METHOD_ID}:
        raise ValueError(f"CSV method mismatch for {path.name}")
    if len(
        {(row["objective"], row["N"], row["seed"]) for row in rows}
    ) != len(rows):
        raise ValueError(f"Duplicate objective/N/seed rows in {path.name}")
    return rows, metadata


def group_key(metadata: dict[str, object], mode: str) -> tuple[object, ...]:
    if mode == "sidhu":
        return (
            metadata["category"],
            metadata["target"],
            metadata["representation"],
        )
    if mode == "apddv2":
        return (metadata["target"], metadata["representation"])
    return (
        metadata["category"],
        metadata["target"],
        metadata["mode"],
        metadata["rater"],
    )


def output_name(
    key: tuple[object, ...], mode: str, *, pairwise_only: bool = False
) -> str:
    if mode == "sidhu":
        category, target, representation = key
        suffix = "-pairwise-v2" if pairwise_only else ""
        return f"sidhu-{category}-{target}-{representation}{suffix}.csv"
    if mode == "apddv2":
        target, representation = key
        return f"apddv2-{APDD_TARGET_SLUGS[str(target)]}-{representation}.csv"
    category, target, rater_mode, rater = key
    return (
        f"sidhu-{category}-{target}-{rater_mode}-r{rater}-"
        "clip-vit-b32.csv"
    )


def validate_metadata_consistency(
    name: str, source_metadata: list[dict[str, object]]
) -> None:
    for field in CONSISTENT_METADATA_FIELDS:
        values = {
            json.dumps(metadata[field], sort_keys=True)
            for metadata in source_metadata
            if field in metadata
        }
        if len(values) > 1:
            raise ValueError(f"Inconsistent {field} metadata for {name}")


def write_group(
    *,
    output: Path,
    rows: list[dict[str, str]],
    source_paths: list[Path],
    source_metadata: list[dict[str, object]],
) -> None:
    validate_metadata_consistency(output.name, source_metadata)
    if len(
        {(row["objective"], row["N"], row["seed"]) for row in rows}
    ) != len(rows):
        raise ValueError(f"Duplicate merged condition in {output.name}")
    rows.sort(
        key=lambda row: (
            OBJECTIVE_ORDER.index(row["objective"]),
            -1 if not row["N"] else int(float(row["N"])),
            int(row["seed"]),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(source_metadata[0])
    metadata.update(
        {
            "objectives": [
                objective
                for objective in OBJECTIVE_ORDER
                if any(row["objective"] == objective for row in rows)
            ],
            "n_values": sorted(
                {int(float(row["N"])) for row in rows if row["N"]}
            ),
            "seeds": sorted({int(row["seed"]) for row in rows}),
            "rows": len(rows),
            "source_count": len(source_paths),
            "merged_from": [path.name for path in source_paths],
            "source_sha256": [item["sha256"] for item in source_metadata],
        }
    )
    metadata["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("sidhu", "apddv2", "rater"), required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.csv"))
    expected_sources = {"sidhu": 240, "apddv2": 660, "rater": 1200}[args.mode]
    if len(paths) != expected_sources:
        raise ValueError(
            f"Expected {expected_sources} {args.mode} sources, found {len(paths)}"
        )

    groups: dict[tuple[object, ...], list[tuple[Path, list[dict[str, str]], dict[str, object]]]] = defaultdict(list)
    for path in paths:
        rows, metadata = validated_source(path)
        if metadata.get("dataset") != ("sidhu" if args.mode != "apddv2" else "apddv2"):
            raise ValueError(f"Dataset mismatch for {path.name}")
        groups[group_key(metadata, args.mode)].append((path, rows, metadata))

    expected_groups = {"sidhu": 8, "apddv2": 22, "rater": 40}[args.mode]
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} groups, found {len(groups)}")

    for key, sources in sorted(groups.items(), key=lambda item: str(item[0])):
        if args.mode == "sidhu":
            for pairwise_only in (False, True):
                selected = [
                    source
                    for source in sources
                    if (source[2]["objectives"][0] != "regression")
                    == pairwise_only
                ]
                rows = [row for _, source_rows, _ in selected for row in source_rows]
                expected_rows = 200 if pairwise_only else 10
                if len(rows) != expected_rows:
                    raise ValueError(
                        f"Expected {expected_rows} rows for {key}, pairwise={pairwise_only}"
                    )
                write_group(
                    output=args.output_dir
                    / output_name(key, args.mode, pairwise_only=pairwise_only),
                    rows=rows,
                    source_paths=[path for path, _, _ in selected],
                    source_metadata=[metadata for _, _, metadata in selected],
                )
        else:
            rows = [row for _, source_rows, _ in sources for row in source_rows]
            if len(rows) != 210:
                raise ValueError(f"Expected 210 rows for {key}, found {len(rows)}")
            write_group(
                output=args.output_dir / output_name(key, args.mode),
                rows=rows,
                source_paths=[path for path, _, _ in sources],
                source_metadata=[metadata for _, _, metadata in sources],
            )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "method_id": METHOD_ID,
                "source_files": len(paths),
                "groups": len(groups),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
