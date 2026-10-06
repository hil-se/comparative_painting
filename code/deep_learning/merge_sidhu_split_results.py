"""Merge objective-split Sidhu result files into the canonical result layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


OBJECTIVE_ORDER = ("regression", "hinge", "bradley_terry")
CONSISTENT_METADATA_FIELDS = (
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
    "training_only_standardization",
)


def validated_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata["rows"]:
        raise ValueError(f"Row-count mismatch for {path.name}")
    return rows, metadata


def output_name(path: Path, mode: str) -> str:
    if mode == "pairwise":
        match = re.fullmatch(
            r"(.+-pairwise-v2)-(?:hinge|bradley_terry)-s\d+\.csv",
            path.name,
        )
    else:
        match = re.fullmatch(
            r"(.+-clip-vit-b32)-(?:regression|hinge|bradley_terry)-s\d+\.csv",
            path.name,
        )
    if match:
        return match.group(1) + ".csv"
    raise ValueError(f"Unexpected split filename: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pairwise", "rater"), required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("sidhu-*.csv"))
    expected_sources = 160 if args.mode == "pairwise" else 1200
    if len(paths) != expected_sources:
        raise ValueError(
            f"Expected {expected_sources} split CSVs, found {len(paths)}"
        )
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[output_name(path, args.mode)].append(path)
    expected_groups = 8 if args.mode == "pairwise" else 40
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} groups, found {len(groups)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, group_paths in sorted(groups.items()):
        rows: list[dict[str, str]] = []
        source_metadata: list[dict[str, object]] = []
        for path in sorted(group_paths):
            source_rows, metadata = validated_rows(path)
            rows.extend(source_rows)
            source_metadata.append(metadata)
        expected_rows = 200 if args.mode == "pairwise" else 210
        if len(rows) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows for {name}")
        keys = ["objective", "N", "seed"]
        if len({tuple(row[key] for key in keys) for row in rows}) != len(rows):
            raise ValueError(f"Duplicate objective/N/seed rows for {name}")
        for field in CONSISTENT_METADATA_FIELDS:
            values = {
                json.dumps(metadata[field], sort_keys=True)
                for metadata in source_metadata
                if field in metadata
            }
            if len(values) > 1:
                raise ValueError(f"Inconsistent {field} metadata for {name}")
        rows.sort(
            key=lambda row: (
                OBJECTIVE_ORDER.index(row["objective"]),
                -1 if not row["N"] else int(float(row["N"])),
                int(row["seed"]),
            )
        )

        output = args.output_dir / name
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
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "source_count": len(group_paths),
                "merged_from": [path.name for path in sorted(group_paths)],
                "source_sha256": [item["sha256"] for item in source_metadata],
            }
        )
        output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"groups": len(groups), "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
