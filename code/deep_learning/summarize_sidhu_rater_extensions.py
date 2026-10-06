"""Validate and summarize the Sidhu CLIP rater-level experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


OBJECTIVES = ("regression", "hinge", "bradley_terry")
MODES = ("within", "cross")


def load_results(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("sidhu-*.csv"))
    if len(paths) != 40:
        raise ValueError(f"Expected 40 Sidhu rater CSVs, found {len(paths)}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        metadata_path = path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {path.name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path.name}")

        frame = pd.read_csv(path)
        if len(frame) != 210:
            raise ValueError(f"Expected 210 rows in {path.name}")
        counts = frame["objective"].value_counts().to_dict()
        if counts != {"hinge": 100, "bradley_terry": 100, "regression": 10}:
            raise ValueError(f"Unexpected objective counts in {path.name}: {counts}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    keys = ["category", "target", "mode", "rater", "objective", "N", "seed"]
    if data[keys].duplicated().any():
        raise ValueError("Duplicate Sidhu rater experimental condition")
    if data[["pearson", "spearman"]].isna().any().any():
        raise ValueError("A reported rater metric contains missing values")
    data["task"] = (
        data["category"].str.title() + " " + data["target"].str.title()
    )
    data["N_key"] = data["N"].fillna(0).astype(int)
    return data


def write_report(output: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Sidhu CLIP rater-extension analysis",
        "",
        "## Integrity",
        "",
        "- The result set contains 40 CSV files and 40 metadata files.",
        "- Each CSV contains 210 unique objective/N/seed rows: 10 regression, "
        "100 hinge, and 100 Bradley-Terry fits.",
        "- All 8,400 rows match the SHA-256 digests recorded in their metadata.",
        "- The design covers four tasks, within-rater and cross-rater protocols, "
        "five target raters, N=1 through N=10, and ten matched seeds.",
        "",
        "## N=1 results",
        "",
        "Values pool five target raters and ten seeds (50 evaluations per entry).",
        "",
        "| Task | Objective | Within Pearson | Within Spearman | Cross Pearson | Cross Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in (
        "Abstract Beauty",
        "Abstract Liking",
        "Representational Beauty",
        "Representational Liking",
    ):
        for objective in OBJECTIVES:
            n_key = 0 if objective == "regression" else 1
            row = summary[
                (summary["task"] == task)
                & (summary["objective"] == objective)
                & (summary["N"] == n_key)
            ]
            values = {
                mode: row[row["mode"] == mode].iloc[0] for mode in MODES
            }
            label = objective.replace("_", " ").title()
            lines.append(
                f"| {task} | {label} | {values['within']['pearson']:.3f} | "
                f"{values['within']['spearman']:.3f} | "
                f"{values['cross']['pearson']:.3f} | "
                f"{values['cross']['spearman']:.3f} |"
            )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_results(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        data.groupby(["task", "category", "target", "mode", "objective", "N_key"])[
            ["pearson", "spearman", "pair_accuracy"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"N_key": "N"})
    )
    summary.to_csv(args.output_dir / "summary_by_budget.csv", index=False)

    n_curve = (
        data[data["objective"].isin(("hinge", "bradley_terry"))]
        .groupby(["mode", "objective", "N"], as_index=False)["spearman"]
        .agg(["mean", "std"])
    )
    n_curve["N"] = n_curve["N"].astype(int)
    n_curve.to_csv(args.output_dir / "n_curve.csv", index=False)
    write_report(args.output_dir / "analysis.md", summary)
    print(
        json.dumps(
            {
                "source_files": 40,
                "valid_rows": len(data),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
