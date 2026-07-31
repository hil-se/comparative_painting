"""Validate and summarize the corrected Sidhu extension experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from summarize_apddv2_extensions import (
    METRICS,
    PAIRWISE_OBJECTIVES,
    REPRESENTATIONS,
    compare_pairwise_with_regression,
    fmt,
    lookup,
    paired_summary,
)


def load_results(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("sidhu-*.csv"))
    if len(paths) != 16:
        raise ValueError(f"Expected 16 Sidhu CSVs, found {len(paths)}")

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
        if "-pairwise-v2" in path.name:
            if len(frame) != 200 or set(frame["objective"]) != set(
                PAIRWISE_OBJECTIVES
            ):
                raise ValueError(f"Invalid corrected pairwise file: {path.name}")
            selected = frame
        else:
            if len(frame) != 210:
                raise ValueError(f"Expected 210 rows in {path.name}")
            # The original pairwise rows predate the elementwise loss-shape fix.
            # Only their unaffected regression rows are valid.
            selected = frame[frame["objective"] == "regression"].copy()
            if len(selected) != 10:
                raise ValueError(f"Expected 10 regression rows in {path.name}")
        frames.append(selected)

    data = pd.concat(frames, ignore_index=True)
    if len(data) != 1_680:
        raise ValueError(f"Expected 1,680 valid rows, found {len(data)}")
    if data[["category", "target", "representation", "objective", "N", "seed"]].duplicated().any():
        raise ValueError("Duplicate Sidhu experimental condition")
    if data[list(METRICS)].isna().any().any():
        raise ValueError("A reported metric contains missing values")
    data["rating"] = data["target"]
    data["target"] = data["category"].str.title() + " " + data["rating"].str.title()
    data["N_key"] = data["N"].fillna(0).astype(int)
    return data


def write_report(
    output: Path,
    representation: pd.DataFrame,
    representation_targets: pd.DataFrame,
    objectives: pd.DataFrame,
    paradigms: pd.DataFrame,
    n_curve: pd.DataFrame,
) -> None:
    lines = [
        "# Corrected Sidhu extension results",
        "",
        "## Validation and analysis unit",
        "",
        "All 16 source files passed SHA-256 verification. The analysis uses the "
        "80 unaffected regression fits from the first run and the 1,600 corrected "
        "pairwise fits from the v2 rerun. The 1,600 pairwise rows from the first "
        "run are excluded because the Keras label/prediction rank mismatch caused "
        "cross-pair broadcasting. Comparisons are paired on condition, seed, split, "
        "and N. Confidence intervals bootstrap the four condition-level mean "
        "differences (20,000 resamples).",
        "",
        "## CLIP versus ResNet-50",
        "",
        "| Objective | Metric | ResNet | CLIP | CLIP − ResNet (95% CI) | CLIP wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for objective in ("regression", "hinge", "bradley_terry"):
        for metric in ("spearman", "pair_accuracy"):
            row = lookup(representation, objective=objective, metric=metric)
            lines.append(
                f"| {objective.replace('_', ' ').title()} | {metric.replace('_', ' ')} | "
                f"{fmt(row['mean_resnet50'])} | {fmt(row['mean_clip-vit-b32'])} | "
                f"{fmt(row['difference_clip-vit-b32_minus_resnet50'])} "
                f"[{fmt(row['difference_ci95_low'])}, {fmt(row['difference_ci95_high'])}] | "
                f"{int(row['clip-vit-b32_wins_targets'])}/4 |"
            )

    regression_targets = representation_targets[
        (representation_targets["objective"] == "regression")
        & (representation_targets["metric"] == "spearman")
    ].sort_values("difference_clip-vit-b32_minus_resnet50", ascending=False)
    lines.extend(
        [
            "",
            "Regression Spearman by condition:",
            "",
            "| Condition | ResNet | CLIP | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in regression_targets.iterrows():
        lines.append(
            f"| {row['target']} | {fmt(row['mean_resnet50'])} | "
            f"{fmt(row['mean_clip-vit-b32'])} | "
            f"{fmt(row['difference_clip-vit-b32_minus_resnet50'])} |"
        )

    lines.extend(
        [
            "",
            "## Bradley-Terry versus hinge",
            "",
            "| Representation | Metric | Hinge | Bradley-Terry | BT − hinge (95% CI) | BT wins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for representation_name in REPRESENTATIONS:
        for metric in ("spearman", "pair_accuracy"):
            row = lookup(objectives, representation=representation_name, metric=metric)
            lines.append(
                f"| {representation_name} | {metric.replace('_', ' ')} | "
                f"{fmt(row['mean_hinge'])} | {fmt(row['mean_bradley_terry'])} | "
                f"{fmt(row['difference_bradley_terry_minus_hinge'])} "
                f"[{fmt(row['difference_ci95_low'])}, {fmt(row['difference_ci95_high'])}] | "
                f"{int(row['bradley_terry_wins_targets'])}/4 |"
            )

    lines.extend(
        [
            "",
            "## Pairwise training versus regression at N=10",
            "",
            "| Representation | Pairwise loss | Metric | Regression | Pairwise | Difference (95% CI) | Pairwise wins |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for representation_name in REPRESENTATIONS:
        for objective in PAIRWISE_OBJECTIVES:
            for metric in ("spearman", "pair_accuracy"):
                row = lookup(
                    paradigms,
                    representation=representation_name,
                    objective=objective,
                    N=10,
                    metric=metric,
                )
                lines.append(
                    f"| {representation_name} | {objective.replace('_', ' ')} | "
                    f"{metric.replace('_', ' ')} | {fmt(row['mean_regression'])} | "
                    f"{fmt(row['mean_pairwise'])} | "
                    f"{fmt(row['difference_pairwise_minus_regression'])} "
                    f"[{fmt(row['difference_ci95_low'])}, {fmt(row['difference_ci95_high'])}] | "
                    f"{int(row['pairwise_wins_targets'])}/4 |"
                )

    lines.extend(
        [
            "",
            "## Effect of comparisons per item",
            "",
            "| Representation | Objective | Spearman N=1 → N=10 | Pair accuracy N=1 → N=10 |",
            "|---|---|---:|---:|",
        ]
    )
    for representation_name in REPRESENTATIONS:
        for objective in PAIRWISE_OBJECTIVES:
            first = lookup(
                n_curve,
                representation=representation_name,
                objective=objective,
                N=1,
            )
            last = lookup(
                n_curve,
                representation=representation_name,
                objective=objective,
                N=10,
            )
            lines.append(
                f"| {representation_name} | {objective.replace('_', ' ')} | "
                f"{fmt(first['spearman_mean'])} → {fmt(last['spearman_mean'])} | "
                f"{fmt(first['pair_accuracy_mean'])} → {fmt(last['pair_accuracy_mean'])} |"
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

    representation, representation_targets = paired_summary(
        data,
        dimension="representation",
        first="resnet50",
        second="clip-vit-b32",
        pair_keys=["target", "objective", "N_key", "seed"],
        group_columns=["objective"],
    )
    representation.to_csv(args.output_dir / "representation_comparison.csv", index=False)
    representation_targets.to_csv(
        args.output_dir / "representation_comparison_by_condition.csv", index=False
    )

    pairwise = data[data["objective"].isin(PAIRWISE_OBJECTIVES)]
    objectives, objective_targets = paired_summary(
        pairwise,
        dimension="objective",
        first="hinge",
        second="bradley_terry",
        pair_keys=["target", "representation", "N_key", "seed"],
        group_columns=["representation"],
    )
    objectives.to_csv(args.output_dir / "objective_comparison.csv", index=False)
    objective_targets.to_csv(
        args.output_dir / "objective_comparison_by_condition.csv", index=False
    )

    paradigms = compare_pairwise_with_regression(data)
    paradigms.to_csv(args.output_dir / "paradigm_comparison.csv", index=False)
    n_curve = (
        pairwise.groupby(["representation", "objective", "N"], as_index=False)[
            list(METRICS)
        ]
        .agg(["mean", "std"])
    )
    n_curve.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in n_curve.columns
    ]
    n_curve["N"] = n_curve["N"].astype(int)
    n_curve.to_csv(args.output_dir / "n_curve.csv", index=False)

    write_report(
        args.output_dir / "analysis.md",
        representation,
        representation_targets,
        objectives,
        paradigms,
        n_curve,
    )
    print(
        json.dumps(
            {
                "source_files": 16,
                "valid_rows": len(data),
                "conditions": data["target"].nunique(),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
