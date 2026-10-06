"""Validate and summarize the controlled APDDv2 extension experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("pair_accuracy", "mae", "r2", "pearson", "spearman", "kendall")
HIGHER_IS_BETTER = {metric: metric != "mae" for metric in METRICS}
REPRESENTATIONS = ("resnet50", "clip-vit-b32")
PAIRWISE_OBJECTIVES = ("hinge", "bradley_terry")
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_SAMPLES = 20_000


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    return tuple(np.quantile(samples.mean(axis=1), (0.025, 0.975)))


def load_results(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(
        path
        for path in raw_dir.glob("apddv2-*.csv")
        if path.name != "apddv2-memory-validation.csv"
    )
    if len(paths) != 22:
        raise ValueError(f"Expected 22 APDDv2 result CSVs, found {len(paths)}")

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
        if len(frame) != 210 or metadata["rows"] != 210:
            raise ValueError(f"Expected 210 rows in {path.name}, found {len(frame)}")
        counts = frame["objective"].value_counts().to_dict()
        if counts != {"hinge": 100, "bradley_terry": 100, "regression": 10}:
            raise ValueError(f"Unexpected objective counts in {path.name}: {counts}")
        if frame[["objective", "N", "seed"]].duplicated().any():
            raise ValueError(f"Duplicate experimental condition in {path.name}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if len(data) != 4_620:
        raise ValueError(f"Expected 4,620 rows, found {len(data)}")
    if set(data["representation"]) != set(REPRESENTATIONS):
        raise ValueError("Representation labels are incomplete")
    if data[list(METRICS)].isna().any().any():
        raise ValueError("A reported metric contains missing values")
    data["N_key"] = data["N"].fillna(0).astype(int)
    return data


def paired_summary(
    data: pd.DataFrame,
    *,
    dimension: str,
    first: str,
    second: str,
    pair_keys: list[str],
    group_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []

    for group_values, group in data.groupby(group_columns, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_fields = dict(zip(group_columns, group_values))
        for metric in METRICS:
            pivot = group.pivot(
                index=pair_keys,
                columns=dimension,
                values=metric,
            )
            if pivot[[first, second]].isna().any().any():
                raise ValueError(
                    f"Unpaired {dimension} values for {group_fields}, metric={metric}"
                )
            target_means = (
                pivot[[first, second]].reset_index().groupby("target")[[first, second]].mean()
            )
            target_means["difference"] = target_means[second] - target_means[first]
            low, high = bootstrap_interval(target_means["difference"].to_numpy())
            wins = (
                target_means["difference"] > 0
                if HIGHER_IS_BETTER[metric]
                else target_means["difference"] < 0
            )
            summary_rows.append(
                {
                    **group_fields,
                    "metric": metric,
                    f"mean_{first}": target_means[first].mean(),
                    f"mean_{second}": target_means[second].mean(),
                    f"difference_{second}_minus_{first}": target_means[
                        "difference"
                    ].mean(),
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                    f"{second}_wins_targets": int(wins.sum()),
                    "target_count": len(target_means),
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                }
            )
            for target, row in target_means.iterrows():
                target_rows.append(
                    {
                        **group_fields,
                        "target": target,
                        "metric": metric,
                        f"mean_{first}": row[first],
                        f"mean_{second}": row[second],
                        f"difference_{second}_minus_{first}": row["difference"],
                        f"{second}_wins": bool(wins.loc[target]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(target_rows)


def lookup(frame: pd.DataFrame, **values: object) -> pd.Series:
    selected = frame
    for column, value in values.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {values}, found {len(selected)}")
    return selected.iloc[0]


def compare_pairwise_with_regression(data: pd.DataFrame) -> pd.DataFrame:
    regression = data[data["objective"] == "regression"][
        ["target", "representation", "seed", *METRICS]
    ]
    pairwise = data[data["objective"].isin(PAIRWISE_OBJECTIVES)][
        ["target", "representation", "objective", "N_key", "seed", *METRICS]
    ]
    merged = pairwise.merge(
        regression,
        on=["target", "representation", "seed"],
        validate="many_to_one",
        suffixes=("_pairwise", "_regression"),
    )
    rows: list[dict[str, object]] = []
    for (representation, objective, n), group in merged.groupby(
        ["representation", "objective", "N_key"], sort=False
    ):
        for metric in METRICS:
            target_means = group.groupby("target")[[
                f"{metric}_regression",
                f"{metric}_pairwise",
            ]].mean()
            target_means["difference"] = (
                target_means[f"{metric}_pairwise"]
                - target_means[f"{metric}_regression"]
            )
            low, high = bootstrap_interval(target_means["difference"].to_numpy())
            wins = (
                target_means["difference"] > 0
                if HIGHER_IS_BETTER[metric]
                else target_means["difference"] < 0
            )
            rows.append(
                {
                    "representation": representation,
                    "objective": objective,
                    "N": int(n),
                    "metric": metric,
                    "mean_regression": target_means[f"{metric}_regression"].mean(),
                    "mean_pairwise": target_means[f"{metric}_pairwise"].mean(),
                    "difference_pairwise_minus_regression": target_means[
                        "difference"
                    ].mean(),
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                    "pairwise_wins_targets": int(wins.sum()),
                    "target_count": len(target_means),
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                }
            )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(
    output: Path,
    representation: pd.DataFrame,
    representation_targets: pd.DataFrame,
    objectives: pd.DataFrame,
    n_curve: pd.DataFrame,
    paradigms: pd.DataFrame,
) -> None:
    lines = [
        "# APDDv2 extension results",
        "",
        "## Validation and analysis unit",
        "",
        "All 22 result files passed SHA-256 verification against their metadata. "
        "Each file contains 210 conditions (10 regression, 100 hinge, and 100 "
        "Bradley-Terry), for 4,620 fitted models total. Comparisons are paired on "
        "target, seed, split, and N where applicable. Confidence intervals are "
        "nonparametric 95% bootstrap intervals over the 11 target-level mean "
        "differences (20,000 resamples); seeds and N values are repeated measures, "
        "not independent samples.",
        "",
        "## CLIP versus ResNet-50",
        "",
        "Positive differences favor CLIP for the correlation and pair-accuracy "
        "metrics. Pairwise rows average N=1 through N=10 before targets are "
        "compared.",
        "",
        "| Objective | Metric | ResNet | CLIP | CLIP − ResNet (95% CI) | CLIP wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for objective in ("regression", "hinge", "bradley_terry"):
        for metric in ("spearman", "pair_accuracy"):
            row = lookup(representation, objective=objective, metric=metric)
            diff = row["difference_clip-vit-b32_minus_resnet50"]
            lines.append(
                f"| {objective.replace('_', ' ').title()} | {metric.replace('_', ' ')} "
                f"| {fmt(row['mean_resnet50'])} | {fmt(row['mean_clip-vit-b32'])} "
                f"| {fmt(diff)} [{fmt(row['difference_ci95_low'])}, "
                f"{fmt(row['difference_ci95_high'])}] | "
                f"{int(row['clip-vit-b32_wins_targets'])}/11 |"
            )

    regression_targets = representation_targets[
        (representation_targets["objective"] == "regression")
        & (representation_targets["metric"] == "spearman")
    ].sort_values("difference_clip-vit-b32_minus_resnet50", ascending=False)
    lines.extend(
        [
            "",
            "Regression Spearman by target:",
            "",
            "| Target | ResNet | CLIP | Difference |",
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
            "Positive differences favor Bradley-Terry. Results average N=1 "
            "through N=10 within each target before target-level comparison.",
            "",
            "| Representation | Metric | Hinge | Bradley-Terry | BT − hinge (95% CI) | BT wins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for representation_name in REPRESENTATIONS:
        for metric in ("spearman", "pair_accuracy"):
            row = lookup(objectives, representation=representation_name, metric=metric)
            diff = row["difference_bradley_terry_minus_hinge"]
            lines.append(
                f"| {representation_name} | {metric.replace('_', ' ')} | "
                f"{fmt(row['mean_hinge'])} | {fmt(row['mean_bradley_terry'])} | "
                f"{fmt(diff)} [{fmt(row['difference_ci95_low'])}, "
                f"{fmt(row['difference_ci95_high'])}] | "
                f"{int(row['bradley_terry_wins_targets'])}/11 |"
            )

    lines.extend(
        [
            "",
            "## Pairwise training versus regression",
            "",
            "N=10 is shown because it is the largest comparison budget tested. "
            "Positive differences favor pairwise training.",
            "",
            "| Representation | Pairwise loss | Metric | Regression | Pairwise N=10 | Difference (95% CI) | Pairwise wins |",
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
                    f"[{fmt(row['difference_ci95_low'])}, "
                    f"{fmt(row['difference_ci95_high'])}] | "
                    f"{int(row['pairwise_wins_targets'])}/11 |"
                )

    lines.extend(
        [
            "",
            "## Effect of comparisons per item",
            "",
            "These are descriptive grand means over targets and seeds.",
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
                f"{fmt(first['pair_accuracy_mean'])} → "
                f"{fmt(last['pair_accuracy_mean'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The primary evidence is the paired target-level difference, not the "
            "raw count of 4,620 fitted models. Intervals crossing zero indicate "
            "that the direction is not consistent enough across the 11 APDDv2 "
            "attributes to claim a general advantage. MAE should not be averaged "
            "as a cross-target headline because APDDv2 attributes use different "
            "numeric scales; rank correlation and pair accuracy are comparable "
            "across targets.",
            "",
        ]
    )
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

    by_target = (
        data.groupby(["target", "representation", "objective"], as_index=False)[
            list(METRICS)
        ]
        .agg(["mean", "std"])
    )
    by_target.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in by_target.columns
    ]
    by_target.to_csv(args.output_dir / "summary_by_target.csv", index=False)

    representation, representation_targets = paired_summary(
        data,
        dimension="representation",
        first="resnet50",
        second="clip-vit-b32",
        pair_keys=["target", "objective", "N_key", "seed"],
        group_columns=["objective"],
    )
    representation.to_csv(
        args.output_dir / "representation_comparison.csv", index=False
    )
    representation_targets.to_csv(
        args.output_dir / "representation_comparison_by_target.csv", index=False
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
        args.output_dir / "objective_comparison_by_target.csv", index=False
    )

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

    paradigms = compare_pairwise_with_regression(data)
    paradigms.to_csv(
        args.output_dir / "paradigm_comparison.csv", index=False
    )

    write_report(
        args.output_dir / "analysis.md",
        representation,
        representation_targets,
        objectives,
        n_curve,
        paradigms,
    )
    print(
        json.dumps(
            {
                "files": 22,
                "rows": len(data),
                "targets": data["target"].nunique(),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
