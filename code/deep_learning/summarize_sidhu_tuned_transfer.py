"""Compare the locked tuned regressor with existing Sidhu regression results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


CONDITIONS = (
    ("abstract", "beauty"),
    ("abstract", "liking"),
    ("representational", "beauty"),
    ("representational", "liking"),
)
METRICS = (
    ("pair_accuracy", "test_pair_accuracy", True),
    ("mae", "test_mae", False),
    ("mse", "test_mse", False),
    ("r2", "test_r2", True),
    ("pearson", "test_pearson", True),
    ("spearman", "test_spearman", True),
    ("kendall", "test_kendall", True),
)


def load_old(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for file in sorted(path.glob("sidhu-*-clip-vit-b32.csv")):
        with file.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["objective"] != "regression":
                    continue
                key = (row["category"], row["target"], int(row["seed"]))
                rows[key] = {old_name: float(row[old_name]) for old_name, _, _ in METRICS}
    return rows


def load_new(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for file in sorted(path.glob("*.csv")):
        with file.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["category"], row["target"], int(row["seed"]))
                rows[key] = {
                    old_name: float(row[new_name])
                    for old_name, new_name, _ in METRICS
                }
    return rows


def expected_keys() -> set[tuple[str, str, int]]:
    return {
        (category, target, seed)
        for category, target in CONDITIONS
        for seed in range(10)
    }


def mean_for(
    rows: dict[tuple[str, str, int], dict[str, float]],
    metric: str,
    category: str | None = None,
    target: str | None = None,
    seed: int | None = None,
) -> float:
    values = [
        metrics[metric]
        for (row_category, row_target, row_seed), metrics in rows.items()
        if (category is None or category == row_category)
        and (target is None or target == row_target)
        and (seed is None or seed == row_seed)
    ]
    return float(np.mean(values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old = load_old(args.old_dir)
    new = load_new(args.new_dir)
    expected = expected_keys()
    if set(old) != expected:
        raise ValueError(f"Existing results mismatch: missing={sorted(expected-set(old))[:5]}")
    if set(new) != expected:
        raise ValueError(f"Tuned results mismatch: missing={sorted(expected-set(new))[:5]}")

    overall: list[dict[str, object]] = []
    by_condition: list[dict[str, object]] = []
    for metric, _, higher_is_better in METRICS:
        old_mean = mean_for(old, metric)
        new_mean = mean_for(new, metric)
        wins = 0
        for category, target in CONDITIONS:
            condition_old = mean_for(old, metric, category, target)
            condition_new = mean_for(new, metric, category, target)
            improved = (
                condition_new > condition_old
                if higher_is_better
                else condition_new < condition_old
            )
            wins += int(improved)
            by_condition.append(
                {
                    "category": category,
                    "target": target,
                    "metric": metric,
                    "previous_mean": condition_old,
                    "tuned_mean": condition_new,
                    "difference_tuned_minus_previous": condition_new - condition_old,
                    "tuned_improves": improved,
                }
            )
        overall.append(
            {
                "metric": metric,
                "previous_mean": old_mean,
                "tuned_mean": new_mean,
                "difference_tuned_minus_previous": new_mean - old_mean,
                "conditions_improved": wins,
                "condition_count": len(CONDITIONS),
                "higher_is_better": higher_is_better,
            }
        )

    seed_rows: list[dict[str, object]] = []
    previous_seed_spearman = []
    tuned_seed_spearman = []
    for seed in range(10):
        previous = mean_for(old, "spearman", seed=seed)
        tuned = mean_for(new, "spearman", seed=seed)
        previous_seed_spearman.append(previous)
        tuned_seed_spearman.append(tuned)
        seed_rows.append(
            {
                "seed": seed,
                "previous_macro_spearman": previous,
                "tuned_macro_spearman": tuned,
                "difference": tuned - previous,
            }
        )

    paired_t = ttest_rel(tuned_seed_spearman, previous_seed_spearman)
    paired_wilcoxon = wilcoxon(
        tuned_seed_spearman,
        previous_seed_spearman,
        alternative="greater",
    )
    spearman_row = next(row for row in overall if row["metric"] == "spearman")
    summary = {
        "comparison": "APDDv2-locked tuned model versus previous Sidhu model",
        "configuration_retuned_on_sidhu": False,
        "conditions": 4,
        "seeds": 10,
        "previous_macro_spearman": spearman_row["previous_mean"],
        "tuned_macro_spearman": spearman_row["tuned_mean"],
        "difference": spearman_row["difference_tuned_minus_previous"],
        "conditions_improved_spearman": spearman_row["conditions_improved"],
        "seeds_improved_spearman": sum(
            row["difference"] > 0 for row in seed_rows
        ),
        "paired_seed_t_statistic": float(paired_t.statistic),
        "paired_seed_t_pvalue_two_sided": float(paired_t.pvalue),
        "paired_seed_wilcoxon_statistic": float(paired_wilcoxon.statistic),
        "paired_seed_wilcoxon_pvalue_one_sided": float(paired_wilcoxon.pvalue),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall_comparison.csv", overall)
    write_csv(args.output_dir / "by_condition.csv", by_condition)
    write_csv(args.output_dir / "seed_macro_spearman.csv", seed_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
