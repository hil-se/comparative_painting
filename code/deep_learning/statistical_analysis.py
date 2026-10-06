"""Reproduce all matched-seed tests for the Art Eval manuscript.

The implementation uses an exact two-sided Wilcoxon signed-rank permutation
test so the calculation has no optional SciPy dependency.
"""

from __future__ import annotations
import argparse
import csv
import math
import itertools
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def exact_wilcoxon(left: list[float], right: list[float]) -> tuple[float, float, int]:
    all_differences = [a - b for a, b in zip(left, right)]
    differences = [value for value in all_differences if value != 0]
    if not differences:
        return (1.0, 0.0, 0)
    ranks = average_ranks([abs(value) for value in differences])
    observed = sum((rank for rank, value in zip(ranks, differences) if value > 0))
    total = sum(ranks)
    distance = abs(observed - total / 2)
    extreme = 0
    permutations = 1 << len(ranks)
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        positive = sum((rank for rank, sign in zip(ranks, signs) if sign))
        if abs(positive - total / 2) >= distance - 1e-12:
            extreme += 1
    return (min(1.0, extreme / permutations), mean(all_differences), len(differences))


def holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=raw.__getitem__)
    adjusted = [0.0] * len(raw)
    running = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def matched_test(label: str, left: dict[int, float], right: dict[int, float]) -> dict:
    p_value, delta, nonzero = exact_wilcoxon(
        [left[i] for i in range(10)], [right[i] for i in range(10)]
    )
    return {
        "contrast": label,
        "mean_difference": delta,
        "p_raw": p_value,
        "n_seeds": 10,
        "n_nonzero": nonzero,
    }


def correct_family(tests: list[dict]) -> list[dict]:
    for test, adjusted in zip(tests, holm([t["p_raw"] for t in tests])):
        test["p_holm"] = adjusted
    return tests


def selected_seeds(
    rows: list[dict],
    representation: str,
    objective: str,
    n: int = 0,
    category: str | None = None,
    target: str | None = None,
) -> dict[int, float]:
    selected = [
        r
        for r in rows
        if r["representation"] == representation
        and r["objective"] == objective
        and (objective == "regression" or int(float(r["N"])) == n)
        and (category is None or r["category"] == category)
        and (target is None or r["target"] == target)
    ]
    by_seed = defaultdict(list)
    for row in selected:
        by_seed[int(row["seed"])].append(float(row["spearman"]))
    return {seed: mean(values) for seed, values in by_seed.items()}


def compute_tests(
    sidhu: list[dict], apdd: list[dict], ols: list[dict]
) -> dict[str, list[dict]]:
    rq1 = []
    for category in ("abstract", "representational"):
        for target in ("beauty", "liking"):
            clip = selected_seeds(
                sidhu, "clip-vit-b32", "regression", category=category, target=target
            )
            resnet = selected_seeds(
                sidhu, "resnet50", "regression", category=category, target=target
            )
            label = f"Sidhu {category} {target}"
            rq1.append(matched_test(label + ": CLIP - ResNet", clip, resnet))
            baseline_rows = [
                r
                for r in ols
                if r["category"] == category
                and r["target"] == target
                and (r["protocol"] == "training_median")
            ]
            baseline = {int(r["seed"]): float(r["spearman"]) for r in baseline_rows}
            rq1.append(matched_test(label + ": CLIP - OLS", clip, baseline))
    rq1.append(
        matched_test(
            "APDDv2 macro: CLIP - ResNet",
            selected_seeds(apdd, "clip-vit-b32", "regression"),
            selected_seeds(apdd, "resnet50", "regression"),
        )
    )
    output = {
        "rq1_tests": correct_family(rq1),
        "rq2_focal_tests": [],
        "rq2_sweep_tests": [],
    }
    for dataset, rows in (("sidhu", sidhu), ("apddv2", apdd)):
        regression = selected_seeds(rows, "clip-vit-b32", "regression")
        sweep = []
        for n in range(1, 11):
            hinge = selected_seeds(rows, "clip-vit-b32", "hinge", n)
            bt = selected_seeds(rows, "clip-vit-b32", "bradley_terry", n)
            contrasts = [
                ("hinge - regression", hinge, regression),
                ("BT - regression", bt, regression),
                ("BT - hinge", bt, hinge),
            ]
            family = [
                dict(matched_test(label, left, right), dataset=dataset, N=n)
                for label, left, right in contrasts
            ]
            if n in (1, 10):
                output["rq2_focal_tests"].extend(correct_family(family))
            sweep.extend((dict(t) for t in family[:2]))
        output["rq2_sweep_tests"].extend(correct_family(sweep))
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results/deep_learning"
    )
    parser.add_argument(
        "--ols-results",
        type=Path,
        default=ROOT / "results/baseline/heldout_ols/metrics.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/deep_learning/paper"
    )
    args = parser.parse_args()
    merged = args.results_root / "aggregate"

    def read_directory(path):
        rows = []
        for source in sorted(path.glob("*.csv")):
            with source.open(newline="", encoding="utf-8") as stream:
                rows.extend(csv.DictReader(stream))
        return rows

    with args.ols_results.open(newline="", encoding="utf-8") as stream:
        ols = list(csv.DictReader(stream))
    output = compute_tests(
        read_directory(merged / "sidhu"), read_directory(merged / "apddv2"), ols
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in output.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    print(
        f"Wrote {sum(map(len, output.values()))} tests in three CSVs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
