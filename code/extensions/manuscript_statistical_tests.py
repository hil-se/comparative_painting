"""Reproduce the matched-seed tests reported in the two manuscripts.

The implementation uses an exact two-sided Wilcoxon signed-rank permutation
test so the calculation has no optional SciPy dependency.
"""

from __future__ import annotations

import csv
import glob
import itertools
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read_rows(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in sorted(glob.glob(str(ROOT / pattern))):
        with open(name, newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


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
        rank = ((start + 1) + end) / 2
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def exact_wilcoxon(left: list[float], right: list[float]) -> tuple[float, float, int]:
    differences = [a - b for a, b in zip(left, right) if a != b]
    ranks = average_ranks([abs(value) for value in differences])
    observed = sum(rank for rank, value in zip(ranks, differences) if value > 0)
    total = sum(ranks)
    distance = abs(observed - total / 2)
    extreme = 0
    permutations = 1 << len(ranks)
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        positive = sum(rank for rank, sign in zip(ranks, signs) if sign)
        if abs(positive - total / 2) >= distance - 1e-12:
            extreme += 1
    return min(1.0, extreme / permutations), mean(differences), len(differences)


def holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=raw.__getitem__)
    adjusted = [0.0] * len(raw)
    running = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def aggregate(rows: list[dict[str, str]], selector) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = selector(row)
        if group is not None:
            values[group][int(row["seed"])].append(float(row["spearman"]))
    return {
        group: {seed: mean(entries) for seed, entries in seed_values.items()}
        for group, seed_values in values.items()
    }


def report_art(dataset: str, rows: list[dict[str, str]]) -> None:
    aggregated = aggregate(
        rows,
        lambda row: row["objective"]
        if row["representation"] == "clip-vit-b32"
        and (
            row["objective"] == "regression"
            or (row["N"] != "" and float(row["N"]) == 10)
        )
        else None,
    )
    comparisons = [
        ("Hinge - regression", "hinge", "regression"),
        ("BT - regression", "bradley_terry", "regression"),
        ("BT - hinge", "bradley_terry", "hinge"),
    ]
    raw = []
    details = []
    for label, left_name, right_name in comparisons:
        seeds = sorted(set(aggregated[left_name]) & set(aggregated[right_name]))
        left = [aggregated[left_name][seed] for seed in seeds]
        right = [aggregated[right_name][seed] for seed in seeds]
        p_value, delta, nonzero = exact_wilcoxon(left, right)
        raw.append(p_value)
        details.append((label, delta, p_value, nonzero))
    adjusted = holm(raw)
    print(dataset)
    for (label, delta, p_value, nonzero), corrected in zip(details, adjusted):
        print(
            f"  {label}: mean difference={delta:+.6f}, "
            f"Wilcoxon p={p_value:.6f}, Holm p={corrected:.6f}, n={nonzero}"
        )


def report_image() -> None:
    rows = read_rows("tmp/image-caption-extension-53692c0/*-vilbert.csv")
    tests = []
    for dataset in ("FlickrExpert", "VICR"):
        by_objective: dict[str, dict[int, float]] = defaultdict(dict)
        for row in rows:
            if row["dataset"] == dataset and row["pair_condition"] == "same_image":
                by_objective[row["objective"]][int(row["seed"])] = float(
                    row["same_image_accuracy"]
                )
        seeds = sorted(
            set(by_objective["bradley_terry"]) & set(by_objective["hinge"])
        )
        bt = [by_objective["bradley_terry"][seed] for seed in seeds]
        hinge = [by_objective["hinge"][seed] for seed in seeds]
        tests.append((dataset, *exact_wilcoxon(bt, hinge)))
    adjusted = holm([test[1] for test in tests])
    print("Image-caption same-image accuracy")
    for (dataset, p_value, delta, nonzero), corrected in zip(tests, adjusted):
        print(
            f"  {dataset}, BT - hinge: mean difference={delta:+.6f}, "
            f"Wilcoxon p={p_value:.6f}, Holm p={corrected:.6f}, n={nonzero}"
        )


def main() -> None:
    sidhu_regression = read_rows(
        "comparative_painting/results/extensions/locked_head/aggregate_merged/sidhu/"
        "sidhu-*-clip-vit-b32.csv"
    )
    sidhu_regression = [
        row for row in sidhu_regression if row["objective"] == "regression"
    ]
    sidhu_pairwise = read_rows(
        "comparative_painting/results/extensions/locked_head/aggregate_merged/sidhu/"
        "sidhu-*-clip-vit-b32-pairwise-v2.csv"
    )
    report_art("Art: Sidhu", sidhu_regression + sidhu_pairwise)
    report_art(
        "Art: APDDv2",
        read_rows(
            "comparative_painting/results/extensions/locked_head/aggregate_merged/apddv2/"
            "apddv2-*-clip-vit-b32.csv"
        ),
    )
    report_image()


if __name__ == "__main__":
    main()
