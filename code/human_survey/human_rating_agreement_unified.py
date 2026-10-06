"""Compute the paper's direct-rating and comparative accuracy matrices."""

import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd


def generate_pairs_absolute(x1, x2):
    """Generate pairwise preferences from absolute ratings"""
    n = len(x1)
    pairs = {"A": [], "B": [], "agree": []}
    for i in range(n):
        for j in range(i + 1, n):
            d1 = d2 = 0
            if x1[i] > x1[j]:
                d1 = 1
            elif x1[i] < x1[j]:
                d1 = -1
            if x2[i] > x2[j]:
                d2 = 1
            elif x2[i] < x2[j]:
                d2 = -1
            if d1 != 0 and d2 != 0:
                pairs["A"].append(d1)
                pairs["B"].append(d2)
                pairs["agree"].append(d1 == d2)
    return pairs


def generate_pairs_comparative(x1, x2):
    """Generate agreement pairs from comparative responses (A/B choices)"""
    n = len(x1)
    pairs = {"A": [], "B": [], "agree": []}
    for i in range(n):
        if x1.iloc[i] not in ("A", "B") or x2.iloc[i] not in ("A", "B"):
            continue
        choice1 = 1 if x1.iloc[i] == "A" else -1
        choice2 = 1 if x2.iloc[i] == "A" else -1
        pairs["A"].append(choice1)
        pairs["B"].append(choice2)
        pairs["agree"].append(choice1 == choice2)
    return pairs


def create_gt_comparative_column(df):
    """Create a GT column for comparative data"""
    candidates = [
        ("GT_A", "GT_B"),
        ("GT_A_Beauty", "GT_B_Beauty"),
        ("GT_A_Liking", "GT_B_Liking"),
    ]
    available = [(a, b) for a, b in candidates if a in df.columns and b in df.columns]
    column_a, column_b = available[0]

    def compare_gt(row):
        if row[column_a] > row[column_b]:
            return "A"
        elif row[column_b] > row[column_a]:
            return "B"
        else:
            return None

    df["GT"] = df.apply(compare_gt, axis=1)
    return df


def analyze_ratings(
    input_file, output_file, summary_file, rating_type="absolute", raters=None
):
    data = pd.read_csv(input_file)
    if rating_type == "comparative":
        data = create_gt_comparative_column(data)
    raters = ["GT", "P1", "P3", "P4", "P5", "P6"] if raters is None else raters
    rows = []
    scores = {rater: [] for rater in raters}
    generate = (
        generate_pairs_absolute
        if rating_type == "absolute"
        else generate_pairs_comparative
    )
    for left, right in itertools.combinations(raters, 2):
        accuracy = float(np.mean(generate(data[left], data[right])["agree"]))
        rows.append({"Pair": f"{left}/{right}", "Acc": f"{accuracy:.2f}"})
        if left == "GT":
            scores[left].append(accuracy)
        elif right == "GT":
            scores[right].append(accuracy)
        else:
            scores[left].append(accuracy)
            scores[right].append(accuracy)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, index=False)
    pd.DataFrame(
        [
            {"Rater": rater, "Avg_Acc": f"{np.mean(values):.3f}"}
            for rater, values in scores.items()
        ]
    ).to_csv(summary_file, index=False)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=root / "results/human_survey/survey_data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/human_survey/paper/rq4_matrices",
    )
    args = parser.parse_args()
    for condition in (
        "Abstract_Beauty",
        "Abstract_Liking",
        "Repr_Beauty",
        "Repr_Liking",
    ):
        for method in ("Absolute", "Comparative"):
            analyze_ratings(
                args.input_dir / f"{condition}_{method}.csv",
                args.output_dir / f"Agreement_{condition}_{method}.csv",
                args.output_dir / f"Summary_{condition}_{method}.csv",
                rating_type="absolute" if method == "Absolute" else "comparative",
            )


if __name__ == "__main__":
    main()
