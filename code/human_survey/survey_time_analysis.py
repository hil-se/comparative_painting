"""Reproduce the paper's annotation times from the Qualtrics responses."""

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "Data/RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv"

# Beauty and liking occupy the same page, so they share page-submit times.
CONDITIONS = {
    "Abstract Beauty": {"direct": range(1, 6), "comparative": range(11, 16)},
    "Abstract Liking": {"direct": range(1, 6), "comparative": range(11, 16)},
    "Repr. Beauty": {"direct": range(6, 11), "comparative": range(16, 21)},
    "Repr. Liking": {"direct": range(6, 11), "comparative": range(16, 21)},
}


def load_and_filter(csv_path):
    """Keep finished, non-preview responses with nonconstant beauty ratings."""
    data = pd.read_csv(csv_path)
    data = data[(data.Finished == "TRUE") & (data.Status != "Survey Preview")].copy()
    keep = []
    for _, row in data.iterrows():
        ratings = [
            float(row[f"Q{q}_2"]) for q in range(1, 11) if pd.notna(row[f"Q{q}_2"])
        ]
        keep.append(len(set(ratings)) >= 2)
    return data[keep].copy()


def compute_per_rater_avg_time(data, questions):
    """Average page-submit times within each rater, then across raters."""
    rater_means = []
    for _, row in data.iterrows():
        times = [
            float(row[f"Q{q}_Time_Page Submit"])
            for q in questions
            if pd.notna(row[f"Q{q}_Time_Page Submit"])
        ]
        if times:
            rater_means.append(np.mean(times))
    return np.mean(rater_means)


def summarize_timing(csv_path):
    data = load_and_filter(csv_path)
    rows = []
    for condition, methods in CONDITIONS.items():
        for method, questions in methods.items():
            seconds = float(compute_per_rater_avg_time(data, questions))
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "seconds_raw": seconds,
                    "seconds": round(seconds, 2),
                }
            )
    for method in ("direct", "comparative"):
        values = [row for row in rows if row["method"] == method]
        paper_mean = sum(Decimal(str(row["seconds"])) for row in values) / Decimal(
            len(values)
        )
        rows.append(
            {
                "condition": "Overall",
                "method": method,
                "seconds_raw": np.mean([row["seconds_raw"] for row in values]),
                "seconds": float(
                    paper_mean.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_FILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/human_survey/paper/rq4_timing.csv",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summarize_timing(args.input).to_csv(args.output, index=False)
    print(f"Wrote annotation times to {args.output}")


if __name__ == "__main__":
    main()
