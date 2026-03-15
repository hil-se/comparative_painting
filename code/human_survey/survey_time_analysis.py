"""
Survey Time Analysis: Direct Ratings vs Comparative Judgments.

Part of RQ4 validation for the IEEE Access paper "Modeling Art Evaluations
from Comparative Judgments." This script produces the annotation time
comparison reported in Table 6 of the paper, demonstrating that comparative
judgments are significantly faster than direct (absolute) ratings.

Pipeline:
    1. Load the raw Qualtrics CSV export from the RIT Human Aesthetic
       Judgment Study.
    2. Filter to completed, non-preview responses only.
    3. Exclude raters who gave fewer than MIN_UNIQUE_RATINGS distinct beauty
       scores across the 10 direct-rating questions (Q1-Q10). This removes
       participants who did not engage meaningfully (e.g., P2 who gave the
       same score for every painting).
    4. For each of the four conditions (Abstract Beauty, Abstract Liking,
       Representational Beauty, Representational Liking), compute the
       per-rater average annotation time for both the direct and comparative
       question blocks, then average across raters.
    5. Report overall means and the percentage time reduction.

Paper values (Table 6):
    Abstract  -- Direct: 25.32s, Comparative: 13.89s
    Repr.     -- Direct: 29.23s, Comparative:  7.53s
    Overall   -- Direct: 27.28s, Comparative: 10.71s  (60% reduction)

Usage:
    cd human_survey/
    python survey_time_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_FILE = "../../Data/RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv"
OUT_DIR = "../../results/human_survey/time_comparison"

# Question ranges per condition.
# Direct ratings: Q1-Q5 (abstract), Q6-Q10 (representational).
# Comparative judgments: Q11-Q15 (abstract pairs), Q16-Q20 (representational pairs).
# Beauty and Liking share the same question pages (each page has both sub-questions),
# so their timing columns are identical within a block.
CONDITIONS = {
    "Abstract Beauty":  {"direct": range(1, 6),   "comparative": range(11, 16)},
    "Abstract Liking":  {"direct": range(1, 6),   "comparative": range(11, 16)},
    "Repr. Beauty":     {"direct": range(6, 11),  "comparative": range(16, 21)},
    "Repr. Liking":     {"direct": range(6, 11),  "comparative": range(16, 21)},
}

# Minimum unique beauty ratings required to include a rater
MIN_UNIQUE_RATINGS = 2


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_and_filter(csv_path):
    """Load the raw Qualtrics CSV and apply quality filters.

    Performs two filtering steps:
        1. Keeps only completed (Finished == "TRUE"), non-preview responses.
        2. Excludes raters whose direct beauty ratings (Q1_2 through Q10_2)
           have fewer than MIN_UNIQUE_RATINGS distinct values. This catches
           participants who clicked the same number for every painting,
           indicating low engagement. In the paper, this excluded P2.

    Parameters
    ----------
    csv_path : str
        Path to the raw Qualtrics CSV export.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with one row per valid rater.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["Finished"] == "TRUE") & (df["Status"] != "Survey Preview")].copy()
    print(f"Loaded {len(df)} completed responses")

    # Identify and exclude raters with insufficient rating variance.
    # Check all 10 direct beauty columns (Q1_2 .. Q10_2) per rater.
    excluded = []
    keep_mask = []
    for i, (idx, row) in enumerate(df.iterrows()):
        beauty_vals = [float(row[f"Q{q}_2"]) for q in range(1, 11)
                       if pd.notna(row.get(f"Q{q}_2"))]
        unique = len(set(beauty_vals))
        if unique < MIN_UNIQUE_RATINGS:
            excluded.append(f"P{i+1}")
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    if excluded:
        print(f"Excluded {len(excluded)} rater(s) with insufficient variance: {', '.join(excluded)}")

    df = df[keep_mask].copy()
    print(f"Using {len(df)} raters for analysis (n={len(df)})\n")
    return df


def compute_per_rater_avg_time(df, q_range):
    """Compute the grand mean annotation time using a two-level averaging scheme.

    For each rater, compute the mean of their page-submit times across the
    questions in q_range. Then average those per-rater means to get the
    grand mean. This two-level approach prevents raters with more non-null
    entries from dominating the result.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered survey DataFrame (one row per rater).
    q_range : range
        Question numbers to include (e.g., range(1, 6) for Q1-Q5).

    Returns
    -------
    float
        Grand mean annotation time in seconds, or 0.0 if no data.
    """
    rater_avgs = []
    for _, row in df.iterrows():
        times = []
        for q in q_range:
            # Qualtrics encodes page-level timing in "Q{n}_Time_Page Submit"
            col = f"Q{q}_Time_Page Submit"
            if col in df.columns and pd.notna(row[col]):
                times.append(float(row[col]))
        if times:
            rater_avgs.append(np.mean(times))
    return np.mean(rater_avgs) if rater_avgs else 0.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point: load data, compute per-condition times, print Table 6, and save CSV."""
    print("=" * 70)
    print("SURVEY TIME ANALYSIS (Table 4)")
    print("Direct Ratings vs Comparative Judgments")
    print("=" * 70 + "\n")

    df = load_and_filter(RAW_FILE)

    # Compute per-condition times (Table 6 in the paper).
    # Each condition yields two rows: one for Direct, one for Comparative.
    rows = []
    for condition, ranges in CONDITIONS.items():
        direct_time = compute_per_rater_avg_time(df, ranges["direct"])
        comp_time = compute_per_rater_avg_time(df, ranges["comparative"])
        rows.append({
            "Condition": condition,
            "Method": "Direct",
            "Time (s)": round(direct_time, 2)
        })
        rows.append({
            "Condition": condition,
            "Method": "Comparative",
            "Time (s)": round(comp_time, 2)
        })

    table4 = pd.DataFrame(rows)

    # Overall summary: average across the four conditions for each method,
    # then compute the percentage time reduction from Direct to Comparative.
    direct_times = table4[table4["Method"] == "Direct"]["Time (s)"]
    comp_times = table4[table4["Method"] == "Comparative"]["Time (s)"]
    overall_direct = direct_times.mean()
    overall_comp = comp_times.mean()
    reduction = (overall_direct - overall_comp) / overall_direct * 100

    # Print
    print("TABLE 4: Annotation Time Comparison (n=5)")
    print("-" * 50)
    for _, row in table4.iterrows():
        marker = "  **" if row["Method"] == "Comparative" else "    "
        print(f"{marker}{row['Condition']:20s} {row['Method']:12s} {row['Time (s)']:6.2f}s")
    print("-" * 50)
    print(f"    {'Overall':20s} {'Direct':12s} {overall_direct:6.2f}s")
    print(f"  **{'Overall':20s} {'Comparative':12s} {overall_comp:6.2f}s")
    print(f"\n    Time reduction: {reduction:.1f}%")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "table4_time_comparison.csv")
    table4.to_csv(out_path, index=False)
    print(f"\n    Saved: {out_path}")


if __name__ == "__main__":
    main()
