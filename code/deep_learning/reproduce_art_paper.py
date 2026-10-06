"""Regenerate the paper tables, statistical analyses, and budget figures from the released experiment data."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
from statistical_analysis import compute_tests, write_csv
from summarize_apddv2_regression_tuning import aggregate, load_rows, read_selection

ROOT = Path(__file__).resolve().parents[2]


def load_final_results(root):
    frames = []
    for name in ("aggregate/sidhu", "aggregate/apddv2", "rater"):
        directory = root / "results/deep_learning" / name
        data = pd.concat(
            [pd.read_csv(path) for path in sorted(directory.glob("*.csv"))],
            ignore_index=True,
        )
        data["N_key"] = data.N.fillna(0).astype(int)
        frames.append(data)
    return tuple(frames)


def summarize_selection(root, output):
    base = root / "results/deep_learning/head_selection"
    for phase in ("screen", "confirm"):
        configs = read_selection(
            None if phase == "screen" else base / "screen/summary/selected.json", phase
        )
        ranking, targets = aggregate(
            load_rows(base / phase / "raw"), configs, "validation_spearman"
        )
        write_csv(output / f"head_{phase}_ranking.csv", ranking)
        write_csv(output / f"head_{phase}_targets.csv", targets)


def seed_curve(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    macro = data.groupby(groups + ["seed"], as_index=False).spearman.mean()
    return macro.groupby(groups).spearman.mean().reset_index(name="mean")


def plot_curves(
    aggregate_curve: pd.DataFrame, rater_curve: pd.DataFrame, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, data, panel_key, panels, ylabel in (
        (
            "art_clip_n_sweep",
            aggregate_curve,
            "dataset",
            ("sidhu", "apddv2"),
            "Spearman correlation",
        ),
        (
            "art_rater_n_sweep",
            rater_curve,
            "mode",
            ("within", "cross"),
            "Mean Spearman correlation",
        ),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for axis, panel in zip(axes, panels):
            for objective, label, color, marker in (
                ("hinge", "Hinge", "#2563eb", "o"),
                ("bradley_terry", "Bradley--Terry", "#c63c00", "s"),
            ):
                selected = data[
                    (data[panel_key] == panel) & (data.objective == objective)
                ].sort_values("N_key")
                axis.plot(
                    selected.N_key,
                    selected["mean"],
                    marker=marker,
                    linewidth=2.5,
                    markersize=6,
                    label=label,
                    color=color,
                )
            baseline = data[
                (data[panel_key] == panel) & (data.objective == "regression")
            ]["mean"].iloc[0]
            axis.axhline(
                baseline,
                color="#4b5563",
                linewidth=2.5,
                linestyle=":",
                label="Regression",
            )
            title = {
                "sidhu": "Sidhu",
                "apddv2": "APDDv2",
                "within": "Within-rater",
                "cross": "Cross-rater",
            }[panel]
            axis.set_title(title, fontsize=16, fontweight="bold", pad=10)
            axis.set_xlabel(
                r"Normalized pair budget ($N = M/n_{\mathrm{train}}$)", fontsize=14
            )
            axis.set_xticks(range(1, 11))
            axis.tick_params(labelsize=12)
            axis.grid(axis="y", color="#d5dbe3", linewidth=1)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(ylabel, fontsize=14)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=13
        )
        figure.tight_layout(rect=(0, 0.13, 1, 1))
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)


def human_summary(root: Path, output: Path) -> None:
    """Reproduce manuscript summaries from released survey responses and GT scores."""
    retained = {"P1", "P3", "P4", "P5", "P6"}
    rows = []
    sys.path.insert(0, str(root / "code/human_survey"))
    from human_rating_agreement_unified import analyze_ratings

    matrices = output / "rq4_matrices"
    matrices.mkdir(exist_ok=True)
    for condition in (
        "Abstract_Beauty",
        "Abstract_Liking",
        "Repr_Beauty",
        "Repr_Liking",
    ):
        for method in ("Absolute", "Comparative"):
            path = matrices / f"Agreement_{condition}_{method}.csv"
            analyze_ratings(
                root / "results/human_survey/survey_data" / f"{condition}_{method}.csv",
                path,
                matrices / f"Summary_{condition}_{method}.csv",
                rating_type="absolute" if method == "Absolute" else "comparative",
                raters=["GT", "P1", "P3", "P4", "P5", "P6"],
            )
            data = pd.read_csv(path)
            pairs = data.Pair.map(lambda p: frozenset(p.split("/")))
            human = data[pairs.map(lambda p: p <= retained)]
            gt = data[pairs.map(lambda p: "GT" in p and p - {"GT"} <= retained)]
            for comparison, selected in (("human_human", human), ("human_GT", gt)):
                rows.append(
                    {
                        "condition": condition,
                        "method": method,
                        "comparison": comparison,
                        "accuracy": selected.Acc.mean(),
                        "pairs": len(selected),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "rq4_agreement.csv", index=False)
    from survey_time_analysis import summarize_timing

    raw = root / "Data/RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv"
    summarize_timing(raw).to_csv(output / "rq4_timing.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/deep_learning/paper"
    )
    parser.add_argument(
        "--human-output-dir", type=Path, default=ROOT / "results/human_survey/paper"
    )
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    root, output = (args.repository.resolve(), args.output_dir.resolve())
    for directory in (output, args.human_output_dir, args.figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    sidhu, apdd, rater = load_final_results(root)
    summarize_selection(root, output)
    ols = pd.read_csv(root / "results/baseline/heldout_ols/metrics.csv")
    tests = compute_tests(
        sidhu.to_dict("records"), apdd.to_dict("records"), ols.to_dict("records")
    )
    for name, rows in tests.items():
        write_csv(output / f"{name}.csv", rows)
    aggregate_data = pd.concat([sidhu, apdd], ignore_index=True)
    aggregate_data = aggregate_data[aggregate_data.representation == "clip-vit-b32"]
    aggregate_curve = seed_curve(aggregate_data, ["dataset", "objective", "N_key"])
    rater_curve = seed_curve(rater, ["mode", "objective", "N_key"])
    aggregate_curve.to_csv(output / "rq2_budget_curve.csv", index=False)
    rater_curve.to_csv(output / "rq3_budget_curve.csv", index=False)
    rq1 = (
        sidhu[sidhu.objective == "regression"]
        .groupby(["category", "target", "representation"])[["mae", "r2", "spearman"]]
        .mean()
        .reset_index()
    )
    rq1.to_csv(output / "rq1_sidhu_regression.csv", index=False)
    apdd[apdd.objective == "regression"].groupby("representation")[
        ["mae", "r2", "spearman"]
    ].mean().to_csv(output / "rq1_apdd_regression.csv")
    pd.concat(
        [rater, sidhu[sidhu.representation == "clip-vit-b32"].assign(mode="aggregate")],
        ignore_index=True,
    ).query("N_key <= 1").groupby(["category", "target", "mode", "objective"])[
        ["mae", "r2", "spearman"]
    ].mean().to_csv(
        output / "rq3_n1.csv"
    )
    human_summary(root, args.human_output_dir)
    plot_curves(aggregate_curve, rater_curve, args.figure_dir)
    print(
        f"Wrote paper tables to {output}, human results to {args.human_output_dir}, and plots to {args.figure_dir}"
    )


if __name__ == "__main__":
    main()
