"""Validate archived evidence and rebuild the Art Eval tables and budget figures.

Runs on a CPU with requirements-analysis.txt. Neural training and the external
APDDv2 image archive are not needed to regenerate the published analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

from manuscript_statistical_tests import compute_tests, write_csv
from summarize_apddv2_regression_tuning import (
    aggregate, load_rows, read_selection, validate_rows,
)
from validate_art_result import validate

ROOT = Path(__file__).resolve().parents[2]
SEEDS = set(range(10))
CONDITIONS = set(itertools.product(("abstract", "representational"), ("beauty", "liking")))
OBJECTIVE_BUDGETS = {("regression", 0)} | set(itertools.product(("hinge", "bradley_terry"), range(1, 11)))


def load_final_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify hashes and the full matrix, rather than silently pairing a subset."""
    base = root / "results/extensions/locked_head"
    frames = []
    for directory, files, rows in ((base / "aggregate_merged/sidhu", 16, 1680),
                                   (base / "aggregate_merged/apddv2", 22, 4620),
                                   (base / "rater_merged", 40, 8400)):
        sources = sorted(directory.glob("*.csv"))
        if len(sources) != files:
            raise ValueError(f"Expected {files} files in {directory}")
        pieces = []
        for path in sources:
            metadata = json.loads(path.with_suffix(".metadata.json").read_text())
            features = (root / "Data/fixed_features/sidhu-clip-vit-b32.npz"
                        if metadata.get("dataset") == "sidhu"
                        and metadata.get("representation") == "clip-vit-b32" else None)
            validate(path, int(metadata["rows"]), features=features)
            pieces.append(pd.read_csv(path))
        data = pd.concat(pieces, ignore_index=True)
        if len(data) != rows:
            raise ValueError(f"Expected {rows} rows in {directory}")
        data["N_key"] = data.N.fillna(0).astype(int)
        if not np.isfinite(data[["mae", "r2", "spearman"]].to_numpy()).all():
            raise ValueError(f"Nonfinite metrics in {directory}")
        frames.append(data)
    sidhu, apdd, rater = frames
    def check(data, keys, expected):
        observed = list(data[keys].itertuples(index=False, name=None))
        if len(set(observed)) != len(observed) or set(observed) != expected:
            raise ValueError(f"Incomplete/duplicated experimental matrix: {keys}")
    check(sidhu, ["category", "target", "representation", "seed", "objective", "N_key"],
          {(c, t, rep, seed, obj, n) for c, t in CONDITIONS
           for rep in ("clip-vit-b32", "resnet50") for seed in SEEDS for obj, n in OBJECTIVE_BUDGETS})
    from tune_apddv2_regression import APDD_TARGETS
    check(apdd, ["target", "representation", "seed", "objective", "N_key"],
          {(t, rep, seed, obj, n) for t in APDD_TARGETS
           for rep in ("clip-vit-b32", "resnet50") for seed in SEEDS for obj, n in OBJECTIVE_BUDGETS})
    check(rater, ["category", "target", "mode", "rater", "seed", "objective", "N_key"],
          {(c, t, mode, r, seed, obj, n) for c, t in CONDITIONS for mode in ("within", "cross")
           for r in range(1, 6) for seed in SEEDS for obj, n in OBJECTIVE_BUDGETS})
    return sidhu, apdd, rater


def verify_selection(root: Path, output: Path) -> dict:
    base = root / "results/extensions/head_selection"
    selected = base / "screen/summary/selected.json"
    winner = json.loads((base / "confirm/summary/winner.json").read_text())
    from run_art_locked_head import LOCKED_METHOD_ID
    if winner["winner_config_id"] != LOCKED_METHOD_ID:
        raise ValueError("The runner's locked method differs from the archived winner")
    phases = {}
    for phase, seeds in (("screen", list(range(3))), ("confirm", list(range(10)))):
        directory = base / phase / "raw"
        for path in directory.glob("*.csv"):
            metadata = json.loads(path.with_suffix(".metadata.json").read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["output_sha256"]:
                raise ValueError(f"Tuning hash mismatch: {path}")
        configs = read_selection(None if phase == "screen" else selected, phase)
        rows = load_rows(directory)
        validate_rows(rows, phase, configs, seeds)
        ranking, targets = aggregate(rows, configs, "validation_spearman", seeds)
        write_csv(output / f"head_{phase}_ranking.csv", ranking)
        write_csv(output / f"head_{phase}_targets.csv", targets)
        if phase == "screen":
            top = [r["config_id"] for r in ranking[:3]]
            if top != json.loads(selected.read_text())["selected_config_ids"]:
                raise ValueError("Archived shortlist differs from validation ranking")
        elif ranking[0]["config_id"] != winner["winner_config_id"]:
            raise ValueError("Archived winner differs from validation ranking")
        elif not np.isclose(ranking[0]["macro_validation_spearman"],
                            winner["validation_macro_spearman"], rtol=0, atol=1e-12):
            raise ValueError("Archived winner score differs from validation ranking")
        phases[phase] = {"configurations": len(configs), "fits": len(rows), "seeds": seeds}
    return {"phases": phases, "winner": winner["winner_config_id"],
            "validation_macro_spearman": winner["validation_macro_spearman"]}


def seed_curve(data: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    # Conditions/targets/raters are averaged within seed before calculating SD.
    macro = data.groupby(groups + ["seed"], as_index=False).spearman.mean()
    return macro.groupby(groups).spearman.agg(mean="mean", std="std", seeds="count").reset_index()


def plot_curves(aggregate_curve: pd.DataFrame, rater_curve: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, data, panel_key, panels in (
        ("art_clip_n_sweep", aggregate_curve, "dataset", ("sidhu", "apddv2")),
        ("art_rater_n_sweep", rater_curve, "mode", ("within", "cross")),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
        for axis, panel in zip(axes, panels, strict=True):
            for objective, label, color in (("hinge", "Hinge", "#3267a8"),
                                            ("bradley_terry", "Bradley–Terry", "#c05a32")):
                selected = data[(data[panel_key] == panel) & (data.objective == objective)].sort_values("N_key")
                x, y, sd = selected.N_key.to_numpy(), selected["mean"].to_numpy(), selected["std"].to_numpy()
                axis.plot(x, y, marker="o", label=label, color=color)
                axis.fill_between(x, y - sd, y + sd, color=color, alpha=.15)
            baseline = data[(data[panel_key] == panel) & (data.objective == "regression")]["mean"].iloc[0]
            axis.axhline(baseline, color="#555555", linestyle="--", label="Regression")
            title = {"sidhu": "Sidhu", "apddv2": "APDDv2", "within": "Within rater", "cross": "Cross rater"}[panel]
            axis.set(title=title, xlabel="Pair-to-item budget N", ylabel="Mean Spearman")
            axis.set_xticks(range(1, 11))
            axis.grid(alpha=.15)
        axes[0].legend(fontsize=8)
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{name}.{extension}", dpi=200)
        plt.close(figure)


def human_summary(root: Path, output: Path) -> None:
    """Reproduce manuscript summaries from released survey responses and GT scores."""
    retained = {"P1", "P3", "P4", "P5", "P6"}
    expected = {frozenset(p) for p in itertools.combinations(retained, 2)}
    rows = []
    sys.path.insert(0, str(root / "code/human_survey"))
    from human_rating_agreement_unified import analyze_ratings
    matrices = output / "rq4_matrices"
    matrices.mkdir(exist_ok=True)
    for condition in ("Abstract_Beauty", "Abstract_Liking", "Repr_Beauty", "Repr_Liking"):
        for method in ("Absolute", "Comparative"):
            path = matrices / f"Agreement_{condition}_{method}.csv"
            analyze_ratings(root / "results/human_survey/survey_data" / f"{condition}_{method}.csv",
                            path, matrices / f"Summary_{condition}_{method}.csv",
                            rating_type="absolute" if method == "Absolute" else "comparative",
                            raters=["GT", "P1", "P3", "P4", "P5", "P6"])
            data = pd.read_csv(path)
            pairs = data.Pair.map(lambda p: frozenset(p.split("/")))
            human = data[pairs.map(lambda p: p <= retained)]
            gt = data[pairs.map(lambda p: "GT" in p and p - {"GT"} <= retained)]
            if set(human.Pair.map(lambda p: frozenset(p.split("/")))) != expected or len(human) != 10 or len(gt) != 5:
                raise ValueError("Unexpected retained human/GT agreement population")
            for comparison, selected in (("human_human", human), ("human_GT", gt)):
                rows.append({"condition": condition, "method": method, "comparison": comparison,
                             "accuracy": selected.Acc.mean(), "pairs": len(selected)})
    pd.DataFrame(rows).to_csv(output / "rq4_agreement.csv", index=False)
    sys.path.insert(0, str(root / "code/human_survey"))
    from survey_time_analysis import load_and_filter, compute_per_rater_avg_time, CONDITIONS as TIMING_CONDITIONS
    raw = root / "Data/RIT-Human-Aesthetic-Judgment-Study_November-27-2025_14.58.csv"
    source = pd.read_csv(raw)
    finished = source[source.Finished == "TRUE"]
    counts = {"finished_entries": len(finished),
              "finished_preview_entries": int((finished.Status == "Survey Preview").sum()),
              "finished_non_preview": int((finished.Status != "Survey Preview").sum())}
    raters = load_and_filter(raw)
    if len(raters) != 5:
        raise ValueError("Expected five retained timing responses")
    counts["retained_timing_responses"] = len(raters)
    (output / "rq4_source_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    times = []
    for condition, methods in TIMING_CONDITIONS.items():
        for method, questions in methods.items():
            seconds = float(compute_per_rater_avg_time(raters, questions))
            times.append({"condition": condition, "method": method,
                          "seconds_raw": seconds, "seconds": round(seconds, 2)})
    for method in ("direct", "comparative"):
        values = [row for row in times if row["method"] == method]
        paper_mean = sum(Decimal(str(row["seconds"])) for row in values) / Decimal(len(values))
        times.append({"condition": "Overall", "method": method,
                      "seconds_raw": np.mean([row["seconds_raw"] for row in values]),
                      "seconds": float(paper_mean.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))})
    pd.DataFrame(times).to_csv(output / "rq4_timing.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper")
    args = parser.parse_args()
    root, output = args.repository.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sidhu, apdd, rater = load_final_results(root)
    selection = verify_selection(root, output)
    ols_path = root / "results/extensions/heldout_ols/metrics.csv"
    metadata = json.loads(ols_path.with_suffix(".metadata.json").read_text())
    if hashlib.sha256(ols_path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("OLS result hash mismatch")
    for name, expected_hash in metadata["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"OLS source hash mismatch: {name}")
    ols = pd.read_csv(ols_path)
    tests = compute_tests(sidhu.to_dict("records"), apdd.to_dict("records"), ols.to_dict("records"))
    for name, rows in tests.items():
        write_csv(output / f"{name}.csv", rows)
    aggregate_data = pd.concat([sidhu, apdd], ignore_index=True)
    aggregate_data = aggregate_data[aggregate_data.representation == "clip-vit-b32"]
    aggregate_curve = seed_curve(aggregate_data, ["dataset", "objective", "N_key"])
    rater_curve = seed_curve(rater, ["mode", "objective", "N_key"])
    aggregate_curve.to_csv(output / "rq2_budget_curve.csv", index=False)
    rater_curve.to_csv(output / "rq3_budget_curve.csv", index=False)
    pair_rows = pd.concat([sidhu, apdd, rater], ignore_index=True)
    pair_rows = pair_rows[pair_rows.objective != "regression"]
    shortfalls = (pair_rows.train_pairs != pair_rows.N_key * pair_rows.train_examples).sum()
    rq1 = sidhu[sidhu.objective == "regression"].groupby(["category", "target", "representation"])[["mae", "r2", "spearman"]].mean().reset_index()
    rq1.to_csv(output / "rq1_sidhu_regression.csv", index=False)
    apdd[apdd.objective == "regression"].groupby("representation")[["mae", "r2", "spearman"]].mean().to_csv(output / "rq1_apdd_regression.csv")
    pd.concat([rater.assign(mode=rater["mode"]),
               sidhu[sidhu.representation == "clip-vit-b32"].assign(mode="aggregate")], ignore_index=True).query("N_key <= 1").groupby(
        ["category", "target", "mode", "objective"])[["mae", "r2", "spearman"]].mean().to_csv(output / "rq3_n1.csv")
    human_summary(root, output)
    plot_curves(aggregate_curve, rater_curve, output)
    audit = {"source_files": 78, "model_fit_rows": len(sidhu) + len(apdd) + len(rater),
             "head_selection": selection, "ols_fit_rows": len(ols),
             "pairwise_fit_rows": len(pair_rows), "pair_budget_shortfalls": int(shortfalls),
             "statistical_contrasts": sum(map(len, tests.values())),
             "bands": "sample SD of ten seed-level macro averages; raters are averaged within seed",
             "human_agreement": "regenerated from released survey data, rounded to two decimals per matrix entry"}
    (output / "integrity.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
