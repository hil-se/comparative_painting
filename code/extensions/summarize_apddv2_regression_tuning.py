"""Validate and aggregate APDDv2 regression tuning phases."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tune_apddv2_regression import APDD_TARGETS, CONFIG_BY_ID, CONFIGS, parse_range


PUBLISHED_ARTCLIP_MEAN_SPEARMAN = 0.771


def read_selection(path: Path | None, phase: str) -> list[str]:
    if phase == "screen":
        return [config.config_id for config in CONFIGS]
    if path is None:
        raise ValueError(f"{phase} aggregation requires --selection")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if phase == "confirm":
        values = payload["selected_config_ids"]
    else:
        values = [payload["winner_config_id"]]
    unknown = sorted(set(values) - set(CONFIG_BY_ID))
    if unknown:
        raise ValueError(f"Unknown selected configs: {unknown}")
    return values


def load_rows(input_dir: Path) -> list[dict[str, str]]:
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {input_dir}")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def validate_rows(
    rows: list[dict[str, str]], phase: str, configs: list[str], seeds: list[int]
) -> None:
    required_metric = "test_spearman" if phase == "test" else "validation_spearman"
    expected = {
        (target, config_id, seed)
        for target in APDD_TARGETS
        for config_id in configs
        for seed in seeds
    }
    observed = [
        (row["target"], row["config_id"], int(row["seed"])) for row in rows
    ]
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate target/config/seed rows found")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"Incomplete results: missing={missing[:5]}, extra={extra[:5]}")
    for row in rows:
        if row["phase"] != phase:
            raise ValueError(f"Unexpected phase {row['phase']!r}")
        if required_metric not in row or not np.isfinite(float(row[required_metric])):
            raise ValueError(f"Invalid {required_metric} in {row}")
        if phase != "test" and any(key.startswith("test_") for key in row):
            raise ValueError("A tuning CSV contains held-out test fields")


def aggregate(
    rows: list[dict[str, str]], configs: list[str], metric: str, seeds: list[int]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranking: list[dict[str, object]] = []
    per_target: list[dict[str, object]] = []
    for config_id in configs:
        target_means: list[float] = []
        for target in APDD_TARGETS:
            values = [
                float(row[metric])
                for row in rows
                if row["config_id"] == config_id and row["target"] == target
            ]
            mean = float(np.mean(values))
            target_means.append(mean)
            per_target.append(
                {
                    "config_id": config_id,
                    "target": target,
                    f"mean_{metric}": mean,
                    f"std_{metric}": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "seeds": len(values),
                }
            )
        seed_macros = []
        for seed in seeds:
            values = [
                float(row[metric])
                for row in rows
                if row["config_id"] == config_id and int(row["seed"]) == seed
            ]
            seed_macros.append(float(np.mean(values)))
        ranking.append(
            {
                "config_id": config_id,
                f"macro_{metric}": float(np.mean(target_means)),
                f"min_target_{metric}": float(np.min(target_means)),
                f"max_target_{metric}": float(np.max(target_means)),
                f"seed_macro_std_{metric}": (
                    float(np.std(seed_macros, ddof=1)) if len(seed_macros) > 1 else 0.0
                ),
                "targets": len(target_means),
                "seeds": len(seeds),
            }
        )
    ranking.sort(key=lambda row: float(row[f"macro_{metric}"]), reverse=True)
    return ranking, per_target


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "confirm", "test"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_range(args.seeds)
    configs = read_selection(args.selection, args.phase)
    rows = load_rows(args.input_dir)
    validate_rows(rows, args.phase, configs, seeds)
    metric = "test_spearman" if args.phase == "test" else "validation_spearman"
    ranking, per_target = aggregate(rows, configs, metric, seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ranking.csv", ranking)
    write_csv(args.output_dir / "per_target.csv", per_target)
    if args.phase == "screen":
        selected_ids = [row["config_id"] for row in ranking[: args.top_k]]
        payload = {
            "phase": "screen",
            "selection_metric": "unweighted macro validation Spearman",
            "selected_config_ids": selected_ids,
            "selected_configs": [asdict(CONFIG_BY_ID[value]) for value in selected_ids],
            "top_k": args.top_k,
            "seeds": seeds,
        }
        name = "selected.json"
    elif args.phase == "confirm":
        winner = str(ranking[0]["config_id"])
        payload = {
            "phase": "confirm",
            "selection_metric": "unweighted macro validation Spearman",
            "winner_config_id": winner,
            "winner_config": asdict(CONFIG_BY_ID[winner]),
            "validation_macro_spearman": ranking[0]["macro_validation_spearman"],
            "seeds": seeds,
            "locked_before_test": True,
        }
        name = "winner.json"
    else:
        test_score = float(ranking[0]["macro_test_spearman"])
        payload = {
            "phase": "test",
            "winner_config_id": ranking[0]["config_id"],
            "test_macro_spearman": test_score,
            "published_derived_artclip_mean_spearman": PUBLISHED_ARTCLIP_MEAN_SPEARMAN,
            "difference": test_score - PUBLISHED_ARTCLIP_MEAN_SPEARMAN,
            "exceeds_published_derived_artclip": test_score > PUBLISHED_ARTCLIP_MEAN_SPEARMAN,
            "comparison_caveat": "ArtCLIP mean is derived from published per-attribute values under a different reported protocol.",
            "seeds": seeds,
        }
        name = "test_summary.json"
    (args.output_dir / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
