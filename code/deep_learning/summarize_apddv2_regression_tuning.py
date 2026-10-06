"""Rank candidate heads by APDDv2 validation macro Spearman."""

from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tune_apddv2_regression import APDD_TARGETS, CONFIG_BY_ID, CONFIGS, parse_range


def read_selection(path, phase):
    if phase == "screen":
        return [config.config_id for config in CONFIGS]
    return json.loads(path.read_text())["selected_config_ids"]


def load_rows(input_dir: Path) -> list[dict[str, str]]:
    paths = sorted(input_dir.glob("*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def aggregate(
    rows: list[dict[str, str]], configs: list[str], metric: str
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
                }
            )
        ranking.append(
            {
                "config_id": config_id,
                f"macro_{metric}": float(np.mean(target_means)),
            }
        )
    ranking.sort(key=lambda row: float(row[f"macro_{metric}"]), reverse=True)
    return (ranking, per_target)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
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
    metric = "validation_spearman"
    ranking, per_target = aggregate(rows, configs, metric)
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
    else:
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
    (args.output_dir / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
