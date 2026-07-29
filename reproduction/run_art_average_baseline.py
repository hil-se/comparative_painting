#!/usr/bin/env python3
"""Reproduce the repository's average-rating art baselines with fixed seeds.

The original experiment functions are imported and called unchanged. This
wrapper adds only:

* explicit seeds for Python, NumPy, and TensorFlow;
* unique checkpoint names;
* bounded process-level parallelism;
* incremental raw-result persistence; and
* comparison against the result CSVs tracked in the repository.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "average_baseline"

PAINTINGS = ("abstract", "representational")
RATING_TYPES = ("beauty", "liking")
METRICS = ("mae", "r2", "rho", "rs")
RAW_FIELDS = (
    "method",
    "painting",
    "rating_type",
    "N",
    "seed",
    "status",
    "elapsed_seconds",
    "mae",
    "r2",
    "rho",
    "rs",
    "error",
)


def parse_integer_spec(value: str) -> list[int]:
    """Parse either ``start:stop`` or a comma-separated integer list."""
    value = value.strip()
    if ":" in value:
        start_text, stop_text = value.split(":", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
        if stop <= start:
            raise argparse.ArgumentTypeError("range stop must be greater than start")
        return list(range(start, stop))

    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def task_key(row: dict[str, Any]) -> tuple[str, str, str, int | None, int]:
    n_value = row.get("N")
    if n_value in ("", None):
        parsed_n = None
    else:
        parsed_n = int(n_value)
    return (
        str(row["method"]),
        str(row["painting"]),
        str(row["rating_type"]),
        parsed_n,
        int(row["seed"]),
    )


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one original repository experiment in an isolated worker."""
    started = time.monotonic()
    method = str(task["method"])
    painting = str(task["painting"])
    rating_type = str(task["rating_type"])
    n_value = task.get("N")
    seed = int(task["seed"])
    threads_per_worker = int(task["threads_per_worker"])
    repo = Path(str(task["repo"])).resolve()
    deep_learning_dir = repo / "code" / "deep_learning"

    result: dict[str, Any] = {
        "method": method,
        "painting": painting,
        "rating_type": rating_type,
        "N": "" if n_value is None else int(n_value),
        "seed": seed,
        "status": "error",
        "elapsed_seconds": "",
        "mae": "",
        "r2": "",
        "rho": "",
        "rs": "",
        "error": "",
    }

    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads_per_worker)
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        os.environ.setdefault(
            "MPLCONFIGDIR", "/private/tmp/codex-mpl-comparative-painting"
        )

        os.chdir(deep_learning_dir)
        if str(deep_learning_dir) not in sys.path:
            sys.path.insert(0, str(deep_learning_dir))

        import numpy as np
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        process_id = (
            f"repro_{method}_{painting}_{rating_type}_"
            f"n{n_value if n_value is not None else 'na'}_s{seed}_{os.getpid()}"
        )

        if method == "regression":
            from experiment_runner import average_rating

            metrics = average_rating(
                painting,
                rating_type,
                origin=False,
                process_id=process_id,
            )
        elif method == "comparative":
            from comparitive_experiments import average_rating

            metrics = average_rating(
                painting,
                rating_type,
                origin=False,
                N=int(n_value),
                process_id=process_id,
            )
        else:
            raise ValueError(f"unsupported method: {method}")

        for metric in METRICS:
            result[metric] = float(metrics[metric])
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()

    return result


def read_raw_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_tasks(
    repo: Path,
    methods: list[str],
    paintings: list[str],
    rating_types: list[str],
    seeds: list[int],
    n_values: list[int],
    threads_per_worker: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for method in methods:
        for painting in paintings:
            for rating_type in rating_types:
                method_n_values: tuple[int | None, ...]
                if method == "comparative":
                    method_n_values = tuple(n_values)
                else:
                    method_n_values = (None,)
                for n_value in method_n_values:
                    for seed in seeds:
                        tasks.append(
                            {
                                "method": method,
                                "painting": painting,
                                "rating_type": rating_type,
                                "N": n_value,
                                "seed": seed,
                                "repo": str(repo),
                                "threads_per_worker": threads_per_worker,
                            }
                        )
    return tasks


def summarize(rows: list[dict[str, Any]], expected_runs: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = task_key(row)[:4]
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(
        groups,
        key=lambda item: (item[0], item[1], item[2], -1 if item[3] is None else item[3]),
    ):
        method, painting, rating_type, n_value = key
        group = groups[key]
        summary: dict[str, Any] = {
            "method": method,
            "painting": painting,
            "rating_type": rating_type,
            "N": "" if n_value is None else n_value,
            "n_success": len(group),
            "n_expected": expected_runs,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_rows.append(summary)
    return summary_rows


def read_reference(
    repo: Path,
    method: str,
    painting: str,
    rating_type: str,
    n_value: int | None,
) -> dict[str, float] | None:
    if method == "regression":
        path = (
            repo
            / "results"
            / "deep_learning"
            / "regression"
            / f"{painting}_{rating_type}_average.csv"
        )
    else:
        path = (
            repo
            / "results"
            / "deep_learning"
            / "comparative"
            / f"{painting}_{rating_type}_average_comparative.csv"
        )

    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if method == "comparative":
        matches = [row for row in rows if int(float(row["N"])) == int(n_value)]
        if not matches:
            return None
        source = matches[0]
    elif rows:
        source = rows[0]
    else:
        return None
    return {metric: float(source[metric]) for metric in METRICS}


def compare_with_repository(
    repo: Path, summary_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    for row in summary_rows:
        n_value = None if row["N"] == "" else int(row["N"])
        reference = read_reference(
            repo,
            str(row["method"]),
            str(row["painting"]),
            str(row["rating_type"]),
            n_value,
        )
        if reference is None:
            continue
        for metric in METRICS:
            reproduced = float(row[f"{metric}_mean"])
            repository_value = reference[metric]
            comparison.append(
                {
                    "method": row["method"],
                    "painting": row["painting"],
                    "rating_type": row["rating_type"],
                    "N": row["N"],
                    "metric": metric,
                    "reproduced_mean": reproduced,
                    "reproduced_std": row[f"{metric}_std"],
                    "repository_mean": repository_value,
                    "delta": reproduced - repository_value,
                    "absolute_delta": abs(reproduced - repository_value),
                }
            )
    return comparison


def git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def package_versions() -> dict[str, str]:
    packages = (
        "tensorflow",
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("regression", "comparative"),
        default=["regression", "comparative"],
    )
    parser.add_argument(
        "--paintings",
        nargs="+",
        choices=PAINTINGS,
        default=list(PAINTINGS),
    )
    parser.add_argument(
        "--rating-types",
        nargs="+",
        choices=RATING_TYPES,
        default=list(RATING_TYPES),
    )
    parser.add_argument(
        "--seeds",
        type=parse_integer_spec,
        default=list(range(10)),
        help="Either start:stop or comma-separated integers (default: 0:10).",
    )
    parser.add_argument(
        "--n-values",
        type=parse_integer_spec,
        default=[1, 10],
        help="Comparative N values (default: 1,10).",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=3,
        help="TensorFlow intra-op CPU threads assigned to each worker (default: 3).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed rows already present in raw_results.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    raw_path = output / "raw_results.csv"
    summary_path = output / "summary.csv"
    comparison_path = output / "comparison_to_repository.csv"
    metadata_path = output / "metadata.json"

    if not (repo / "code" / "deep_learning").is_dir():
        raise SystemExit(f"repository not found or incomplete: {repo}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.threads_per_worker < 1:
        raise SystemExit("--threads-per-worker must be at least 1")

    output.mkdir(parents=True, exist_ok=True)
    all_tasks = build_tasks(
        repo,
        args.methods,
        args.paintings,
        args.rating_types,
        args.seeds,
        args.n_values,
        args.threads_per_worker,
    )
    existing_rows = [] if args.no_resume else read_raw_results(raw_path)
    completed_keys = {
        task_key(row) for row in existing_rows if row.get("status") == "ok"
    }
    pending = [task for task in all_tasks if task_key(task) not in completed_keys]

    metadata = {
        "created_or_resumed_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "repository_commit": git_commit(repo),
        "model_code_modified": False,
        "feature_variant": "resized (origin=False)",
        "train_size": 140,
        "methods": args.methods,
        "paintings": args.paintings,
        "rating_types": args.rating_types,
        "seeds": args.seeds,
        "comparative_N_values": args.n_values,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "notes": [
            "Calls the repository's average_rating functions unchanged.",
            "Seeds were not supplied by the repository; this run fixes seeds explicitly.",
            "Comparative MAE/R2 are retained for fidelity, but raw hinge utilities have arbitrary offset and scale.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Repository: {repo}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        f"Tasks: {len(all_tasks)} total, {len(completed_keys)} already complete, "
        f"{len(pending)} pending; workers={args.workers}, "
        f"threads_per_worker={args.threads_per_worker}",
        flush=True,
    )

    results_by_key: dict[
        tuple[str, str, str, int | None, int], dict[str, Any]
    ] = {task_key(row): row for row in existing_rows}

    if pending:
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            future_to_task = {
                executor.submit(run_task, task): task for task in pending
            }
            finished = len(completed_keys)
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        **task,
                        "N": "" if task["N"] is None else task["N"],
                        "status": "error",
                        "elapsed_seconds": "",
                        "mae": "",
                        "r2": "",
                        "rho": "",
                        "rs": "",
                        "error": f"worker failure: {type(exc).__name__}: {exc}",
                    }
                results_by_key[task_key(row)] = row
                finished += 1
                ordered_rows = sorted(
                    results_by_key.values(),
                    key=lambda item: (
                        str(item["method"]),
                        str(item["painting"]),
                        str(item["rating_type"]),
                        -1 if item.get("N") in ("", None) else int(item["N"]),
                        int(item["seed"]),
                    ),
                )
                write_csv(raw_path, ordered_rows, RAW_FIELDS)

                label_n = "" if task["N"] is None else f" N={task['N']}"
                if row["status"] == "ok":
                    print(
                        f"[{finished}/{len(all_tasks)}] {task['method']} "
                        f"{task['painting']} {task['rating_type']}{label_n} "
                        f"seed={task['seed']} rho={float(row['rho']):.4f} "
                        f"rs={float(row['rs']):.4f} "
                        f"({float(row['elapsed_seconds']):.1f}s)",
                        flush=True,
                    )
                else:
                    print(
                        f"[{finished}/{len(all_tasks)}] ERROR {task['method']} "
                        f"{task['painting']} {task['rating_type']}{label_n} "
                        f"seed={task['seed']}: {row['error']}",
                        flush=True,
                    )

    final_rows = sorted(
        results_by_key.values(),
        key=lambda item: (
            str(item["method"]),
            str(item["painting"]),
            str(item["rating_type"]),
            -1 if item.get("N") in ("", None) else int(item["N"]),
            int(item["seed"]),
        ),
    )
    write_csv(raw_path, final_rows, RAW_FIELDS)
    summary_rows = summarize(final_rows, expected_runs=len(args.seeds))
    summary_fields = (
        "method",
        "painting",
        "rating_type",
        "N",
        "n_success",
        "n_expected",
        *(f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")),
    )
    write_csv(summary_path, summary_rows, tuple(summary_fields))

    comparisons = compare_with_repository(repo, summary_rows)
    comparison_fields = (
        "method",
        "painting",
        "rating_type",
        "N",
        "metric",
        "reproduced_mean",
        "reproduced_std",
        "repository_mean",
        "delta",
        "absolute_delta",
    )
    write_csv(comparison_path, comparisons, comparison_fields)

    failures = [row for row in final_rows if row.get("status") != "ok"]
    missing = len(all_tasks) - sum(
        1 for task in all_tasks if task_key(task) in results_by_key
    )
    print(
        f"Finished: {len(final_rows) - len(failures)} successful rows, "
        f"{len(failures)} failed rows, {missing} missing rows.",
        flush=True,
    )
    print(f"Summary: {summary_path}", flush=True)
    print(f"Comparison: {comparison_path}", flush=True)
    return 0 if not failures and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
