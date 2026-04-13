#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


RESULT_COLUMNS = [
    "benchmark",
    "dataset_id",
    "dataset_dir",
    "dataset_name",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "accuracy",
    "f1_weighted",
    "logloss",
    "fit_seconds",
    "predict_seconds",
    "status",
    "error",
]


def discover_worker_csvs(result_dir: Path, pattern: str) -> List[Path]:
    return sorted(
        [
            path
            for path in result_dir.glob(pattern)
            if path.is_file() and path.name != "all_classification_results.csv"
        ],
        key=lambda p: p.name,
    )


def read_worker_outputs(worker_csvs: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for worker_csv in worker_csvs:
        with worker_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = {column: raw_row.get(column, "") for column in RESULT_COLUMNS}
                if not any(str(value).strip() for value in row.values()):
                    continue
                rows.append(row)
    return rows


def to_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def write_summary(summary_path: Path, rows: List[Dict[str, str]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") == "fail"]

    accuracy_values = [v for v in (to_float(row.get("accuracy", "")) for row in ok_rows) if v is not None]
    f1_values = [v for v in (to_float(row.get("f1_weighted", "")) for row in ok_rows) if v is not None]
    logloss_values = [v for v in (to_float(row.get("logloss", "")) for row in ok_rows) if v is not None]

    lines = [
        f"discovered_datasets: {len(rows)}",
        f"processed_datasets: {len(rows)}",
        f"ok_count: {len(ok_rows)}",
        f"failed_count: {len(failed_rows)}",
        f"avg_accuracy_ok: {sum(accuracy_values) / len(accuracy_values):.6f}" if accuracy_values else "avg_accuracy_ok: (none)",
        f"avg_f1_weighted_ok: {sum(f1_values) / len(f1_values):.6f}" if f1_values else "avg_f1_weighted_ok: (none)",
        f"avg_logloss_ok: {sum(logloss_values) / len(logloss_values):.6f}" if logloss_values else "avg_logloss_ok: (none)",
    ]

    failed_names = sorted(
        {
            str(row.get("dataset_name", "")).strip()
            for row in failed_rows
            if str(row.get("dataset_name", "")).strip()
        }
    )
    lines.append(
        f"failed_datasets: {', '.join(failed_names)}"
        if failed_names
        else "failed_datasets: (none)"
    )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_result_dir(result_dir: Path, worker_glob: str) -> None:
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    worker_csvs = discover_worker_csvs(result_dir, worker_glob)
    if not worker_csvs:
        raise FileNotFoundError(
            f"No worker csv files found in {result_dir} with pattern {worker_glob!r}"
        )

    all_rows = read_worker_outputs(worker_csvs)
    all_csv = result_dir / "all_classification_results.csv"
    write_csv(all_csv, all_rows)
    write_summary(result_dir / "summary.txt", all_rows)

    benchmark_names = sorted(
        {
            str(row.get("benchmark", "")).strip()
            for row in all_rows
            if str(row.get("benchmark", "")).strip()
            and str(row.get("benchmark", "")).strip() != "__worker__"
        }
    )

    for benchmark in benchmark_names:
        benchmark_dir = result_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_rows = [row for row in all_rows if row.get("benchmark") == benchmark]
        write_csv(benchmark_dir / "all_classification_results.csv", benchmark_rows)
        write_summary(benchmark_dir / "summary.txt", benchmark_rows)

    print(f"worker_csv_count: {len(worker_csvs)}")
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {result_dir / 'summary.txt'}")
    print("saved_benchmark_summaries:")
    for benchmark in benchmark_names:
        print(f"  {benchmark}: {result_dir / benchmark / 'summary.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate worker_*.csv files in a classification result directory into "
            "root and per-benchmark all_classification_results.csv/summary.txt files."
        )
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory containing worker_*.csv files.",
    )
    parser.add_argument(
        "--worker-glob",
        default="worker_*.csv",
        help="Glob pattern used to find worker csv files. Default: worker_*.csv",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir).expanduser()
    try:
        result_dir = result_dir.resolve()
    except Exception:
        pass

    aggregate_result_dir(result_dir=result_dir, worker_glob=args.worker_glob)


if __name__ == "__main__":
    main()
