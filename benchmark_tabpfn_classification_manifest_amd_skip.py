#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import benchmark_tabpfn_classification_amd_skip as base


def load_manifest_target_path(manifest_path: Path) -> Path:
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        return Path(line)
    raise ValueError(f"Manifest file does not contain a dataset path: {manifest_path}")


def build_manifest_tasks(
    root: Path,
    benchmark_specs: Sequence[str],
    skip_entries: set[str],
) -> Tuple[List[Tuple[str, str]], Dict[str, int], Dict[str, List[str]]]:
    tasks: List[Tuple[str, str]] = []
    discovered: Dict[str, int] = {}
    skipped: Dict[str, List[str]] = {}

    for benchmark_name, manifest_dir in base.parse_benchmark_specs(root, benchmark_specs):
        manifest_files = base.discover_csv_files(manifest_dir) if manifest_dir.exists() else []
        discovered[benchmark_name] = len(manifest_files)
        skipped[benchmark_name] = []

        for manifest_path in manifest_files:
            target_path = load_manifest_target_path(manifest_path)
            skip_tokens = base.build_skip_tokens(benchmark_name, target_path)
            skip_tokens.add(base.normalize_skip_entry(manifest_path.name))
            skip_tokens.add(base.normalize_skip_entry(manifest_path.stem))
            skip_tokens.add(
                base.normalize_skip_entry(f"{base.normalize_skip_entry(benchmark_name)}/{manifest_path.name}")
            )

            if skip_entries and skip_tokens.intersection(skip_entries):
                skipped[benchmark_name].append(target_path.name or manifest_path.name)
                continue

            tasks.append((benchmark_name, str(target_path)))

    return tasks, discovered, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabPFN classification benchmarks from manifest csv files whose "
            "contents are dataset paths, with AMD/ROCm multi-GPU and skip support."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--out-dir", default="result/recover_worker0_root_classification_8gpu")
    parser.add_argument(
        "--model-path",
        default="ckpt/TabPFN-2.5/tabpfn-v2.5-classifier-v2.5_default.ckpt",
    )
    parser.add_argument(
        "--skip-file",
        default="skip.txt",
        help="Optional newline-delimited skip list. Missing files are ignored.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--fit-mode",
        default="fit_preprocessors",
        choices=["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    )
    parser.add_argument(
        "--inference-precision",
        default="auto",
        choices=["auto", "autocast"],
    )
    parser.add_argument("--memory-saving-mode", default="auto")
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--ignore-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_true",
        help="Ignore TabPFN's pretraining sample/feature limits. Enabled by default.",
    )
    limit_group.add_argument(
        "--enforce-pretraining-limits",
        dest="ignore_pretraining_limits",
        action="store_false",
        help="Re-enable TabPFN's pretraining sample/feature limit checks.",
    )
    parser.set_defaults(ignore_pretraining_limits=True)
    parser.add_argument("--softmax-temperature", type=float, default=0.9)
    parser.add_argument("--balance-probabilities", action="store_true")
    parser.add_argument("--average-before-softmax", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base.ensure_runtime_deps()

    root = Path(args.root).expanduser()
    try:
        root = root.resolve()
    except Exception:
        pass
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    model_path = base.normalize_local_ckpt_path(args.model_path)
    skip_file = base.resolve_optional_path(root, args.skip_file)
    skip_entries = base.load_skip_entries(skip_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_specs = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    benchmark_names = [name for name, _ in base.parse_benchmark_specs(root, benchmark_specs)]
    tasks, discovered, skipped = build_manifest_tasks(root, benchmark_specs, skip_entries)
    if not tasks:
        raise FileNotFoundError(
            "No manifest-backed classification tasks remain after discovery and skip filtering."
        )

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")

    model_kwargs: Dict[str, object] = {
        "model_path": str(model_path),
        "n_estimators": args.n_estimators,
        "softmax_temperature": args.softmax_temperature,
        "balance_probabilities": args.balance_probabilities,
        "average_before_softmax": args.average_before_softmax,
        "fit_mode": args.fit_mode,
        "inference_precision": args.inference_precision,
        "memory_saving_mode": args.memory_saving_mode,
        "n_preprocessing_jobs": 1,
        "random_state": args.random_state,
        "ignore_pretraining_limits": args.ignore_pretraining_limits,
    }

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    ready_queue: mp.Queue = mp.Queue()
    task_queue: mp.Queue = mp.Queue()
    start_event = mp.Event()
    processes: List[mp.Process] = []

    for task in tasks:
        task_queue.put(task)
    for _ in range(args.workers):
        task_queue.put(None)

    for worker_id in range(args.workers):
        proc = mp.Process(
            target=base.run_worker,
            args=(
                worker_id,
                gpu_ids[worker_id],
                task_queue,
                ready_queue,
                start_event,
                str(out_dir / f"worker_{worker_id}.csv"),
                dict(model_kwargs),
                args.test_size,
                args.random_state,
                args.verbose,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    ready_workers: set[int] = set()
    while len(ready_workers) < args.workers:
        try:
            message = ready_queue.get(timeout=10)
        except Exception:
            dead_workers = [
                str(idx)
                for idx, proc in enumerate(processes)
                if not proc.is_alive() and idx not in ready_workers
            ]
            if dead_workers:
                raise RuntimeError(
                    "Some workers exited before initialization completed: "
                    + ", ".join(dead_workers)
                )
            continue

        if message.get("status") == "ready":
            ready_workers.add(int(message["worker_id"]))
            if args.verbose:
                print(
                    f"[worker {message['worker_id']} | gpu {message['gpu_id']}] "
                    f"ready assigned={message.get('assigned_count', '?')}"
                )
            continue

        if message.get("status") == "crash":
            raise RuntimeError(
                f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                f"during initialization:\n{message.get('error', '(no traceback)')}"
            )

    start_event.set()

    for proc in processes:
        proc.join()

    dfs = base.collect_worker_outputs(out_dir, args.workers)
    columns = list(base.ResultRow.__annotations__.keys())
    all_df = base.pd.concat(dfs, ignore_index=True) if dfs else base.pd.DataFrame(columns=columns)
    all_csv = out_dir / "all_classification_results.csv"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    skipped_all = [name for names in skipped.values() for name in names]
    base.write_summary(
        out_dir / "summary.txt",
        all_df,
        sum(discovered.values()),
        skipped_all,
        wall_seconds,
    )

    for benchmark in benchmark_names:
        benchmark_dir = out_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_df = (
            all_df[all_df["benchmark"] == benchmark].copy()
            if len(all_df)
            else base.pd.DataFrame(columns=columns)
        )
        benchmark_df.to_csv(benchmark_dir / "all_classification_results.csv", index=False)
        base.write_summary(
            benchmark_dir / "summary.txt",
            benchmark_df,
            discovered.get(benchmark, 0),
            skipped.get(benchmark, []),
            wall_seconds,
        )

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {out_dir / 'summary.txt'}")
    print("saved_benchmark_summaries:")
    for benchmark in benchmark_names:
        print(f"  {benchmark}: {out_dir / benchmark / 'summary.txt'}")
    print(f"skip_file: {skip_file if skip_file and skip_file.exists() else '(none)'}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
