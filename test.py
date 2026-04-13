#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_BENCHMARKS = [
    "tabarena_cls=dataset/tabarena/cls",
    "talent_binclass=dataset/talent_cls/binclass",
    "talent_multiclass=dataset/talent_cls/multiclass",
]
TARGET_CANDIDATES = ["target", "label", "class", "y", "TARGET", "Label", "Class", "Y"]

np = None
pd = None
accuracy_score = None
f1_score = None
log_loss = None
train_test_split = None
_MANYCLASS_CLASS = None
_MANYCLASS_IMPORT_ATTEMPTED = False
_MANYCLASS_IMPORT_ERROR = None


@dataclass
class ResultRow:
    benchmark: str
    dataset_id: str
    dataset_dir: str
    dataset_name: str
    n_train: int
    n_test: int
    n_features: int
    n_classes: Optional[int]
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    logloss: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


def ensure_runtime_deps() -> None:
    global np
    global pd
    global accuracy_score
    global f1_score
    global log_loss
    global train_test_split

    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd
        from sklearn.metrics import (
            accuracy_score as _accuracy_score,
            f1_score as _f1_score,
            log_loss as _log_loss,
        )
        from sklearn.model_selection import train_test_split as _train_test_split

        np = _np
        pd = _pd
        accuracy_score = _accuracy_score
        f1_score = _f1_score
        log_loss = _log_loss
        train_test_split = _train_test_split


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def sanitize_dataset_id(csv_path: Path) -> str:
    match = re.search(r"(OpenML-ID-\d+)", str(csv_path))
    return match.group(1) if match else csv_path.stem


def infer_target_column(df) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def parse_benchmark_specs(root: Path, specs: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, rel_path = spec.split("=", 1)
            benchmark_name = name.strip()
            benchmark_path = Path(rel_path.strip())
        else:
            benchmark_path = Path(spec.strip())
            benchmark_name = benchmark_path.name

        if not benchmark_path.is_absolute():
            benchmark_path = (root / benchmark_path).resolve()
        else:
            benchmark_path = benchmark_path.resolve()
        parsed.append((benchmark_name, benchmark_path))
    return parsed


def discover_csv_files(benchmark_dir: Path) -> List[Path]:
    csv_files: List[Path] = []
    for csv_path in benchmark_dir.rglob("*.csv"):
        name = csv_path.name
        if name.endswith("_train.csv") or name.endswith("_test.csv"):
            continue
        csv_files.append(csv_path)
    return sorted(csv_files, key=lambda p: p.name)


def build_tasks(root: Path, benchmark_specs: List[str]) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    tasks: List[Tuple[str, str]] = []
    discovered: Dict[str, int] = {}
    for benchmark_name, benchmark_dir in parse_benchmark_specs(root, benchmark_specs):
        csv_files = discover_csv_files(benchmark_dir) if benchmark_dir.exists() else []
        discovered[benchmark_name] = len(csv_files)
        tasks.extend((benchmark_name, str(csv_path)) for csv_path in csv_files)
    return tasks, discovered


def shard_items(items: List[Tuple[str, str]], num_workers: int, worker_id: int) -> List[Tuple[str, str]]:
    return items[worker_id::num_workers]


def normalize_local_ckpt_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    try:
        path = path.resolve()
    except Exception:
        pass
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def collect_torch_diagnostics() -> Dict[str, object]:
    import torch

    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        device_count = f"error: {exc}"

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": device_count,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def get_manyclass_classifier():
    global _MANYCLASS_CLASS
    global _MANYCLASS_IMPORT_ATTEMPTED
    global _MANYCLASS_IMPORT_ERROR

    if _MANYCLASS_IMPORT_ATTEMPTED:
        return _MANYCLASS_CLASS

    _MANYCLASS_IMPORT_ATTEMPTED = True
    try:
        from tabpfn_extensions.manyclass_classifier import ManyClassClassifier

        _MANYCLASS_CLASS = ManyClassClassifier
    except Exception as exc:
        _MANYCLASS_CLASS = None
        _MANYCLASS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

    return _MANYCLASS_CLASS


def build_classifier(model_kwargs: Dict[str, object], n_classes: int):
    from tabpfn import TabPFNClassifier

    worker_kwargs = dict(model_kwargs)
    worker_kwargs["device"] = "cuda:0"
    base_clf = TabPFNClassifier(**worker_kwargs)

    if n_classes <= 10:
        return base_clf, "tabpfn"

    ManyClassClassifier = get_manyclass_classifier()
    if ManyClassClassifier is not None:
        return ManyClassClassifier(estimator=base_clf), "manyclass"

    from sklearn.multiclass import OneVsRestClassifier

    return OneVsRestClassifier(base_clf), "ovr"


def evaluate_one_dataset(
    benchmark: str,
    csv_path: Path,
    *,
    model_kwargs: Dict[str, object],
    test_size: float,
    random_state: int,
) -> ResultRow:
    ensure_runtime_deps()
    dataset_id = sanitize_dataset_id(csv_path)
    try:
        df = pd.read_csv(csv_path)
        target_col = infer_target_column(df)
        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]
        n_classes_raw = int(y.nunique(dropna=True))

        if len(X) < 2:
            raise ValueError("Not enough valid rows after dropping missing target.")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )

        clf, _strategy = build_classifier(model_kwargs, n_classes_raw)

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_pred = clf.predict(X_test)
        try:
            proba = clf.predict_proba(X_test)
            ll = log_loss(y_test, proba, labels=clf.classes_)
        except Exception:
            proba = None
            ll = None
        predict_seconds = time.time() - t1

        # 使用 getattr 获取训练后的类别数，ManyClassClassifier 也支持该属性
        n_classes = int(getattr(clf, "n_classes_", y_train.nunique()))
        
        return ResultRow(
            benchmark=benchmark,
            dataset_id=dataset_id,
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            n_classes=n_classes,
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_weighted=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            logloss=float(ll) if ll is not None else None,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
        )
    except Exception as exc:
        return ResultRow(
            benchmark=benchmark,
            dataset_id=dataset_id,
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=0,
            n_test=0,
            n_features=0,
            n_classes=None,
            accuracy=None,
            f1_weighted=None,
            logloss=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def run_worker(
    worker_id: int,
    gpu_id: int,
    task_items: List[Tuple[str, str]],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    test_size: float,
    random_state: int,
    verbose: bool,
) -> None:
    ensure_runtime_deps()

    try:
        gpu_id_str = str(gpu_id)
        os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import torch
        # 核心修改：导入 ManyClassClassifier 扩展包
        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        
        # 核心逻辑：用 ManyClassClassifier 包装原有的 TabPFNClassifier
        # 当类别数 > 10 时，它会自动采用拆分策略绕过限制

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "status": "ready",
                "assigned_count": len(task_items),
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        for benchmark, csv_path_str in task_items:
            row = evaluate_one_dataset(
                benchmark=benchmark,
                csv_path=Path(csv_path_str),
                model_kwargs=model_kwargs,
                test_size=test_size,
                random_state=random_state,
            )
            rows.append(row)

            if verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[ok] {benchmark}/{row.dataset_name} acc={row.accuracy:.6f} classes={row.n_classes}"
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[fail] {benchmark}/{row.dataset_name} error={row.error}"
                    )

            clear_torch_cache()

        columns = list(ResultRow.__annotations__.keys())
        worker_df = pd.DataFrame([asdict(row) for row in rows]) if rows else pd.DataFrame(columns=columns)
        worker_df.to_csv(worker_out_csv, index=False)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        # 错误兜底
        pd.DataFrame(
            [
                {
                    "benchmark": "__worker__",
                    "dataset_id": f"__WORKER_CRASH__{worker_id}",
                    "dataset_dir": "__worker__",
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "n_train": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "n_classes": None,
                    "accuracy": None,
                    "f1_weighted": None,
                    "logloss": None,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        ).to_csv(worker_out_csv, index=False)


def collect_worker_outputs(out_dir: Path, workers: int) -> List:
    ensure_runtime_deps()
    dfs: List = []
    for worker_id in range(workers):
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        if worker_csv.exists():
            try:
                dfs.append(pd.read_csv(worker_csv))
            except pd.errors.EmptyDataError:
                continue
    return dfs


def write_summary(summary_path: Path, result_df, discovered_datasets: int, wall_seconds: float) -> None:
    ensure_runtime_deps()
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {discovered_datasets}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        (
            f"avg_accuracy_ok: {ok_df['accuracy'].dropna().mean():.6f}"
            if len(ok_df) and ok_df["accuracy"].notna().any()
            else "avg_accuracy_ok: (none)"
        ),
        (
            f"avg_f1_weighted_ok: {ok_df['f1_weighted'].dropna().mean():.6f}"
            if len(ok_df) and ok_df["f1_weighted"].notna().any()
            else "avg_f1_weighted_ok: (none)"
        ),
        (
            f"avg_logloss_ok: {ok_df['logloss'].dropna().mean():.6f}"
            if len(ok_df) and ok_df["logloss"].notna().any()
            else "avg_logloss_ok: (none)"
        ),
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(sorted(set(failed_df["dataset_name"].dropna().astype(str).tolist())))
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabPFN classification benchmarks across multiple benchmark folders "
            "with AMD/ROCm multi-GPU. Now supports 10+ classes via ManyClassClassifier."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--out-dir", default="result/TabPFN_2_5_official_classification_8gpu")
    parser.add_argument(
        "--model-path",
        default="ckpt/TabPFN-2.5/tabpfn-v2.5-classifier-v2.5_default.ckpt",
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

    ensure_runtime_deps()

    root = Path(args.root).expanduser()
    try:
        root = root.resolve()
    except Exception:
        pass
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    model_path = normalize_local_ckpt_path(args.model_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_specs = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    benchmark_names = [name for name, _ in parse_benchmark_specs(root, benchmark_specs)]
    tasks, discovered = build_tasks(root, benchmark_specs)
    if not tasks:
        raise FileNotFoundError("No single-file classification CSVs found in the configured benchmark directories.")

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
    start_event = mp.Event()
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        task_items = shard_items(tasks, args.workers, worker_id)
        proc = mp.Process(
            target=run_worker,
            args=(
                worker_id,
                gpu_ids[worker_id],
                task_items,
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

    dfs = collect_worker_outputs(out_dir, args.workers)
    columns = list(ResultRow.__annotations__.keys())
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)
    all_csv = out_dir / "all_classification_results.csv"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(out_dir / "summary.txt", all_df, sum(discovered.values()), wall_seconds)

    for benchmark in benchmark_names:
        benchmark_dir = out_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_df = all_df[all_df["benchmark"] == benchmark].copy() if len(all_df) else pd.DataFrame(columns=columns)
        benchmark_df.to_csv(benchmark_dir / "all_classification_results.csv", index=False)
        write_summary(benchmark_dir / "summary.txt", benchmark_df, discovered.get(benchmark, 0), wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {out_dir / 'summary.txt'}")
    print("saved_benchmark_summaries:")
    for benchmark in benchmark_names:
        print(f"  {benchmark}: {out_dir / benchmark / 'summary.txt'}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
