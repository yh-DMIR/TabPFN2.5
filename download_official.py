#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_MODEL_NAME = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
ALL_V25_REGRESSOR_MODELS = [
    "tabpfn-v2.5-regressor-v2.5_default.ckpt",
    "tabpfn-v2.5-regressor-v2.5_low-skew.ckpt",
    "tabpfn-v2.5-regressor-v2.5_quantiles.ckpt",
    "tabpfn-v2.5-regressor-v2.5_real-variant.ckpt",
    "tabpfn-v2.5-regressor-v2.5_real.ckpt",
    "tabpfn-v2.5-regressor-v2.5_small-samples.ckpt",
    "tabpfn-v2.5-regressor-v2.5_variant.ckpt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official TabPFN 2.5 regression checkpoints for offline use."
    )
    parser.add_argument("--target-dir", default="ckpt/TabPFN-2.5")
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        help="Checkpoint filename to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all-regression-models",
        action="store_true",
        help="Download all official TabPFN 2.5 regression checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import download_model, get_cache_dir

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    target_dir = Path(args.target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.all_regression_models:
        model_names = list(ALL_V25_REGRESSOR_MODELS)
    elif args.model_names:
        model_names = list(dict.fromkeys(args.model_names))
    else:
        model_names = [DEFAULT_MODEL_NAME]

    logger.info(f"target_dir: {target_dir.resolve()}")
    logger.info(f"cache_dir: {get_cache_dir()}")

    failures: list[tuple[str, str]] = []
    for model_name in model_names:
        dst = target_dir / model_name
        logger.info("")
        logger.info(f"downloading: {model_name}")

        result = download_model(
            to=dst,
            version=ModelVersion.V2_5,
            which="regressor",
            model_name=model_name,
        )
        if result == "ok":
            logger.info(f"saved_to: {dst.resolve()}")
            continue

        cache_candidate = get_cache_dir() / model_name
        if cache_candidate.exists() and not dst.exists():
            shutil.copy2(cache_candidate, dst)
            logger.info(f"saved_to: {dst.resolve()}")
            continue

        message = "; ".join(str(err) for err in result)
        failures.append((model_name, message))
        logger.error(f"failed: {model_name}")
        logger.error(message)

    if failures:
        raise RuntimeError(
            "Failed to download some checkpoints:\n"
            + "\n".join(f"- {name}: {message}" for name, message in failures)
        )


if __name__ == "__main__":
    main()
