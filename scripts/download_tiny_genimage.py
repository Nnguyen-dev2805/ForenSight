#!/usr/bin/env python3
"""Download and prepare yangsangtai/tiny-genimage from Kaggle for ForenSight.

Usage:
    python scripts/download_tiny_genimage.py \\
        --output-dir data/raw/tiny_genimage \\
        --manifest-dir data/manifests

This command:
1. Downloads the canonical Tiny-GenImage dataset from Kaggle via kagglehub.
2. Scans the original directory tree without rewriting raw images.
3. Builds sealed ForenSight manifests with SD1.5 train/val/in-domain and six unseen-generator OOD tests.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Sequence

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forensight.data.tiny_genimage import (
    DEFAULT_TINY_GENIMAGE_KAGGLE_DATASET,
    build_tiny_genimage_manifests,
    download_kaggle_tiny_genimage,
    scan_kaggle_tiny_genimage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare Tiny-GenImage dataset for ForenSight."
    )
    parser.add_argument(
        "--dataset-ref",
        default=DEFAULT_TINY_GENIMAGE_KAGGLE_DATASET,
        help=f"Kaggle dataset handle (default: {DEFAULT_TINY_GENIMAGE_KAGGLE_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/tiny_genimage",
        help="Target directory for raw extracted image files (default: data/raw/tiny_genimage).",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests",
        help="Directory to save generated JSONL manifests (default: data/manifests).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download and scan the existing output directory.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force KaggleHub to replace an existing downloaded dataset.",
    )
    parser.add_argument(
        "--experiment",
        choices=["all", "single", "logo", "all7"],
        default="all",
        help="Experiment protocols to generate manifests for (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible partitioning and shuffling (default: 42).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.5,
        help="Fraction of validation SD1.5 allocated to val (vs in-domain test) (default: 0.5).",
    )
    return parser.parse_args(args)


def run_download_and_prep(
    dataset_ref: str = DEFAULT_TINY_GENIMAGE_KAGGLE_DATASET,
    output_dir: str | Path = "data/raw/tiny_genimage",
    manifest_dir: str | Path = "data/manifests",
    skip_download: bool = False,
    force_download: bool = False,
    experiment: str = "all",
    val_ratio: float = 0.5,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    manifest_dir = Path(manifest_dir)

    if skip_download:
        dataset_root = output_dir
        logger.info("Skipping download and scanning existing Kaggle dataset at %s...", dataset_root)
    else:
        logger.info("Downloading Tiny-GenImage from Kaggle dataset %s...", dataset_ref)
        dataset_root = download_kaggle_tiny_genimage(
            dataset_ref=dataset_ref,
            output_dir=output_dir,
            force_download=force_download,
        )

    all_records = scan_kaggle_tiny_genimage(dataset_root)
    logger.info("Scanned %d images from %s.", len(all_records), dataset_root)

    # Build manifests
    logger.info("Building sealed ForenSight manifests in %s (experiment: %s)...", manifest_dir, experiment)
    from forensight.data.tiny_genimage import (
        build_all7_manifests,
        build_all_experiment_manifests,
        build_all_logo_manifests,
        build_single_generator_manifests,
    )

    if experiment == "all":
        build_all_experiment_manifests(
            extracted_records=all_records,
            output_dir=manifest_dir,
            val_in_domain_ratio=val_ratio,
            seed=seed,
        )
        # Also build flat compatibility manifests
        build_tiny_genimage_manifests(
            extracted_records=all_records,
            manifest_dir=manifest_dir,
            dataset_name="genimage",
            val_in_domain_ratio=val_ratio,
            seed=seed,
        )
    elif experiment == "single":
        build_single_generator_manifests(
            extracted_records=all_records,
            manifest_dir=manifest_dir / "single",
            val_in_domain_ratio=val_ratio,
            seed=seed,
        )
    elif experiment == "logo":
        build_all_logo_manifests(
            extracted_records=all_records,
            manifest_dir=manifest_dir / "logo",
            seed=seed,
        )
    elif experiment == "all7":
        build_all7_manifests(
            extracted_records=all_records,
            manifest_dir=manifest_dir / "all7",
            seed=seed,
        )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest_provenance.json").write_text(json.dumps({
        "dataset_ref": dataset_ref,
        "split_seed": seed,
        "skip_download": skip_download,
        "experiment": experiment,
        "scanned_images": len(all_records),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Tiny-GenImage Preparation Complete")
    print("=" * 60)
    print(f"Output directory:   {output_dir}")
    print(f"Manifest directory: {manifest_dir}")
    print(f"Experiment mode:    {experiment}")
    print(f"Random seed:        {seed}")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_download_and_prep(
            dataset_ref=args.dataset_ref,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            skip_download=args.skip_download,
            force_download=args.force_download,
            experiment=args.experiment,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        return 0
    except Exception as exc:
        logger.error("Failed to prepare Tiny-GenImage: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
