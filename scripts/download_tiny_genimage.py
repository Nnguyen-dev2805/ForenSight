#!/usr/bin/env python3
"""Download and prepare TheKernel01/Tiny-GenImage dataset for ForenSight.

Usage:
    python scripts/download_tiny_genimage.py \\
        --output-dir data/raw/tiny_genimage \\
        --manifest-dir data/manifests

This command:
1. Downloads Tiny-GenImage Parquet files from Hugging Face Hub (~8.3GB).
2. Extracts image files into `data/raw/tiny_genimage/<generator>/<split>/`.
3. Builds sealed ForenSight manifests guaranteeing ZERO generator leakage into train/val.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forensight.data.tiny_genimage import (
    DEFAULT_TINY_GENIMAGE_REPO_ID,
    build_tiny_genimage_manifests,
    download_tiny_genimage_parquets,
    extract_parquet_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare Tiny-GenImage dataset for ForenSight."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_TINY_GENIMAGE_REPO_ID,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_TINY_GENIMAGE_REPO_ID}).",
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
        "--token",
        default=None,
        help="Optional Hugging Face authentication token.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading Parquet files if already present in output-dir/_parquets.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.5,
        help="Fraction of validation SD1.4 allocated to val (vs in-domain test) (default: 0.5).",
    )
    return parser.parse_args(args)


def run_download_and_prep(
    repo_id: str = DEFAULT_TINY_GENIMAGE_REPO_ID,
    output_dir: str | Path = "data/raw/tiny_genimage",
    manifest_dir: str | Path = "data/manifests",
    token: str | None = None,
    skip_download: bool = False,
    val_ratio: float = 0.5,
) -> None:
    output_dir = Path(output_dir)
    manifest_dir = Path(manifest_dir)
    parquet_dir = output_dir / "_parquets"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[Path]]
    if skip_download:
        logger.info("Skipping download, scanning existing parquets in %s...", parquet_dir)
        splits = {
            "train": sorted(parquet_dir.glob("*train*.parquet")),
            "validation": sorted(parquet_dir.glob("*validation*.parquet")),
        }
    else:
        logger.info("Downloading Tiny-GenImage Parquet shards from %s...", repo_id)
        splits = download_tiny_genimage_parquets(repo_id=repo_id, output_dir=parquet_dir, token=token)

    total_parquets = len(splits.get("train", [])) + len(splits.get("validation", []))
    logger.info("Found %d Parquet files to extract.", total_parquets)
    if total_parquets == 0:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}.")

    all_records: list[dict] = []
    counter = 0

    for split_name in ("train", "validation"):
        parquet_files = splits.get(split_name, [])
        logger.info("Extracting %d Parquet files for raw split '%s'...", len(parquet_files), split_name)
        for pfile in parquet_files:
            logger.info("  Processing %s...", pfile.name)
            records, counter = extract_parquet_images(
                parquet_path=pfile,
                output_root=output_dir,
                raw_split=split_name,
                start_index=counter,
            )
            all_records.extend(records)

    logger.info("Extracted %d total images to %s.", len(all_records), output_dir)

    # Build manifests
    logger.info("Building sealed ForenSight manifests in %s...", manifest_dir)
    manifests = build_tiny_genimage_manifests(
        extracted_records=all_records,
        manifest_dir=manifest_dir,
        dataset_name="genimage",
        val_in_domain_ratio=val_ratio,
    )

    print("\n" + "=" * 60)
    print("Tiny-GenImage Download & Preparation Complete")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Manifest directory: {manifest_dir}")
    print("\nManifests Created:")
    for name, m in manifests.items():
        print(f"  - {name}.jsonl: {len(m)} records (split='{m[0].split if len(m) else 'empty'}')")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_download_and_prep(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            token=args.token,
            skip_download=args.skip_download,
            val_ratio=args.val_ratio,
        )
        return 0
    except Exception as exc:
        logger.error("Failed to prepare Tiny-GenImage: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
