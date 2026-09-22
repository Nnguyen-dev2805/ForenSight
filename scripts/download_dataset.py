#!/usr/bin/env python3
"""Download ForenSight raw dataset archives from Hugging Face.

R0/R1 should start with SD1.4 only:

    python scripts/download_dataset.py --dataset genimage --generator sd14

The command downloads only the selected generator folder from the Hub mirror and
pins the download to the resolved commit SHA. It does not extract the multi-part
archive automatically because extraction can require substantial additional disk
space; the downloaded files and ``source.json`` remain immutable provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forensight.data.hf_download import (
    DEFAULT_GENIMAGE_REPO_ID,
    GENIMAGE_HUB_SPECS,
    download_genimage_subset,
)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selectively download ForenSight datasets from Hugging Face."
    )
    parser.add_argument(
        "--dataset",
        choices=["genimage"],
        default="genimage",
        help="Dataset to download (currently: genimage).",
    )
    parser.add_argument(
        "--generator",
        choices=sorted(GENIMAGE_HUB_SPECS),
        default="sd14",
        help="GenImage generator subset to download (default: sd14).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_GENIMAGE_REPO_ID,
        help=f"Hugging Face dataset mirror (default: {DEFAULT_GENIMAGE_REPO_ID}).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch/tag/commit to resolve and pin (default: main).",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/genimage/_downloads",
        help="Directory for immutable downloaded archives.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent Hub download workers (default: 8).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even when Hugging Face metadata says files are current.",
    )
    return parser.parse_args(args)


def main(cli_args: Sequence[str] | None = None) -> int:
    args = parse_args(cli_args)
    try:
        subset_root, record = download_genimage_subset(
            generator_id=args.generator,
            repo_id=args.repo_id,
            revision=args.revision,
            output_root=args.output_root,
            max_workers=args.max_workers,
            force_download=args.force_download,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    expected_archive_dir = subset_root / record.remote_prefix
    canonical_extract_dir = Path("data/raw/genimage") / record.local_name
    print("Hugging Face download complete.")
    print(f"Repository:        {record.repo_id}")
    print(f"Pinned revision:   {record.resolved_revision}")
    print(f"Generator:         {record.generator_id}")
    print(f"Archive directory: {expected_archive_dir}")
    print(f"Provenance:        {subset_root / 'source.json'}")
    print(f"Extract later to:  {canonical_extract_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
