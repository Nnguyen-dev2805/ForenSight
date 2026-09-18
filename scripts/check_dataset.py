#!/usr/bin/env python3
"""CLI utility to audit ForenSight dataset manifests for leakage, bias, and distribution confounds.

Usage examples:
    # Audit a single manifest
    python scripts/check_dataset.py --manifest data/manifests/train_sd14.jsonl --output-json report.json --output-md report.md

    # Audit multiple manifests and check cross-split leakage
    python scripts/check_dataset.py --train-manifest data/train.jsonl --eval-manifest data/val.jsonl data/cross_gen.jsonl --output-md report.md --strict
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from forensight.data.audit import (
    AuditReport,
    audit_manifest_images,
    audit_manifest_leakage,
    generate_audit_markdown,
)
from forensight.data.split import Manifest


def load_manifest_from_path(path: str | Path) -> Manifest:
    """Load a Manifest instance from CSV or JSONL path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {p}")
    if p.suffix.lower() == ".csv":
        return Manifest.from_csv(p)
    return Manifest.from_jsonl(p)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ForenSight dataset manifests for leakage, duplication, and literature confounds."
    )
    parser.add_argument(
        "--manifest",
        "-m",
        nargs="+",
        help="One or more manifest file paths (CSV or JSONL) to audit.",
    )
    parser.add_argument(
        "--train-manifest",
        help="Training manifest path for cross-split leakage checks.",
    )
    parser.add_argument(
        "--eval-manifest",
        nargs="+",
        help="Evaluation/test manifest path(s) to verify leakage against train manifest.",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        help="Base directory to prepend to relative image paths in manifest.",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        help="Output path for machine-readable JSON audit report.",
    )
    parser.add_argument(
        "--output-md",
        "-o",
        help="Output path for human-readable Markdown audit report.",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=4,
        help="Number of concurrent worker threads for image inspection (default: 4).",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=2,
        help="Hamming distance threshold for near-duplicate dHash detection (default: 2).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional max number of records to audit (deterministic subsample).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any CRITICAL finding or leakage failure occurs.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console stdout summary.",
    )
    return parser.parse_args(args)


def main(cli_args: Sequence[str] | None = None) -> int:
    args = parse_args(cli_args)

    if not args.manifest and not args.train_manifest:
        print(
            "Error: You must provide either --manifest or --train-manifest (with --eval-manifest).",
            file=sys.stderr,
        )
        return 2

    # 1. Load manifests
    loaded_manifests: list[Manifest] = []
    manifest_names: list[str] = []

    if args.manifest:
        for m_path in args.manifest:
            m = load_manifest_from_path(m_path)
            loaded_manifests.append(m)
            manifest_names.append(Path(m_path).stem)

    train_m: Manifest | None = None
    if args.train_manifest:
        train_m = load_manifest_from_path(args.train_manifest)
        if args.train_manifest not in (args.manifest or []):
            loaded_manifests.append(train_m)
            manifest_names.append(Path(args.train_manifest).stem)

    eval_manifest_map: dict[str, Manifest] = {}
    if args.eval_manifest:
        for e_path in args.eval_manifest:
            em = load_manifest_from_path(e_path)
            eval_manifest_map[Path(e_path).stem] = em
            if e_path not in (args.manifest or []):
                loaded_manifests.append(em)
                manifest_names.append(Path(e_path).stem)

    # Combine records for image attribute auditing
    combined_records = []
    for m in loaded_manifests:
        combined_records.extend(m.records)

    # Subsample if requested
    if args.sample_size is not None and len(combined_records) > args.sample_size:
        import random
        rng = random.Random(42)
        combined_records = rng.sample(combined_records, args.sample_size)
        combined_records.sort(key=lambda r: r.sample_id)

    # 2. Check cross-split leakage if train and eval manifests provided
    leakage_results = None
    if train_m is not None and eval_manifest_map:
        leakage_results = audit_manifest_leakage(
            train_manifest=train_m,
            eval_manifests=eval_manifest_map,
            base_dir=args.base_dir,
            check_generators=True,
        )

    # 3. Audit image records
    audit_name = "+".join(manifest_names) if manifest_names else "dataset_audit"
    report = audit_manifest_images(
        manifest=combined_records,
        base_dir=args.base_dir,
        manifest_name=audit_name,
        max_workers=args.max_workers,
        compute_phash=True,
        near_duplicate_threshold=args.near_duplicate_threshold,
        leakage_check_results=leakage_results,
    )

    # 4. Print summary if not quiet
    if not args.quiet:
        print(f"\n=======================================================")
        print(f" ForenSight Dataset Audit: {report.manifest_name}")
        print(f"=======================================================")
        print(f"Total Samples Audited: {report.total_samples} (Real: {report.num_real}, Fake: {report.num_fake})")
        print(f"Exact Duplicates (SHA256): {len(report.duplicate_check)} group(s)")
        print(f"Near Duplicates (dHash):   {len(report.near_duplicate_check)} pair(s)")
        print(f"Split Collisions:          {len(report.sample_leakage)} collision(s)")
        print(f"Generator Overlaps:        {len(report.generator_leakage)} leak(s)")
        print(f"-------------------------------------------------------")
        print("Findings:")
        for f in report.summary_findings:
            badge = f"[{f.get('status', 'INFO')}]"
            print(f"  {badge:<10} {f.get('check')}: {f.get('message')}")
        print(f"-------------------------------------------------------")

    # 5. Output files
    if args.output_json:
        report.save_json(args.output_json)
        if not args.quiet:
            print(f"JSON report saved to: {args.output_json}")

    if args.output_md:
        report.save_markdown(args.output_md)
        if not args.quiet:
            print(f"Markdown report saved to: {args.output_md}")

    # 6. Exit code handling
    if args.strict:
        if report.has_critical_findings() or not report.passed():
            if not args.quiet:
                print("\n[STRICT MODE FAILURE] Critical findings or leakage violations detected.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
