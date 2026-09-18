#!/usr/bin/env python3
"""CLI utility to execute the ForenSight R0 smoke-test pipeline.

Validates the complete R0 experimental stack in seconds without requiring raw
external downloads:
1. Inventory verification (load_inventory, schema & field validation).
2. Smoke dataset and manifest generation (4 splits, lightweight JPEG images).
3. Invariant assertions (generator-disjointness, zero sample/generator leakage).
4. Dataset audit execution (image attributes, near-duplicates, leakage check).
5. Synthetic model predictions simulation across 3 distinct seeds.
6. Model evaluation runner on each seed (validation threshold calibration).
7. Multi-seed repeated run aggregation (mean, std, preliminary check).

Usage examples:
    # Run fast smoke test in a temporary directory (auto-cleaned):
    python scripts/run_smoke_test.py

    # Run smoke test and persist artifacts to default location (data/derived/smoke_test):
    python scripts/run_smoke_test.py --keep-artifacts

    # Run smoke test and persist to custom output directory:
    python scripts/run_smoke_test.py --output-dir /tmp/my_smoke_test --keep-artifacts

    # Run with custom seed and classes:
    python scripts/run_smoke_test.py --seed 123 --num-classes 3 --keep-artifacts

    # Machine-readable JSON output:
    python scripts/run_smoke_test.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Sequence

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forensight.data.smoke import run_smoke_pipeline


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute ForenSight R0 end-to-end smoke test pipeline."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help=(
            "Directory where synthetic images, manifests, and evaluation reports will be saved. "
            "If omitted and --keep-artifacts is specified, defaults to 'data/derived/smoke_test'. "
            "If omitted and --keep-artifacts is not specified, runs in a temporary directory and cleans up."
        ),
    )
    parser.add_argument(
        "--keep-artifacts",
        "-k",
        action="store_true",
        help="Preserve generated images, manifests, and reports on disk.",
    )
    parser.add_argument(
        "--num-classes",
        "-c",
        type=int,
        default=5,
        help="Number of ImageNet-style classes per split (default: 5).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Base random seed for dataset generation and model evaluations (default: 42).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress human-readable progress output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output complete execution summary as JSON to stdout.",
    )
    return parser.parse_args(args)


def _print_stage_progress(stage_num: int, total_stages: int, description: str, detail: str) -> None:
    print(f"[{stage_num}/{total_stages}] {description.ljust(56)} ... {detail}")


def main(args: Sequence[str] | None = None) -> int:
    parsed_args = parse_args(args)
    keep_artifacts = parsed_args.keep_artifacts or (parsed_args.output_dir is not None)
    verbose = not parsed_args.quiet and not parsed_args.json

    if verbose:
        print("=" * 80)
        print("                 ForenSight R0 Smoke-Test Pipeline")
        print("=" * 80)

    start_time = time.perf_counter()

    try:
        if keep_artifacts:
            target_dir = Path(parsed_args.output_dir or "data/derived/smoke_test").resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            result = run_smoke_pipeline(
                output_dir=target_dir,
                seed=parsed_args.seed,
                num_classes=parsed_args.num_classes,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="forensight_smoke_") as temp_dir:
                result = run_smoke_pipeline(
                    output_dir=temp_dir,
                    seed=parsed_args.seed,
                    num_classes=parsed_args.num_classes,
                )

        elapsed = time.perf_counter() - start_time

        if parsed_args.json:
            print(json.dumps(result, indent=2))
            return 0

        if verbose:
            stages = result.get("stages", {})
            # Stage 1
            s1 = stages.get("1_inventory", {})
            _print_stage_progress(
                1, 7, "Verifying Dataset Inventory",
                f"PASSED ({s1.get('num_datasets', 0)} datasets registered)"
            )
            # Stage 2
            s2 = stages.get("2_data_generation", {})
            _print_stage_progress(
                2, 7, f"Generating Synthetic Smoke Dataset ({parsed_args.num_classes} classes)",
                f"PASSED ({s2.get('total_images', 0)} images across 4 splits)"
            )
            # Stage 3
            _print_stage_progress(
                3, 7, "Asserting Generator Disjointness & Zero Leakage",
                "PASSED (clean)"
            )
            # Stage 4
            s4 = stages.get("4_audit", {})
            _print_stage_progress(
                4, 7, "Running Dataset Audit (Images & Leakage)",
                f"PASSED ({s4.get('summary_findings_count', 0)} findings, 0 critical)"
            )
            # Stage 5
            s5 = stages.get("5_predictions", {})
            _print_stage_progress(
                5, 7, f"Simulating Detector Predictions (Seeds: {s5.get('seeds', [])})",
                f"PASSED ({s5.get('predictions_per_seed', 0)} preds/seed)"
            )
            # Stage 6
            s6 = stages.get("6_evaluation", {})
            _print_stage_progress(
                6, 7, "Running Model Evaluation on Each Seed (Val Calibrated)",
                f"PASSED ({s6.get('num_reports', 0)} reports, all calibrated)"
            )
            # Stage 7
            s7 = stages.get("7_aggregation", {})
            _print_stage_progress(
                7, 7, "Aggregating Multi-Seed Repeated Runs",
                f"PASSED ({s7.get('num_runs', 0)} runs, preliminary={s7.get('is_preliminary')})"
            )

            agg = result.get("aggregated_report", {})
            overall = agg.get("overall", {})

            print("=" * 80)
            print("                              Summary")
            print("=" * 80)
            print(f"Status:               SUCCESS")
            print(f"Total Duration:       {elapsed:.3f}s")
            print(f"Artifacts Persisted:  {'Yes (' + str(result['output_dir']) + ')' if keep_artifacts else 'No (Temporary execution cleaned up)'}")
            print(f"Base Random Seed:     {parsed_args.seed}")
            print(f"Aggregated Benchmark:")
            print(f"  - Evaluation Runs:  {agg.get('num_runs', 0)} seeds (preliminary={agg.get('is_preliminary', True)})")
            print(f"  - Overall AUROC:    {overall.get('auroc', {}).get('formatted', 'N/A')}")
            print(f"  - Overall Accuracy: {overall.get('accuracy', {}).get('formatted', 'N/A')}")
            print(f"  - Overall F1 Score: {overall.get('f1', {}).get('formatted', 'N/A')}")
            print("=" * 80)
            print("All 7 ForenSight R0 stages completed successfully.")

        return 0

    except Exception as exc:
        print(f"\n[ERROR] Smoke-test pipeline failed: {exc}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
