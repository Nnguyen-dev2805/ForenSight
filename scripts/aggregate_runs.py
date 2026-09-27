#!/usr/bin/env python3
"""CLI utility to aggregate multiple ForenSight evaluation runs across seeds.

Enforces ForenSight R0 evaluation invariants:
- Aggregates multiple single-run evaluation reports (JSON).
- Quantifies uncertainty as sample mean ± sample standard deviation (ddof=1 for n >= 2, 0.0 for n = 1).
- Flags runs with fewer than 3 seeds as preliminary (is_preliminary: True).
- Outputs machine-readable aggregated JSON and human-readable Markdown reports.

Usage examples:
    # Aggregate 3 repeated runs across seeds
    python scripts/aggregate_runs.py \\
        --reports results/run_seed42.json results/run_seed43.json results/run_seed44.json \\
        --output-json results/aggregated_sd15.json \\
        --output-md results/aggregated_sd15.md

    # Using --report alias with custom report name
    python scripts/aggregate_runs.py \\
        --report results/s1.json results/s2.json results/s3.json \\
        --report-name "SD1.5 Baseline 3-Seed" \\
        --output-json results/sd15_sealed.json
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

from forensight.evaluation.aggregate import aggregate_reports
from forensight.evaluation.runner import EvaluationReport


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ForenSight Multi-Seed Evaluation Aggregator (R0 Protocol Compliant)."
    )
    parser.add_argument(
        "--reports",
        "--report",
        "-r",
        nargs="+",
        required=True,
        help="Paths to single-run EvaluationReport JSON files to aggregate.",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        help="Path to save machine-readable aggregated JSON report.",
    )
    parser.add_argument(
        "--output-md",
        "-o",
        help="Path to save human-readable Markdown aggregated report.",
    )
    parser.add_argument(
        "--report-name",
        "-n",
        default="ForenSight Multi-Seed Evaluation",
        help="Title / identifier for this aggregated evaluation report.",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print full Markdown report to stdout instead of summary table.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output.",
    )
    return parser.parse_args(args)


def main(cli_args: Sequence[str] | None = None) -> int:
    """CLI entrypoint for aggregating ForenSight evaluation runs."""
    args = parse_args(cli_args)
    report_paths: list[str] = list(args.reports)
    if not report_paths:
        print("Error: No report files specified.", file=sys.stderr)
        return 1

    # 1. Load evaluation reports from disk
    reports: list[EvaluationReport] = []
    for p_str in report_paths:
        p = Path(p_str)
        if not p.exists():
            print(f"Error: Report file not found: {p}", file=sys.stderr)
            return 1
        try:
            content = p.read_text(encoding="utf-8")
            report = EvaluationReport.from_json(content)
            reports.append(report)
        except Exception as exc:
            print(f"Error reading report {p}: {exc}", file=sys.stderr)
            return 1

    # 2. Aggregate reports
    try:
        agg_report = aggregate_reports(
            reports=reports,
            metadata={"report_name": args.report_name},
        )
    except Exception as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        return 1

    # 3. Output to console
    if not args.quiet:
        if args.print_markdown:
            print(agg_report.generate_markdown())
        else:
            status_str = (
                "⚠️ PRELIMINARY (< 3 seeds)"
                if agg_report.is_preliminary
                else "✅ SEALED BENCHMARK (≥ 3 seeds)"
            )

            print("\n=======================================================")
            print(f" ForenSight Multi-Seed Aggregation: {args.report_name}")
            print("=======================================================")
            print(f"Runs Aggregated:    {agg_report.num_runs} (Seeds: {agg_report.seeds})")
            print(f"Benchmark Status:   {status_str}")
            print("Uncertainty Metric: mean ± std (ddof=1 for N ≥ 2, 0.0 for N = 1)")
            print("-------------------------------------------------------")
            print("Overall Benchmark Performance:")
            auroc_str = (
                agg_report.overall.auroc.formatted
                if agg_report.overall.auroc is not None
                else "N/A"
            )
            print(f"  AUROC:     {auroc_str}")
            print(f"  Accuracy:  {agg_report.overall.accuracy.formatted}")
            print(f"  F1 Score:  {agg_report.overall.f1.formatted}")
            print(f"  Precision: {agg_report.overall.precision.formatted}")
            print(f"  Recall:    {agg_report.overall.recall.formatted}")

            if agg_report.by_split:
                print("-------------------------------------------------------")
                print("Performance by Split:")
                for s, sl in sorted(agg_report.by_split.items()):
                    s_auc = sl.auroc.formatted if sl.auroc is not None else "N/A"
                    print(
                        f"  {s:<20} AUROC: {s_auc:<18} Acc: {sl.accuracy.formatted:<18} F1: {sl.f1.formatted}"
                    )

            if agg_report.by_generator:
                print("-------------------------------------------------------")
                print("Performance by Generator:")
                for g, sl in sorted(agg_report.by_generator.items()):
                    g_auc = sl.auroc.formatted if sl.auroc is not None else "N/A"
                    print(
                        f"  {g:<20} AUROC: {g_auc:<18} Acc: {sl.accuracy.formatted:<18} Rec: {sl.recall.formatted}"
                    )

            if agg_report.by_dataset:
                print("-------------------------------------------------------")
                print("Performance by Dataset:")
                for d, sl in sorted(agg_report.by_dataset.items()):
                    d_auc = sl.auroc.formatted if sl.auroc is not None else "N/A"
                    print(
                        f"  {d:<20} AUROC: {d_auc:<18} Acc: {sl.accuracy.formatted:<18} F1: {sl.f1.formatted}"
                    )
            print("=======================================================\n")

    # 4. Save files if requested
    if args.output_json:
        agg_report.save_json(args.output_json)
        if not args.quiet:
            print(f"Saved aggregated JSON report to: {args.output_json}")

    if args.output_md:
        agg_report.save_markdown(args.output_md)
        if not args.quiet:
            print(f"Saved aggregated Markdown report to: {args.output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
