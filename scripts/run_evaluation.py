#!/usr/bin/env python3
"""CLI utility to evaluate AI-generated image detector predictions for ForenSight.

Enforces ForenSight R0 evaluation invariants:
- Evaluates model-agnostic prediction records (CSV or JSONL).
- Calibrates decision threshold strictly on validation partition (or default if absent).
- Preserves threshold invariance across test/benchmark distributions.
- Outputs machine-readable JSON and human-readable Markdown evaluation reports.

Usage examples:
    # Single prediction file containing val and test splits
    python scripts/run_evaluation.py --predictions data/preds/baseline.jsonl \
        --val-split val --threshold-strategy f1 \
        --output-json results/report.json --output-md results/report.md

    # Separate validation and test prediction files
    python scripts/run_evaluation.py --predictions data/preds/test_ood.csv \
        --val-predictions data/preds/val.csv \
        --threshold-strategy youden --output-json results/ood_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

# Ensure src/ is on sys.path for direct script execution
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forensight.evaluation.metrics import VALID_STRATEGIES
from forensight.evaluation.reproducibility import create_reproducibility_record
from forensight.evaluation.runner import (
    EvaluationReport,
    PredictionSet,
    evaluate_predictions,
)


def load_prediction_set(path: str | Path) -> PredictionSet:
    """Load a PredictionSet from CSV or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return PredictionSet.from_csv(p)
    elif suffix in (".jsonl", ".ndjson"):
        return PredictionSet.from_jsonl(p)
    else:
        # Auto-detect based on first line format
        try:
            return PredictionSet.from_jsonl(p)
        except Exception:
            return PredictionSet.from_csv(p)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ForenSight Model-Agnostic Evaluation Runner (R0 Protocol Compliant)."
    )
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="Path to prediction records file (CSV or JSONL).",
    )
    parser.add_argument(
        "--val-predictions",
        help="Optional path to separate validation predictions file (CSV or JSONL) for threshold calibration.",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        help="Name of the validation split within the predictions file (default: 'val').",
    )
    parser.add_argument(
        "--threshold-strategy",
        "-s",
        choices=sorted(VALID_STRATEGIES),
        default="f1",
        help="Criterion to optimize decision threshold on validation data (default: 'f1').",
    )
    parser.add_argument(
        "--default-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Default fallback threshold if validation partition is absent or single-class (default: 0.5).",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        help="Optional explicit list of split names to include in overall benchmark performance.",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        help="Path to save machine-readable JSON evaluation report.",
    )
    parser.add_argument(
        "--output-md",
        "-o",
        help="Path to save human-readable Markdown evaluation report.",
    )
    parser.add_argument(
        "--save-reproducibility",
        "-r",
        help="Path to save JSON reproducibility record conforming to ForenSight Section 12 schema.",
    )
    parser.add_argument(
        "--split-version",
        default="r0-default",
        help="Version tag or identifier of the dataset split (default: 'r0-default').",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        help="Optional path to configuration file (JSON) to embed in the reproducibility record.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional experiment notes or rationale to store in the reproducibility record.",
    )
    parser.add_argument(
        "--run-name",
        "-n",
        help="Optional identifier or name for this evaluation run.",
    )
    parser.add_argument(
        "--seed",
        help="Random seed for this run (e.g. 42).",
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
    """CLI entrypoint for running ForenSight evaluations."""
    args = parse_args(cli_args)

    # 1. Load predictions
    try:
        preds = load_prediction_set(args.predictions)
    except Exception as exc:
        print(f"Error loading predictions from {args.predictions}: {exc}", file=sys.stderr)
        return 1

    val_preds: PredictionSet | None = None
    if args.val_predictions:
        try:
            val_preds = load_prediction_set(args.val_predictions)
        except Exception as exc:
            print(
                f"Error loading validation predictions from {args.val_predictions}: {exc}",
                file=sys.stderr,
            )
            return 1

    # 2. Run evaluation
    run_name = args.run_name or Path(args.predictions).stem
    run_meta = {"run_name": run_name, "predictions_path": str(args.predictions)}
    if args.val_predictions:
        run_meta["val_predictions_path"] = str(args.val_predictions)

    seed_val: int | str | None = None
    if args.seed is not None:
        try:
            seed_val = int(args.seed)
        except ValueError:
            seed_val = str(args.seed).strip()

    try:
        report = evaluate_predictions(
            predictions=preds,
            val_split_name=args.val_split,
            threshold_strategy=args.threshold_strategy,
            default_threshold=args.default_threshold,
            val_predictions=val_preds,
            eval_splits=args.eval_splits,
            run_metadata=run_meta,
            seed=seed_val,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    # 3. Print output to console
    if not args.quiet:
        if args.print_markdown:
            print(report.generate_markdown())
        else:
            thresh = report.threshold_metadata.get("threshold", report.overall.threshold)
            source = report.threshold_metadata.get("threshold_source", report.overall.threshold_source)
            calibrated = report.threshold_metadata.get("calibrated", False)
            calib_status = "CALIBRATED (Validation)" if calibrated else "UNSET (Default Fallback)"

            print("\n=======================================================")
            print(f" ForenSight Evaluation: {run_name}")
            print("=======================================================")
            print(f"Decision Threshold: {thresh:.4f} [{calib_status}] (Source: {source})")
            print(f"Evaluated Samples:  {report.run_metadata.get('evaluated_samples', len(preds)):,}")
            print("-------------------------------------------------------")
            print("Overall Benchmark Performance:")
            auroc_str = f"{report.overall.auroc:.4f}" if report.overall.auroc is not None else "N/A"
            print(f"  AUROC:     {auroc_str}")
            print(f"  Accuracy:  {report.overall.accuracy:.4f}")
            print(f"  F1 Score:  {report.overall.f1:.4f}")
            print(f"  Precision: {report.overall.precision:.4f}")
            print(f"  Recall:    {report.overall.recall:.4f}")
            cm = report.overall.confusion_matrix
            print(f"  Confusion: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")

            if report.by_split:
                print("-------------------------------------------------------")
                print("Performance by Split:")
                for s, m in sorted(report.by_split.items()):
                    s_auc = f"{m.auroc:.4f}" if m.auroc is not None else "N/A"
                    print(f"  {s:<20} AUROC: {s_auc:<8} Acc: {m.accuracy:.4f}  F1: {m.f1:.4f}")

            if report.by_generator:
                print("-------------------------------------------------------")
                print("Performance by Generator:")
                for g, m in sorted(report.by_generator.items()):
                    g_auc = f"{m.auroc:.4f}" if m.auroc is not None else "N/A"
                    print(f"  {g:<20} AUROC: {g_auc:<8} Acc: {m.accuracy:.4f}  Recall: {m.recall:.4f}")
            print("=======================================================\n")

    # Config resolution
    config_dict: dict[str, Any] = {
        "predictions": str(args.predictions),
        "val_predictions": str(args.val_predictions) if args.val_predictions else None,
        "val_split": args.val_split,
        "threshold_strategy": args.threshold_strategy,
        "default_threshold": args.default_threshold,
        "eval_splits": args.eval_splits,
        "seed": seed_val,
    }
    if args.config_file:
        cfg_path = Path(args.config_file)
        if not cfg_path.exists():
            print(f"Error loading config file: File does not exist: {cfg_path}", file=sys.stderr)
            return 1
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded_cfg = json.load(f)
            if not isinstance(loaded_cfg, dict):
                print(
                    f"Error: Config file must contain a JSON dictionary, got: {type(loaded_cfg).__name__}",
                    file=sys.stderr,
                )
                return 1
            config_dict = {**loaded_cfg, **config_dict}
        except Exception as exc:
            print(f"Error parsing config file {cfg_path}: {exc}", file=sys.stderr)
            return 1

    # 4. Save report files if requested
    if args.output_json:
        report.save_json(args.output_json)
        if not args.quiet:
            print(f"Saved JSON report to: {args.output_json}")

    if args.output_md:
        report.save_markdown(args.output_md)
        if not args.quiet:
            print(f"Saved Markdown report to: {args.output_md}")

    # 5. Save reproducibility record if requested
    if args.save_reproducibility:
        repro_rec = create_reproducibility_record(
            run_id=run_name,
            experiment_name=args.run_name or run_name,
            split_version=args.split_version,
            config=config_dict,
            report=report,
            seed=seed_val,
            notes=args.notes,
        )
        repro_rec.save_json(args.save_reproducibility)
        if not args.quiet:
            print(f"Saved reproducibility record to: {args.save_reproducibility}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
