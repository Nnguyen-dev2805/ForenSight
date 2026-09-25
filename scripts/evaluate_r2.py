#!/usr/bin/env python3
"""Evaluation entrypoint for ForenSight R2 detector variants.

Loads a trained model checkpoint, generates predictions across validation
and evaluation (test/OOD) manifests, and integrates directly with the R0
leak-free evaluation engine. Decision thresholds are calibrated strictly
on validation data and frozen when scoring test sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from forensight.data.r2_dataset import R2ImageDataset, load_manifest_file
from forensight.evaluation.reproducibility import create_reproducibility_record
from forensight.evaluation.runner import PredictionSet, evaluate_predictions
from forensight.training.r2 import (
    build_r2_model,
    load_checkpoint,
    load_r2_config,
    predict_to_prediction_set,
    select_device,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ForenSight R2 checkpoint using R0 leak-free evaluation protocol."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to R2 JSON experiment configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        required=True,
        help="Path to validation manifest (JSONL/CSV) for threshold calibration.",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Optional path to training manifest (JSONL/CSV) to verify train ↔ test and train ↔ val leakage.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=str,
        nargs="+",
        required=True,
        help="One or more evaluation test/OOD manifests (JSONL/CSV).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for resolving relative image paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to automatically save all run artifacts (predictions, evaluation, reproducibility, manifests).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Tiny-GenImage",
        help="Name of dataset evaluated (default: Tiny-GenImage).",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default="hf:TheKernel01/Tiny-GenImage@v1.0",
        help="Dataset revision or commit SHA for exact provenance tracking.",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default=None,
        help="Optional path to export combined PredictionSet (.jsonl).",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to save EvaluationReport as JSON.",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default=None,
        help="Optional path to save EvaluationReport as Markdown.",
    )
    parser.add_argument(
        "--repro-json",
        type=str,
        default=None,
        help="Optional path to save ReproducibilityRecord as JSON.",
    )
    parser.add_argument(
        "--split-version",
        type=str,
        default="r2-sealed",
        help="Dataset split version tag for reproducibility tracking.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g. cpu, cuda, mps). Defaults to auto-selection.",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    config = load_r2_config(args.config)
    device = torch.device(args.device) if args.device else select_device()
    print(f"Using compute device: {device}")

    variant = config["variant"]
    print(f"Assembling model for variant: {variant}")
    model, clip_transform, forensic_transform = build_r2_model(config)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device)
    model.eval()

    batch_size = config["training"].get("batch_size", 16)
    num_workers = config["training"].get("num_workers", 0)

    def predict_manifest(manifest_path: str | Path) -> PredictionSet:
        path = Path(manifest_path)
        print(f"  Predicting manifest: {path.name} ...")
        manifest = load_manifest_file(path)
        dataset = R2ImageDataset(
            manifest,
            base_dir=args.base_dir,
            clip_transform=clip_transform,
            forensic_transform=forensic_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return predict_to_prediction_set(model, loader, device=device, variant=variant)

    print("Generating validation predictions (for threshold calibration)...")
    val_predictions = predict_manifest(args.val_manifest)

    eval_prediction_sets: list[PredictionSet] = []
    print(f"Generating predictions for {len(args.eval_manifest)} evaluation manifests...")
    for eval_path in args.eval_manifest:
        eval_prediction_sets.append(predict_manifest(eval_path))

    # Combine into a single PredictionSet
    all_records = list(val_predictions.records)
    for pset in eval_prediction_sets:
        all_records.extend(pset.records)
    combined = PredictionSet(all_records)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.predictions_out:
            args.predictions_out = str(out_dir / "predictions.jsonl")
        if not args.report_json:
            args.report_json = str(out_dir / "evaluation.json")
        if not args.report_md:
            args.report_md = str(out_dir / "evaluation.md")
        if not args.repro_json:
            args.repro_json = str(out_dir / "reproducibility.json")

    # Evaluate using the leak-free R0 runner (must run before saving predictions to calibrate tau_star)
    threshold_strategy = config.get("evaluation", {}).get("threshold_strategy", "f1")
    print(f"Running leak-free evaluation (val calibration strategy='{threshold_strategy}')...")
    report = evaluate_predictions(
        combined,
        val_split_name="val",
        threshold_strategy=threshold_strategy,
        seed=config.get("seed"),
    )

    if args.predictions_out:
        out_path = Path(args.predictions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_jsonl(out_path)
        print(f"Saved {len(combined)} prediction records to {out_path}")

    if args.report_json:
        report_json_path = Path(args.report_json)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_json(report_json_path)
        print(f"Saved evaluation JSON report to {report_json_path}")

    if args.report_md:
        report_md_path = Path(args.report_md)
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_markdown(report_md_path)
        print(f"Saved evaluation Markdown report to {report_md_path}")

    if args.repro_json:
        repro_path = Path(args.repro_json)
        repro_path.parent.mkdir(parents=True, exist_ok=True)
        repro = create_reproducibility_record(
            experiment_name=f"r2_{variant}",
            split_version=args.split_version,
            config=config,
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            threshold_source=report.threshold_metadata["threshold_source"],
            threshold_value=report.threshold_metadata.get("threshold_value", report.threshold_metadata["threshold"]),
            metrics=report.to_dict(),
            seed=config.get("seed"),
        )
        repro.save_json(repro_path)
        print(f"Saved reproducibility record to {repro_path}")

    if out_dir:
        from forensight.data.audit import audit_manifest_leakage
        from forensight.data.split import Manifest

        # 1. Resolve train manifest if available (from CLI or pre-existing in out_dir from train_r2.py)
        train_m = None
        train_path = None
        if args.train_manifest and Path(args.train_manifest).exists():
            train_path = Path(args.train_manifest)
        elif (out_dir / "train_manifest.jsonl").exists():
            train_path = out_dir / "train_manifest.jsonl"

        if train_path is not None:
            train_m = load_manifest_file(train_path)

        # 2. Persist val and combined test manifests into out_dir
        val_m = load_manifest_file(args.val_manifest)
        val_m.to_jsonl(out_dir / "val_manifest.jsonl")

        all_eval_records = []
        eval_m_dict = {}
        for p in args.eval_manifest:
            m = load_manifest_file(p)
            eval_m_dict[Path(p).stem] = m
            all_eval_records.extend(m.records)

        combined_test_m = Manifest(all_eval_records)
        combined_test_m.to_jsonl(out_dir / "test_manifest.jsonl")

        # 3. Comprehensive pairwise leakage audit across train, val, and all test partitions
        eval_splits_to_check = dict(eval_m_dict)
        eval_splits_to_check["val"] = val_m

        if train_m is not None:
            leakage_check = audit_manifest_leakage(train_m, eval_splits_to_check)
        else:
            leakage_check = audit_manifest_leakage(val_m, eval_m_dict)

        audit_data: dict[str, Any] = {
            "val_samples": len(val_m),
            "test_samples": len(combined_test_m),
            "val_reals": len(val_m.filter(label=0)),
            "val_fakes": len(val_m.filter(label=1)),
            "test_reals": len(combined_test_m.filter(label=0)),
            "test_fakes": len(combined_test_m.filter(label=1)),
            "val_generators": sorted(list({r.generator for r in val_m if r.generator})),
            "test_generators": sorted(list({r.generator for r in combined_test_m if r.generator})),
            "leakage": leakage_check,
        }

        if train_m is not None:
            audit_data["train_samples"] = len(train_m)
            audit_data["train_reals"] = len(train_m.filter(label=0))
            audit_data["train_fakes"] = len(train_m.filter(label=1))
            audit_data["train_generators"] = sorted(list({r.generator for r in train_m if r.generator}))

        audit_path = out_dir / "dataset_audit.json"
        with audit_path.open("w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2)
        print(f"Saved complete unified dataset audit and manifests to {out_dir}")

    thresh_val = report.threshold_metadata.get("threshold_value", report.threshold_metadata["threshold"])
    print("\n--- Evaluation Summary ---")
    print(f"Overall AUROC:    {report.overall.auroc}")
    print(f"Overall Accuracy: {report.overall.accuracy:.4f}")
    print(f"Overall F1:       {report.overall.f1:.4f}")
    print(f"Threshold:        {thresh_val:.4f} ({report.threshold_metadata['threshold_source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
