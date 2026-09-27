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
        "--experiment",
        type=str,
        choices=["single", "logo", "all7"],
        default=None,
        help="Experiment protocol paradigm: 'single', 'logo', or 'all7'.",
    )
    parser.add_argument(
        "--leave-out",
        type=str,
        default=None,
        help="Held-out generator architecture for LOGO experiment.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests",
        help="Root directory for resolved manifests (default: data/manifests).",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
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
        default=None,
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
        "--out-dir",
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
        default=None,
        help="Exact dataset revision for saved reproducibility reports.",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.output_dir or args.repro_json) and not args.dataset_revision:
        raise ValueError("--dataset-revision is required when saving a reproducibility report")

    # 1. Resolve experiment mode and manifest paths
    manifest_base = Path(args.manifest_dir)
    train_manifest_path: Path | None = None
    val_manifest_path: Path | None = None
    eval_manifest_paths: list[str] = []

    if args.experiment == "single":
        train_manifest_path = manifest_base / "single" / "train.jsonl"
        val_manifest_path = manifest_base / "single" / "val.jsonl"
        eval_manifest_paths = [
            str(manifest_base / "single" / "in_domain_test.jsonl"),
            str(manifest_base / "single" / "cross_generator_ood.jsonl"),
        ]
    elif args.experiment == "logo":
        if not args.leave_out:
            raise ValueError("--leave-out is required when --experiment is 'logo'")
        train_manifest_path = manifest_base / "logo" / f"leave_{args.leave_out}" / "train.jsonl"
        val_manifest_path = manifest_base / "logo" / f"leave_{args.leave_out}" / "val.jsonl"
        eval_manifest_paths = [
            str(manifest_base / "logo" / f"leave_{args.leave_out}" / "test_in_domain_seen.jsonl"),
            str(manifest_base / "logo" / f"leave_{args.leave_out}" / f"test_{args.leave_out}.jsonl"),
        ]
    elif args.experiment == "all7":
        train_manifest_path = manifest_base / "all7" / "train.jsonl"
        val_manifest_path = manifest_base / "all7" / "val.jsonl"
        eval_manifest_paths = [str(manifest_base / "all7" / "test_all_combined.jsonl")]

    if args.train_manifest:
        train_manifest_path = Path(args.train_manifest)
    if args.val_manifest:
        val_manifest_path = Path(args.val_manifest)
    if args.eval_manifest:
        eval_manifest_paths = list(args.eval_manifest)

    if not val_manifest_path or not eval_manifest_paths:
        raise ValueError(
            "Missing manifest paths. Provide either --experiment {single,logo,all7} or both --val-manifest and --eval-manifest."
        )

    from forensight.data.audit import audit_manifest_leakage
    from forensight.data.split import Manifest

    out_dir = Path(args.output_dir) if args.output_dir else None
    train_path = None
    if train_manifest_path and Path(train_manifest_path).exists():
        train_path = Path(train_manifest_path)
    elif out_dir and (out_dir / "train_manifest.jsonl").exists():
        train_path = out_dir / "train_manifest.jsonl"
    train_m = load_manifest_file(train_path) if train_path else None
    val_m = load_manifest_file(val_manifest_path)
    eval_m_dict = {Path(path).stem: load_manifest_file(path) for path in eval_manifest_paths}
    eval_splits_to_check = {"val": val_m, **eval_m_dict}
    leakage_check = audit_manifest_leakage(
        train_m if train_m is not None else val_m,
        eval_splits_to_check if train_m is not None else eval_m_dict,
        base_dir=args.base_dir,
        check_content_hashes=True,
    )
    if leakage_check["has_leakage"]:
        raise RuntimeError("Evaluation leakage audit FAILED: " + "; ".join(leakage_check["violations"]))

    config = load_r2_config(args.config)
    device = torch.device(args.device) if args.device else select_device()
    print(f"Using compute device: {device}")

    variant = config["variant"]
    print(f"Assembling model for variant: {variant}")
    model, clip_transform, forensic_transform = build_r2_model(config)
    checkpoint_state = load_checkpoint(args.checkpoint, model, map_location=device)
    checkpoint_config = checkpoint_state.get("config")
    if isinstance(checkpoint_config, dict):
        checkpoint_variant = checkpoint_config.get("variant")
        if checkpoint_variant is not None and checkpoint_variant != variant:
            raise ValueError(
                f"Checkpoint was trained as variant {checkpoint_variant!r} but the supplied "
                f"config declares {variant!r}."
            )
        if checkpoint_config.get("model") != config["model"]:
            raise ValueError("Checkpoint model config differs from the supplied evaluation model config")
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
    val_predictions = predict_manifest(val_manifest_path)

    eval_prediction_sets: list[PredictionSet] = []
    print(f"Generating predictions for {len(eval_manifest_paths)} evaluation manifests...")
    for eval_path in eval_manifest_paths:
        eval_prediction_sets.append(predict_manifest(eval_path))

    # Combine into a single PredictionSet
    all_records = list(val_predictions.records)
    for pset in eval_prediction_sets:
        all_records.extend(pset.records)
    combined = PredictionSet(all_records)

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
    exp_name = args.experiment or f"r2_{variant}"
    eval_run_meta: dict[str, Any] = {
        "variant": variant,
        "model_name": variant,
        "model": variant,
        "architecture": config.get("model", {}).get("architecture", variant),
        "experiment": exp_name,
        "experiment_name": exp_name,
        "split_version": args.split_version,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_state.get("epoch"),
    }
    report = evaluate_predictions(
        combined,
        val_split_name="val",
        threshold_strategy=threshold_strategy,
        seed=config.get("seed"),
        run_metadata=eval_run_meta,
    )

    from forensight.evaluation.metrics import compute_metrics
    import numpy as np

    test_only = PredictionSet([r for r in combined if r.split not in ("val", "validation", "valid")])
    default_metrics = None
    if len(test_only) > 0 and len(np.unique(test_only.y_true)) >= 2:
        default_metrics = compute_metrics(
            test_only.y_true,
            test_only.y_scores,
            threshold=0.5,
            threshold_source="default_0.50",
        )

    if args.predictions_out:
        out_path = Path(args.predictions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save evaluated test records (strictly excluding validation partition)
        test_only.to_jsonl(out_path)
        print(f"Saved {len(test_only)} test prediction records to {out_path}")

    if args.report_json:
        report_json_path = Path(args.report_json)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = report.to_dict()
        if default_metrics is not None:
            thresh_val_meta = report.threshold_metadata.get("threshold_value", report.threshold_metadata["threshold"])
            report_data["dual_threshold_comparison"] = {
                "calibrated_threshold": {
                    "threshold": float(thresh_val_meta),
                    "source": str(report.threshold_metadata["threshold_source"]),
                    "accuracy": float(report.overall.accuracy),
                    "f1": float(report.overall.f1),
                    "precision": float(report.overall.precision),
                    "recall": float(report.overall.recall),
                },
                "default_threshold": {
                    "threshold": 0.50,
                    "source": "default_0.50",
                    "accuracy": float(default_metrics.accuracy),
                    "f1": float(default_metrics.f1),
                    "precision": float(default_metrics.precision),
                    "recall": float(default_metrics.recall),
                },
            }
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
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
            **({"git_commit": config["git_commit"]} if "git_commit" in config else {}),
        )
        repro.save_json(repro_path)
        print(f"Saved reproducibility record to {repro_path}")

    if out_dir:
        # Persist the manifests and the audit that passed before inference.
        val_m.to_jsonl(out_dir / "val_manifest.jsonl")

        combined_test_m = Manifest([record for manifest in eval_m_dict.values() for record in manifest])
        combined_test_m.to_jsonl(out_dir / "test_manifest.jsonl")

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
    print(f"Overall AUROC:       {report.overall.auroc}")
    if report.overall.eer is not None:
        print(f"Overall EER:         {report.overall.eer:.4f}")
    if report.overall.pr_auc is not None:
        print(f"Overall PR-AUC:      {report.overall.pr_auc:.4f}")
    if report.overall.tpr_at_1pct_fpr is not None:
        print(f"Overall TPR@1%FPR:   {report.overall.tpr_at_1pct_fpr:.4f}")
    print(f"\n[At Calibrated Threshold tau* = {thresh_val:.4f}] ({report.threshold_metadata['threshold_source']})")
    print(f"  Accuracy:          {report.overall.accuracy:.4f}")
    print(f"  F1:                {report.overall.f1:.4f}")
    print(f"  Recall (TPR):      {report.overall.recall:.4f}")
    print(f"  Precision:         {report.overall.precision:.4f}")
    if default_metrics is not None:
        print(f"\n[At Default Baseline Threshold tau = 0.5000] (default_0.50)")
        print(f"  Accuracy:          {default_metrics.accuracy:.4f}")
        print(f"  F1:                {default_metrics.f1:.4f}")
        print(f"  Recall (TPR):      {default_metrics.recall:.4f}")
        print(f"  Precision:         {default_metrics.precision:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
