#!/usr/bin/env python3
"""Training entrypoint for ForenSight R2 detector variants.

Orchestrates training on sealed Stage-1 train and validation manifests,
saving the best checkpoint according to validation AUROC (with fallback to validation loss).
Test or OOD manifests are strictly prohibited from this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from forensight.data.r2_dataset import (
    R2ImageDataset,
    load_manifest_file,
)
from forensight.training.r2 import (
    build_r2_model,
    load_r2_config,
    save_checkpoint,
    select_device,
    set_seed,
    train_one_epoch,
    validate_one_epoch,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ForenSight R2 detector variant on sealed Stage-1 train/val manifests."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to R2 JSON experiment configuration. Defaults to configs/r2/<variant>.json if omitted.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["single", "logo", "all7"],
        default=None,
        help="Experiment protocol paradigm: 'single' (SD1.5 train -> 6 OOD), 'logo' (Leave-One-Generator-Out), or 'all7' (All-in-one upper bound).",
    )
    parser.add_argument(
        "--leave-out",
        type=str,
        default=None,
        help="Held-out generator architecture for LOGO experiment (e.g. 'midjourney', 'adm', etc.).",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests",
        help="Root directory for resolved manifests (default: data/manifests).",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Path to sealed training manifest (JSONL/CSV). Overrides --experiment default if provided.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Path to sealed validation manifest (JSONL/CSV). Overrides --experiment default if provided.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["fusion", "semantic_only", "forensic_only", "semantic", "forensic"],
        default=None,
        help="Detector architecture variant (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for model initialization and training (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for AdamW (overrides config).",
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
        required=True,
        help="Directory to save checkpoints, history, and config.",
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

    # 1. Resolve experiment mode and manifest paths
    manifest_base = Path(args.manifest_dir)
    train_manifest_path: Path | None = None
    val_manifest_path: Path | None = None

    if args.experiment == "single":
        train_manifest_path = manifest_base / "single" / "train.jsonl"
        val_manifest_path = manifest_base / "single" / "val.jsonl"
    elif args.experiment == "logo":
        if not args.leave_out:
            raise ValueError("--leave-out is required when --experiment is 'logo'")
        from forensight.data.tiny_genimage import KAGGLE_TINY_GENIMAGE_GENERATORS
        if args.leave_out not in KAGGLE_TINY_GENIMAGE_GENERATORS:
            raise ValueError(
                f"Invalid --leave-out '{args.leave_out}'. Must be one of: {list(KAGGLE_TINY_GENIMAGE_GENERATORS)}"
            )
        train_manifest_path = manifest_base / "logo" / f"leave_{args.leave_out}" / "train.jsonl"
        val_manifest_path = manifest_base / "logo" / f"leave_{args.leave_out}" / "val.jsonl"
    elif args.experiment == "all7":
        train_manifest_path = manifest_base / "all7" / "train.jsonl"
        val_manifest_path = manifest_base / "all7" / "val.jsonl"

    if args.train_manifest:
        train_manifest_path = Path(args.train_manifest)
    if args.val_manifest:
        val_manifest_path = Path(args.val_manifest)

    if not train_manifest_path or not val_manifest_path:
        raise ValueError(
            "Missing manifest paths. Provide either --experiment {single,logo,all7} or both --train-manifest and --val-manifest."
        )

    # 2. Resolve configuration
    raw_variant = args.variant or "fusion"
    norm_variant = "semantic" if raw_variant in ("semantic", "semantic_only") else (
        "forensic" if raw_variant in ("forensic", "forensic_only") else "fusion"
    )

    if args.config:
        config = load_r2_config(args.config)
    else:
        config = load_r2_config(Path(f"configs/r2/{norm_variant}.json"))

    if args.variant and config["variant"] != norm_variant:
        raise ValueError(
            f"--variant {norm_variant} conflicts with config variant {config['variant']}"
        )

    # Apply CLI overrides to config
    if args.variant:
        config["variant"] = norm_variant
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.experiment is not None:
        config["experiment_mode"] = args.experiment
    if args.leave_out is not None:
        config["leave_out"] = args.leave_out

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["seed"])
    device = torch.device(args.device) if args.device else select_device()
    print(f"Using compute device: {device}")

    print(f"Loading training manifest: {train_manifest_path}")
    train_manifest = load_manifest_file(train_manifest_path)
    print(f"Loading validation manifest: {val_manifest_path}")
    val_manifest = load_manifest_file(val_manifest_path)

    # 2. Pre-flight leakage audit on train + val manifests (strictly before training)
    print("Performing pre-flight leakage audit on train and validation manifests...")
    from forensight.data.audit import audit_manifest_leakage
    leak_check = audit_manifest_leakage(
        train_manifest, [val_manifest], base_dir=args.base_dir, check_content_hashes=True
    )
    audit_data: dict[str, Any] = {
        "train_samples": len(train_manifest),
        "val_samples": len(val_manifest),
        "train_reals": len(train_manifest.filter(label=0)),
        "train_fakes": len(train_manifest.filter(label=1)),
        "val_reals": len(val_manifest.filter(label=0)),
        "val_fakes": len(val_manifest.filter(label=1)),
        "train_generators": sorted(list({r.generator for r in train_manifest if r.generator})),
        "val_generators": sorted(list({r.generator for r in val_manifest if r.generator})),
        "leakage": leak_check,
    }
    with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # Persist sealed manifests immediately
    train_manifest.to_jsonl(output_dir / "train_manifest.jsonl")
    val_manifest.to_jsonl(output_dir / "val_manifest.jsonl")

    if leak_check.get("has_leakage"):
        violations_str = "\n - ".join(leak_check.get("violations", []))
        raise RuntimeError(
            f"Pre-flight leakage audit FAILED! Critical leakage detected before training:\n - {violations_str}"
        )
    print("Pre-flight leakage audit PASSED (0 sample ID / path / SHA256 collisions).")

    variant = config["variant"]
    print(f"Building R2 variant: {variant}")
    model, clip_transform, forensic_transform = build_r2_model(config)
    model.to(device)

    # Training and evaluation intentionally share the same branch transforms so that the
    # forensic stream sees the identical source window in both phases.
    train_dataset = R2ImageDataset(
        train_manifest,
        base_dir=args.base_dir,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
    )

    val_dataset = R2ImageDataset(
        val_manifest,
        base_dir=args.base_dir,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )

    trainable_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["epochs"]
    best_val_auroc = -1.0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    print(f"Starting training: {epochs} epochs, {len(trainable_parameters)} parameter tensors trainable")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            variant=variant,
        )
        val_loss, val_auroc = validate_one_epoch(
            model,
            val_loader,
            device=device,
            variant=variant,
            return_metrics=True,
        )
        auroc_str = f"{val_auroc:.4f}" if val_auroc is not None else "N/A"
        print(f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUROC: {auroc_str}")

        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_auroc": round(float(val_auroc), 4) if val_auroc is not None else None,
        })

        is_best = False
        if val_auroc is not None:
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_loss = min(best_val_loss, val_loss)
                is_best = True
                metric_reason = f"Val AUROC: {best_val_auroc:.4f}"
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
            metric_reason = f"Val Loss: {best_val_loss:.4f}"

        if is_best:
            ckpt_path = output_dir / "checkpoint.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, config=config)
            print(f"  -> New best validation metric ({metric_reason})! Saved checkpoint to {ckpt_path}")

    # 1. Persist train_history.json
    with (output_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 2. Persist config.json
    config_dest = output_dir / "config.json"
    with config_dest.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_summary = f"Best Val AUROC: {best_val_auroc:.4f}" if best_val_auroc >= 0.0 else f"Best Val Loss: {best_val_loss:.4f}"
    print(f"Training completed. {best_summary}. Artifacts saved in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
