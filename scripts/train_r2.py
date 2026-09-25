#!/usr/bin/env python3
"""Training entrypoint for ForenSight R2 detector variants.

Orchestrates training strictly on GenImage SD1.4 train and validation splits,
saving the best checkpoint exclusively according to validation loss.
Test or OOD manifests are strictly prohibited from this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from forensight.data.r2_dataset import R2ImageDataset, load_manifest_file
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
        description="Train ForenSight R2 detector variant on SD1.4 train/val manifests."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to R2 JSON experiment configuration.",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        required=True,
        help="Path to sealed training manifest (JSONL/CSV).",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        required=True,
        help="Path to sealed validation manifest (JSONL/CSV).",
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


def main() -> int:
    args = parse_args()
    config = load_r2_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["seed"])
    device = torch.device(args.device) if args.device else select_device()
    print(f"Using compute device: {device}")

    variant = config["variant"]
    print(f"Building R2 variant: {variant}")
    model, clip_transform, forensic_transform = build_r2_model(config)
    model.to(device)

    print(f"Loading training manifest: {args.train_manifest}")
    train_manifest = load_manifest_file(args.train_manifest)
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

    print(f"Loading validation manifest: {args.val_manifest}")
    val_manifest = load_manifest_file(args.val_manifest)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / "checkpoint.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, config=config)
            # Also keep best.pt as alias for compatibility
            shutil.copyfile(ckpt_path, output_dir / "best.pt")
            print(f"  -> New best validation loss! Saved checkpoint to {ckpt_path}")

    # 1. Persist train_history.json (and history.json alias)
    for hname in ("train_history.json", "history.json"):
        with (output_dir / hname).open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # 2. Persist config.json
    config_dest = output_dir / "config.json"
    with config_dest.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 3. Persist sealed manifests
    train_manifest.to_jsonl(output_dir / "train_manifest.jsonl")
    val_manifest.to_jsonl(output_dir / "val_manifest.jsonl")

    # 4. Generate dataset_audit.json for train + val
    from forensight.data.audit import audit_manifest_leakage
    audit_data: dict[str, Any] = {
        "train_samples": len(train_manifest),
        "val_samples": len(val_manifest),
        "train_reals": len(train_manifest.filter(label=0)),
        "train_fakes": len(train_manifest.filter(label=1)),
        "val_reals": len(val_manifest.filter(label=0)),
        "val_fakes": len(val_manifest.filter(label=1)),
        "train_generators": sorted(list({r.generator for r in train_manifest if r.generator})),
        "val_generators": sorted(list({r.generator for r in val_manifest if r.generator})),
    }
    leak_check = audit_manifest_leakage(train_manifest, [val_manifest])
    audit_data["leakage"] = leak_check
    with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}. Artifacts saved in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
