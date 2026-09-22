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
        val_loss = validate_one_epoch(
            model,
            val_loader,
            device=device,
            variant=variant,
        )
        print(f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best.pt"
            save_checkpoint(checkpoint_path, model, optimizer, epoch=epoch, config=config)
            print(f"  -> New best validation loss! Saved checkpoint to {checkpoint_path}")

    # Persist history.json and copy config.json
    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    config_dest = output_dir / "config.json"
    with config_dest.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}. Artifacts saved in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
