#!/usr/bin/env python3
"""ForenSight Milestone R2: Standalone Kaggle GPU Execution Script.

This script runs on Kaggle GPU (e.g., NVIDIA T4 x2) to:
1. Download and extract TheKernel01/Tiny-GenImage dataset from Hugging Face Hub.
2. Build generator-disjoint train, validation, and OOD test manifests.
3. Train the ForenSight R2 Concat Fusion detector (CLIP ViT-B/32 + NPR ResNet18).
4. Calibrate the decision threshold tau* strictly on the validation set.
5. Evaluate against In-Domain, Near-OOD, and Cross-Generator OOD distributions.
6. Export best_model.pt, eval_report.json, and predictions.jsonl to /kaggle/working/.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

# Ensure required packages are present on Kaggle environment
for pkg in ["open_clip_torch", "pyarrow", "huggingface_hub"]:
    try:
        __import__(pkg.split("_")[0])
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

import open_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForenSight-Kaggle-R2")


# ==============================================================================
# 1. MODEL ARCHITECTURE (R2 CONCAT FUSION)
# ==============================================================================

class NPRTransform(nn.Module):
    """Deterministic NPR-inspired residual transform.
    residual = image - upsample(downsample(image, s=0.5))
    """

    def __init__(self, scale_factor: float = 0.5, mode: str = "bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        reduced = F.interpolate(
            image,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
            recompute_scale_factor=False,
        )
        reconstructed = F.interpolate(
            reduced,
            size=image.shape[-2:],
            mode=self.mode,
            align_corners=False,
        )
        return image - reconstructed


class ForensicEncoder(nn.Module):
    """Forensic branch: NPR transform -> ResNet18 -> 256-d projection."""

    def __init__(self, projection_dim: int = 256):
        super().__init__()
        self.npr = NPRTransform(scale_factor=0.5, mode="bilinear")
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(in_features, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residuals = self.npr(image)
        feats = self.backbone(residuals)
        return self.norm(self.projection(feats))


class SemanticEncoder(nn.Module):
    """Semantic branch: Frozen CLIP ViT-B/32 -> 256-d projection."""

    def __init__(self, projection_dim: int = 256):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.backbone = clip_model.visual
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.projection = nn.Linear(512, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(image)
        return self.norm(self.projection(feats.float()))


class FusionDetector(nn.Module):
    """Concat Fusion detector: [Semantic (256), Forensic (256)] -> MLP -> Logit."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.semantic_encoder = SemanticEncoder(projection_dim=256)
        self.forensic_encoder = ForensicEncoder(projection_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_img: torch.Tensor, forensic_img: torch.Tensor) -> torch.Tensor:
        sem = self.semantic_encoder(clip_img)
        foren = self.forensic_encoder(forensic_img)
        fused = torch.cat([sem, foren], dim=-1)
        return self.classifier(fused)


class SemanticOnlyDetector(nn.Module):
    """Semantic-only binary detector: frozen CLIP -> 256 -> MLP -> logit."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = SemanticEncoder(projection_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_img: torch.Tensor, forensic_img: torch.Tensor = None) -> torch.Tensor:
        feat = self.encoder(clip_img)
        return self.classifier(feat)


class ForensicOnlyDetector(nn.Module):
    """Forensic-only binary detector: NPR -> ResNet18 -> 256 -> MLP -> logit."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = ForensicEncoder(projection_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_img: torch.Tensor = None, forensic_img: torch.Tensor = None) -> torch.Tensor:
        feat = self.encoder(forensic_img)
        return self.classifier(feat)


# ==============================================================================

# 2. DATASET DOWNLOAD & ZERO LEAKAGE PARTITIONING
# ==============================================================================

TINY_GENIMAGE_GEN_MAP: dict[int, str] = {
    0: "nature",
    1: "adm",
    2: "biggan",
    3: "glide",
    4: "midjourney",
    5: "sd14",
    6: "sd15",
    7: "vqdm",
    8: "wukong",
}

CROSS_OOD_GEN_IDS = ["midjourney", "adm", "glide", "wukong", "vqdm", "biggan"]


def extract_raw_bytes(image_val: Any) -> bytes:
    if hasattr(image_val, "as_py"):
        image_val = image_val.as_py()
    if isinstance(image_val, dict):
        raw = image_val.get("bytes")
        if raw is not None:
            return bytes(raw)
        path = image_val.get("path")
        if path and Path(path).exists():
            return Path(path).read_bytes()
    if isinstance(image_val, (bytes, bytearray)):
        return bytes(image_val)
    raise ValueError(f"Unable to extract image bytes: {type(image_val)}")


def detect_ext(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return ".png"


def prepare_tiny_genimage(
    repo_id: str = "TheKernel01/Tiny-GenImage",
    target_root: str = "/tmp/tiny_genimage",
) -> dict[str, list[dict]]:
    """Download Parquet shards and extract images to disk with zero leakage."""
    target_path = Path(target_root)
    target_path.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    train_parquets = sorted([f for f in repo_files if "train-" in f and f.endswith(".parquet")])
    val_parquets = sorted([f for f in repo_files if "validation-" in f and f.endswith(".parquet")])

    logger.info("Found %d train parquets and %d val parquets on Hugging Face.", len(train_parquets), len(val_parquets))

    records: list[dict] = []
    counter = 0

    # Process all parquets
    for split_name, file_list in [("train", train_parquets), ("validation", val_parquets)]:
        logger.info("Downloading and extracting %d %s parquets...", len(file_list), split_name)
        for rel_file in file_list:
            t0 = time.time()
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=rel_file,
                repo_type="dataset",
                local_dir="/tmp/hf_cache",
            )
            # Read Parquet
            table = pq.read_table(downloaded)
            img_col = table.column("image")
            lbl_col = table.column("label")
            gen_col = table.column("generator")

            for i in range(len(table)):
                raw_bytes = extract_raw_bytes(img_col[i])
                label = int(lbl_col[i].as_py())
                gen_int = int(gen_col[i].as_py())
                generator = TINY_GENIMAGE_GEN_MAP.get(gen_int, f"gen_{gen_int}")

                ext = detect_ext(raw_bytes)
                try:
                    with Image.open(io.BytesIO(raw_bytes)) as test_img:
                        test_img.verify()
                except Exception as e:
                    logger.warning("Corrupt image encountered and skipped: gen=%s idx=%d (%s)", generator, i, e)
                    continue

                out_dir = target_path / generator / split_name
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{generator}_{split_name}_{counter:06d}{ext}"
                file_path = out_dir / filename

                if not file_path.exists():
                    file_path.write_bytes(raw_bytes)

                records.append({
                    "image_path": str(file_path),
                    "label": label,
                    "generator": generator,
                    "raw_split": split_name,
                    "sample_id": f"tiny_genimage_{generator}_{split_name}_{counter:06d}",
                })
                counter += 1

            # Delete downloaded parquet to conserve disk space
            try:
                os.remove(downloaded)
            except Exception:
                pass

            logger.info("  Extracted %s (%d images in %.1fs)", Path(rel_file).name, len(table), time.time() - t0)

    logger.info("Total images extracted: %d", len(records))
    return records


def build_logo_splits(
    records: list[dict],
    leave_out_gen: str = "midjourney",
    n_val_per_gen: int = 100,
) -> dict[str, list[dict]]:
    """Build Leave-One-Generator-Out (LOGO) splits.
    - Train: All train fakes from seen generators (6 gens x 2,000 = 12,000 fakes + 12,000 reals = 24,000 images).
    - Val: n_val_per_gen fakes from seen generators (6 x 100 = 600 fakes + 600 reals = 1,200 images).
    - Test Held-Out (leave_out_gen): All available fakes (2,500 fakes + 2,500 reals = 5,000 images).
    - Test In-Domain Seen: Remaining validation fakes from seen generators (2,400 fakes + 2,400 reals = 4,800 images).
    """
    all_fake_gens = sorted(list({r["generator"] for r in records if r["label"] == 1}))
    seen_gens = [g for g in all_fake_gens if g != leave_out_gen]

    train_reals = [r for r in records if r["raw_split"] == "train" and r["label"] == 0]
    val_reals = [r for r in records if r["raw_split"] == "validation" and r["label"] == 0]

    # 1. Train split
    train_fakes = [r for r in records if r["raw_split"] == "train" and r["generator"] in seen_gens and r["label"] == 1]
    n_train = min(len(train_fakes), len(train_reals))
    random.seed(42)
    selected_train_fakes = train_fakes[:n_train]
    selected_train_reals = train_reals[:n_train]
    train_set = selected_train_fakes + selected_train_reals
    random.shuffle(train_set)

    # 2. Allocate validation reals
    real_offset = 0

    def allocate_val_reals(n: int) -> list[dict]:
        nonlocal real_offset
        alloc = val_reals[real_offset : real_offset + n]
        real_offset += len(alloc)
        return alloc

    val_set: list[dict] = []
    test_seen_set: list[dict] = []

    for gen in seen_gens:
        gen_val_fakes = [r for r in records if r["raw_split"] == "validation" and r["generator"] == gen and r["label"] == 1]
        n_val = min(n_val_per_gen, len(gen_val_fakes) // 2) if len(gen_val_fakes) >= 2 else len(gen_val_fakes)
        cur_val_fakes = gen_val_fakes[:n_val]
        cur_test_fakes = gen_val_fakes[n_val:]

        val_set.extend(cur_val_fakes)
        val_set.extend(allocate_val_reals(len(cur_val_fakes)))

        test_seen_set.extend(cur_test_fakes)
        test_seen_set.extend(allocate_val_reals(len(cur_test_fakes)))

    # 3. Held-out test set
    held_out_fakes = [r for r in records if r["generator"] == leave_out_gen and r["label"] == 1]
    n_held_out = len(held_out_fakes)
    remaining_val_reals = len(val_reals) - real_offset
    held_out_reals: list[dict] = []

    if remaining_val_reals >= n_held_out:
        held_out_reals = allocate_val_reals(n_held_out)
    else:
        held_out_reals.extend(allocate_val_reals(remaining_val_reals))
        needed_from_train = n_held_out - len(held_out_reals)
        unused_train_reals = train_reals[n_train : n_train + needed_from_train]
        held_out_reals.extend(unused_train_reals)

    if len(held_out_reals) < len(held_out_fakes):
        held_out_fakes = held_out_fakes[:len(held_out_reals)]

    test_held_out = held_out_fakes + held_out_reals

    splits = {
        "train": train_set,
        "val": val_set,
        f"test_{leave_out_gen}": test_held_out,
        "test_in_domain_seen": test_seen_set,
    }

    logger.info("LOGO Splits created (held-out: %s):", leave_out_gen)
    for k, v in splits.items():
        reals = sum(1 for r in v if r["label"] == 0)
        fakes = sum(1 for r in v if r["label"] == 1)
        logger.info("  %s: %d total (%d real, %d fake)", k, len(v), reals, fakes)

    return splits


def build_all_in_one_splits(
    records: list[dict],
    n_val_per_gen: int = 100,
) -> dict[str, list[dict]]:
    """Build All-In-One multi-generator splits (Upper Bound).
    - Train: All train fakes from all 7 generators (7 x 2,000 = 14,000 fakes + 14,000 reals = 28,000 images).
    - Val: 100 fakes from each generator (7 x 100 = 700 fakes + 700 reals = 1,400 images).
    - Per-gen Test: Remaining val fakes (400 fakes + 400 reals = 800 images per generator).
    - Test All Combined: All 7 test sets combined (5,600 images).
    """
    all_fake_gens = sorted(list({r["generator"] for r in records if r["label"] == 1}))

    train_reals = [r for r in records if r["raw_split"] == "train" and r["label"] == 0]
    val_reals = [r for r in records if r["raw_split"] == "validation" and r["label"] == 0]

    train_fakes = [r for r in records if r["raw_split"] == "train" and r["label"] == 1]
    n_train = min(len(train_fakes), len(train_reals))
    random.seed(42)
    selected_train_fakes = train_fakes[:n_train]
    selected_train_reals = train_reals[:n_train]
    train_set = selected_train_fakes + selected_train_reals
    random.shuffle(train_set)

    real_offset = 0

    def allocate_val_reals(n: int) -> list[dict]:
        nonlocal real_offset
        alloc = val_reals[real_offset : real_offset + n]
        real_offset += len(alloc)
        return alloc

    val_set: list[dict] = []
    test_combined: list[dict] = []
    per_gen_test: dict[str, list[dict]] = {}

    for gen in all_fake_gens:
        gen_val_fakes = [r for r in records if r["raw_split"] == "validation" and r["generator"] == gen and r["label"] == 1]
        n_val = min(n_val_per_gen, len(gen_val_fakes) // 2) if len(gen_val_fakes) >= 2 else len(gen_val_fakes)
        cur_val_fakes = gen_val_fakes[:n_val]
        cur_test_fakes = gen_val_fakes[n_val:]

        val_set.extend(cur_val_fakes)
        val_set.extend(allocate_val_reals(len(cur_val_fakes)))

        cur_test = cur_test_fakes + allocate_val_reals(len(cur_test_fakes))
        per_gen_test[f"test_{gen}"] = cur_test
        test_combined.extend(cur_test)

    splits = {
        "train": train_set,
        "val": val_set,
        "test_all_combined": test_combined,
    }
    splits.update(per_gen_test)

    logger.info("All-In-One Splits created:")
    for k, v in splits.items():
        reals = sum(1 for r in v if r["label"] == 0)
        fakes = sum(1 for r in v if r["label"] == 1)
        logger.info("  %s: %d total (%d real, %d fake)", k, len(v), reals, fakes)

    return splits


class FastImageDataset(Dataset):
    """PyTorch Dataset applying CLIP transform and Forensic transform to raw images."""

    def __init__(self, records: list[dict], clip_transform: Any, forensic_transform: Any):
        self.records = records
        self.clip_transform = clip_transform
        self.forensic_transform = forensic_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float, str, str]:
        rec = self.records[idx]
        img_path = rec["image_path"]
        try:
            with Image.open(img_path) as opened:
                rgb = opened.convert("RGB")
                clip_tensor = self.clip_transform(rgb)
                foren_tensor = self.forensic_transform(rgb)
                return clip_tensor, foren_tensor, float(rec["label"]), rec["generator"], img_path
        except Exception as e:
            raise RuntimeError(f"Corrupt or unreadable image at {img_path}: {e}") from e


# ==============================================================================
# 3. EVALUATION & THRESHOLD CALIBRATION
# ==============================================================================

def select_threshold(y_true: np.ndarray, y_scores: np.ndarray, strategy: str = "f1") -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_metric = -1.0
    for thresh in candidates:
        preds = (y_scores >= thresh).astype(int)
        metric = f1_score(y_true, preds, zero_division=0) if strategy == "f1" else accuracy_score(y_true, preds)
        if metric > best_metric:
            best_metric = metric
            best_thresh = float(thresh)
    return best_thresh


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, Any]:
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_gens: list[str] = []

    with torch.no_grad():
        for clip_imgs, foren_imgs, labels, gens, _ in loader:
            clip_imgs = clip_imgs.to(device)
            foren_imgs = foren_imgs.to(device)
            logits = model(clip_imgs, foren_imgs).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            all_scores.extend(probs)
            all_labels.extend(int(l) for l in labels.numpy())
            all_gens.extend(gens)

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = (y_scores >= threshold).astype(int)

    auroc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else None
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auroc": auroc,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "threshold": threshold,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "total_samples": len(y_true),
        "scores": all_scores,
        "labels": all_labels,
        "generators": all_gens,
    }


def train_and_eval_experiment(
    experiment_name: str,
    model: nn.Module,
    splits: dict[str, list[dict]],
    clip_transform: Any,
    forensic_transform: Any,
    device: torch.device,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, Any]:
    logger.info("\n" + "=" * 80)
    logger.info("STARTING EXPERIMENT: %s (Train size: %d)", experiment_name.upper(), len(splits["train"]))
    logger.info("=" * 80)

    train_ds = FastImageDataset(splits["train"], clip_transform, forensic_transform)
    val_ds = FastImageDataset(splits["val"], clip_transform, forensic_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_path = output_dir / f"{experiment_name}_best.pt"

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for clip_imgs, foren_imgs, labels, _, _ in train_loader:
            clip_imgs = clip_imgs.to(device)
            foren_imgs = foren_imgs.to(device)
            targets = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(clip_imgs, foren_imgs).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        val_scores: list[float] = []
        val_targets: list[int] = []

        with torch.no_grad():
            for clip_imgs, foren_imgs, labels, _, _ in val_loader:
                clip_imgs = clip_imgs.to(device)
                foren_imgs = foren_imgs.to(device)
                targets = labels.float().to(device)

                logits = model(clip_imgs, foren_imgs).squeeze(-1)
                loss = criterion(logits, targets)
                val_loss_total += loss.item()
                val_batches += 1

                val_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                val_targets.extend(int(l) for l in labels.numpy())

        val_loss = val_loss_total / max(1, val_batches)
        val_auroc = roc_auc_score(val_targets, val_scores) if len(set(val_targets)) > 1 else 0.5
        elapsed = time.time() - start_time

        logger.info(
            "[%s] Epoch %d/%d [%.1fs] | Train Loss: %.4f | Val Loss: %.4f | Val AUROC: %.4f",
            experiment_name, epoch, epochs, elapsed, train_loss, val_loss, val_auroc
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  --> Saved new best checkpoint to %s", checkpoint_path)

    # Threshold calibration on validation set
    logger.info("[%s] Loading best checkpoint for threshold calibration...", experiment_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    val_metrics = evaluate_split(model, val_loader, device, threshold=0.5)
    calibrated_tau = select_threshold(np.array(val_metrics["labels"]), np.array(val_metrics["scores"]), strategy="f1")
    logger.info("[%s] Calibrated validation threshold: tau* = %.4f (frozen for all test sets)", experiment_name, calibrated_tau)

    # Evaluate across all test splits
    test_splits = [k for k in splits.keys() if k.startswith("test_")]
    results = {}
    for split_name in test_splits:
        test_ds = FastImageDataset(splits[split_name], clip_transform, forensic_transform)
        if len(test_ds) == 0:
            continue
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        res = evaluate_split(model, test_loader, device, threshold=calibrated_tau)
        results[split_name] = {
            "auroc": res["auroc"],
            "accuracy": res["accuracy"],
            "f1": res["f1"],
            "precision": res["precision"],
            "recall": res["recall"],
            "confusion_matrix": res["confusion_matrix"],
            "total_samples": res["total_samples"],
        }
        logger.info("  %-25s | AUROC: %.4f | Acc: %.4f | F1: %.4f (N=%d)", split_name, res["auroc"] or 0.0, res["accuracy"], res["f1"], res["total_samples"])

    reproducibility = {
        "run_id": f"kaggle_r2_{experiment_name}",
        "experiment_name": experiment_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": None,
        "seed": 42,
        "split_version": "tiny_genimage_v1",
        "config": {
            "architecture": "FusionDetector (CLIP ViT-B/32 + ResNet18 NPR)",
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset": "TheKernel01/Tiny-GenImage",
        },
        "threshold_source": "val_optimal_f1",
        "threshold_value": float(calibrated_tau),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "protocol_notes": (
            "Kaggle fast exploration protocol: Tiny-GenImage dataset, ViT-B/32 semantic branch, "
            "ResNet-18 forensic branch. Calibrated on seen generator validation split."
        ),
    }

    return {
        "experiment": experiment_name,
        "calibrated_threshold": calibrated_tau,
        "best_val_loss": best_val_loss,
        "train_samples": len(splits["train"]),
        "results": results,
        "reproducibility": reproducibility,
    }


# ==============================================================================
# 4. MAIN ORCHESTRATION FUNCTION
# ==============================================================================

def main():
    print("=" * 80)
    print("      FORENSIGHT: R2 MULTI-GENERATOR & LEAVE-ONE-OUT GPU RUNNER     ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = True

    output_dir = Path("/kaggle/working")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = 5
    batch_size = 64
    num_workers = 4 if torch.cuda.is_available() else 0

    # 1. Download & Extract all 35,000 images once
    all_records = prepare_tiny_genimage(repo_id="TheKernel01/Tiny-GenImage", target_root="/tmp/tiny_genimage")

    # 2. Build Transforms
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    forensic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # --------------------------------------------------------------------------
    # EXPERIMENT 1: LEAVE-ONE-GENERATOR-OUT (LOGO: Midjourney Held-Out)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# EXPERIMENT 1: LEAVE-ONE-OUT (Train on 6 gens, Test on Held-Out Midjourney)")
    print("#" * 80)

    logo_splits = build_logo_splits(all_records, leave_out_gen="midjourney", n_val_per_gen=100)
    logo_model = FusionDetector(hidden_dim=128, dropout=0.2)
    logo_results = train_and_eval_experiment(
        experiment_name="logo_midjourney",
        model=logo_model,
        splits=logo_splits,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
        device=device,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logo_report_path = output_dir / "logo_report.json"
    with open(logo_report_path, "w", encoding="utf-8") as f:
        json.dump(logo_results, f, indent=2)
    logger.info("Saved LOGO report to %s", logo_report_path)

    # Clean memory between runs
    del logo_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # EXPERIMENT 2: ALL-IN-ONE MULTI-GENERATOR (Upper Bound on All 7 Generators)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# EXPERIMENT 2: ALL-IN-ONE (Train on All 7 Generators, Test on All 7)")
    print("#" * 80)

    all_splits = build_all_in_one_splits(all_records, n_val_per_gen=100)
    aio_model = FusionDetector(hidden_dim=128, dropout=0.2)
    aio_results = train_and_eval_experiment(
        experiment_name="all_in_one",
        model=aio_model,
        splits=all_splits,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
        device=device,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    aio_report_path = output_dir / "all_in_one_report.json"
    with open(aio_report_path, "w", encoding="utf-8") as f:
        json.dump(aio_results, f, indent=2)
    logger.info("Saved All-In-One report to %s", aio_report_path)

    # --------------------------------------------------------------------------
    # BACKWARDS COMPATIBILITY ARTIFACTS
    # --------------------------------------------------------------------------
    aio_ckpt = output_dir / "all_in_one_best.pt"
    if aio_ckpt.exists():
        import shutil
        shutil.copyfile(aio_ckpt, output_dir / "best_model.pt")

    eval_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "logo_results": logo_results,
        "all_in_one_results": aio_results,
    }
    with open(output_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # --------------------------------------------------------------------------
    # PRINT FINAL COMPARATIVE BENCHMARK SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("                     FINAL MULTI-GENERATOR BENCHMARK EVALUATION                     ")
    print("=" * 96)
    print("--- EXPERIMENT 1: LEAVE-ONE-OUT (LOGO: Midjourney Held-Out) ---")
    for s, m in logo_results["results"].items():
        print(f"  {s:<28} | AUROC: {m['auroc']:.4f} | Accuracy: {m['accuracy']:.4f} | F1: {m['f1']:.4f} (N={m['total_samples']})")

    print("\n--- EXPERIMENT 2: ALL-IN-ONE (Upper Bound on All 7 Generators) ---")
    for s, m in aio_results["results"].items():
        print(f"  {s:<28} | AUROC: {m['auroc']:.4f} | Accuracy: {m['accuracy']:.4f} | F1: {m['f1']:.4f} (N={m['total_samples']})")
    print("=" * 96)

    print("\nExecution complete! Checkpoints and reports are ready in /kaggle/working/.")


if __name__ == "__main__":
    main()


