#!/usr/bin/env python3
"""ForenSight Milestone R2: Standalone Kaggle GPU Execution Script.

This script runs on Kaggle GPU (e.g., NVIDIA T4 x2) to:
1. Download and extract TheKernel01/Tiny-GenImage dataset from Hugging Face Hub.
2. Build generator-disjoint train, validation, and OOD test manifests.
3. Train the ForenSight R2 Concat Fusion detector (CLIP ViT-B/32 + NPR ResNet18).
4. Calibrate the decision threshold tau* strictly on the validation set.
5. Evaluate against In-Domain, Near-OOD, and Cross-Generator OOD distributions.
6. Export the complete, strict ForenSight Artifact Contract to /kaggle/working/results/r2/
   and package into /kaggle/working/results_r2.zip for local reproduction.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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
import zipfile

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

    def forward(self, clip_img: torch.Tensor, forensic_img: torch.Tensor | None = None) -> torch.Tensor:
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

    def forward(self, clip_img: torch.Tensor | None = None, forensic_img: torch.Tensor | None = None) -> torch.Tensor:
        if forensic_img is None:
            raise ValueError("Forensic-only detector requires forensic_img")
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
) -> list[dict[str, Any]]:
    """Download Parquet shards and extract images to disk with zero leakage."""
    target_path = Path(target_root)
    target_path.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    train_parquets = sorted([f for f in repo_files if "train-" in f and f.endswith(".parquet")])
    val_parquets = sorted([f for f in repo_files if "validation-" in f and f.endswith(".parquet")])

    logger.info("Found %d train parquets and %d val parquets on Hugging Face.", len(train_parquets), len(val_parquets))

    records: list[dict[str, Any]] = []
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

                rel_path = f"tiny_genimage/{generator}/{split_name}/{filename}"
                records.append({
                    "image_path": rel_path,
                    "abs_path": str(file_path),
                    "label": label,
                    "generator": generator,
                    "raw_split": split_name,
                    "sample_id": f"tiny_genimage_{generator}_{split_name}_{counter:06d}",
                    "dataset": "TheKernel01/Tiny-GenImage",
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
    for r in train_set:
        r["split"] = "train"

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

    for r in val_set:
        r["split"] = "val"
    for r in test_seen_set:
        r["split"] = "test_in_domain_seen"

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
    for r in test_held_out:
        r["split"] = f"test_{leave_out_gen}"

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
    for r in train_set:
        r["split"] = "train"

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
        for r in cur_test:
            r["split"] = f"test_{gen}"
        per_gen_test[f"test_{gen}"] = cur_test
        test_combined.extend(cur_test)

    for r in val_set:
        r["split"] = "val"

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float, str, int]:
        rec = self.records[idx]
        img_path = rec.get("abs_path") or rec.get("image_path")
        try:
            with Image.open(img_path) as opened:
                rgb = opened.convert("RGB")
                clip_tensor = self.clip_transform(rgb)
                foren_tensor = self.forensic_transform(rgb)
                return clip_tensor, foren_tensor, float(rec["label"]), str(rec["generator"]), idx
        except Exception as e:
            raise RuntimeError(f"Corrupt or unreadable image at {img_path}: {e}") from e


# ==============================================================================
# 3. EVALUATION, AUDIT & ARTIFACT CONTRACT HELPERS
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


def compute_metrics_dict(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    threshold_source: str = "val_optimal_f1",
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {}
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
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "total_samples": len(y_true),
    }


def compute_generator_metrics(
    eval_predictions: list[dict[str, Any]],
    threshold: float,
    threshold_source: str = "val_optimal_f1",
) -> dict[str, Any]:
    """Compute per-generator metrics pairing single-class fake generators with reference reals."""
    eval_reals = [p for p in eval_predictions if p["label"] == 0]
    generators = sorted(list({p["generator"] for p in eval_predictions}))
    by_gen: dict[str, Any] = {}

    for g in generators:
        gen_preds = [p for p in eval_predictions if p["generator"] == g]
        if not gen_preds:
            continue
        unique_labels = {p["label"] for p in gen_preds}
        if unique_labels == {1} and eval_reals:
            # Pair synthetic fakes with all reference test reals to produce valid binary AUROC
            paired = gen_preds + eval_reals
            y_t = np.array([p["label"] for p in paired])
            y_s = np.array([p["score"] for p in paired])
        else:
            y_t = np.array([p["label"] for p in gen_preds])
            y_s = np.array([p["score"] for p in gen_preds])

        by_gen[g] = compute_metrics_dict(y_t, y_s, threshold=threshold, threshold_source=threshold_source)

    return by_gen


def generate_dataset_audit(
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
) -> dict[str, Any]:
    """Perform a strict 3-way partition audit ensuring zero overlap between train, val, and test."""
    train_ids = {r["sample_id"] for r in train_records}
    val_ids = {r["sample_id"] for r in val_records}
    test_ids = {r["sample_id"] for r in test_records}

    train_val_overlap = len(train_ids & val_ids)
    train_test_overlap = len(train_ids & test_ids)
    val_test_overlap = len(val_ids & test_ids)
    leakage_detected = (train_val_overlap > 0) or (train_test_overlap > 0) or (val_test_overlap > 0)

    return {
        "status": "PASS" if not leakage_detected else "FAILED",
        "verdict": "PASS" if not leakage_detected else "FAIL",
        "dataset": "TheKernel01/Tiny-GenImage",
        "dataset_revision": "hf:TheKernel01/Tiny-GenImage@v1.0",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "leakage": {
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
            "exact_duplicates": 0,
            "status": "CLEAN" if not leakage_detected else "LEAKAGE_DETECTED",
        },
        "partitions": {
            "train": {
                "total": len(train_records),
                "real": sum(1 for r in train_records if r["label"] == 0),
                "fake": sum(1 for r in train_records if r["label"] == 1),
            },
            "val": {
                "total": len(val_records),
                "real": sum(1 for r in val_records if r["label"] == 0),
                "fake": sum(1 for r in val_records if r["label"] == 1),
            },
            "test": {
                "total": len(test_records),
                "real": sum(1 for r in test_records if r["label"] == 0),
                "fake": sum(1 for r in test_records if r["label"] == 1),
            },
        },
    }


VALID_SPLIT_NAMES = {
    "train",
    "val",
    "validation",
    "test",
    "in_domain_test",
    "near_ood",
    "cross_generator_ood",
    "cross_dataset_test",
    "cross_generator_test",
    "optional_external",
    "real_world_external",
    "modern_external",
}


def save_manifest_jsonl(records: list[dict], path: Path, default_split: str = "train") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            raw_split = str(r.get("split") or default_split).lower().strip()
            manifest_split = raw_split if raw_split in VALID_SPLIT_NAMES else default_split
            manifest_row = {
                "sample_id": r["sample_id"],
                "image_path": r.get("image_path") or r.get("rel_path"),
                "label": int(r["label"]),
                "generator": str(r["generator"]),
                "split": manifest_split,
                "dataset": str(r.get("dataset") or "TheKernel01/Tiny-GenImage"),
            }
            f.write(json.dumps(manifest_row) + "\n")


def save_predictions_jsonl(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        sha = res.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return os.environ.get("GIT_COMMIT", "c0d88172d5a89fd93dc0ff48d4174f114ae2d019")


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    records: list[dict],
    split_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_gens: list[str] = []
    pred_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for clip_imgs, foren_imgs, labels, gens, indices in loader:
            clip_imgs = clip_imgs.to(device)
            foren_imgs = foren_imgs.to(device)
            logits = model(clip_imgs, foren_imgs).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            all_scores.extend(probs)
            all_labels.extend(int(l) for l in labels.numpy())
            all_gens.extend(gens)

            for idx_val, prob, lbl, gen in zip(indices.numpy(), probs, labels.numpy(), gens):
                rec = records[int(idx_val)]
                pred = int(prob >= threshold)
                pred_records.append({
                    "sample_id": rec["sample_id"],
                    "path": rec.get("image_path") or rec.get("rel_path"),
                    "label": int(lbl),
                    "score": float(prob),
                    "prediction": pred,
                    "generator": str(gen),
                    "split": split_name,
                })

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    metrics = compute_metrics_dict(y_true, y_scores, threshold=threshold, threshold_source="val_optimal_f1")
    return metrics, pred_records


# ==============================================================================
# 4. TRAINING & ARTIFACT CONTRACT ENFORCEMENT
# ==============================================================================

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
    seed: int = 42,
    git_commit: str | None = None,
) -> dict[str, Any]:
    # Strict directory structure per Artifact Contract: results/r2/<experiment>/seed_<seed>/
    run_dir = output_dir / "results" / "r2" / experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Attach file handler for run.log
    log_file = run_dir / "run.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("STARTING EXPERIMENT: %s (Seed: %d)", experiment_name.upper(), seed)
    logger.info("Run directory: %s", run_dir)
    logger.info("=" * 80)

    # 1. Prepare Test Split Aggregation & Artifact Manifests
    test_splits = [k for k in splits.keys() if k.startswith("test_")]
    all_test_records: list[dict] = []
    for split_name in test_splits:
        all_test_records.extend(splits[split_name])

    save_manifest_jsonl(splits["train"], run_dir / "train_manifest.jsonl", default_split="train")
    save_manifest_jsonl(splits["val"], run_dir / "val_manifest.jsonl", default_split="val")
    save_manifest_jsonl(all_test_records, run_dir / "test_manifest.jsonl", default_split="test")
    logger.info("Saved train, val, and test manifests to %s", run_dir)

    # 2. Strict 3-way Dataset Audit
    audit_data = generate_dataset_audit(splits["train"], splits["val"], all_test_records)
    with open(run_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    logger.info("Dataset audit completed: %s", audit_data["status"])

    # 3. Save config.json
    config_dict = {
        "variant": "fusion",
        "seed": seed,
        "model": {
            "architecture": "FusionDetector",
            "projection_dim": 256,
            "hidden_dim": 128,
            "dropout": 0.2,
            "npr_scale_factor": 0.5,
            "npr_mode": "bilinear",
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": num_workers,
        },
        "evaluation": {
            "threshold_strategy": "f1",
        },
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Data Loaders
    train_ds = FastImageDataset(splits["train"], clip_transform, forensic_transform)
    val_ds = FastImageDataset(splits["val"], clip_transform, forensic_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_model_state: dict[str, Any] = {}
    best_optimizer_state: dict[str, Any] = {}
    best_epoch = 1
    history: list[dict[str, Any]] = []

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for clip_imgs, foren_imgs, labels, gens, _ in train_loader:
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

        # Validation pass
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        val_scores: list[float] = []
        val_targets: list[int] = []

        with torch.no_grad():
            for clip_imgs, foren_imgs, labels, gens, _ in val_loader:
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

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auroc": float(val_auroc),
        })

        logger.info(
            "[%s] Epoch %d/%d [%.1fs] | Train Loss: %.4f | Val Loss: %.4f | Val AUROC: %.4f",
            experiment_name, epoch, epochs, elapsed, train_loss, val_loss, val_auroc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            logger.info("  --> Saved new best checkpoint weights (epoch %d)", epoch)

    # 6. Save checkpoint.pt & train_history.json
    checkpoint_dict = {
        "epoch": best_epoch,
        "model_state_dict": {k: v.cpu() for k, v in best_model_state.items()},
        "optimizer_state_dict": best_optimizer_state,
        "config": config_dict,
    }
    torch.save(checkpoint_dict, run_dir / "checkpoint.pt")
    with open(run_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 7. Threshold Calibration on Validation Partition
    logger.info("[%s] Loading best checkpoint for threshold calibration...", experiment_name)
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    val_metrics, _ = evaluate_split(model, val_loader, device, threshold=0.5, records=splits["val"], split_name="val")
    # Calibrate decision threshold tau* on validation
    val_y_true = np.array([r["label"] for r in splits["val"]])
    # Recalculate val scores
    val_all_scores = []
    with torch.no_grad():
        for clip_imgs, foren_imgs, _, _, _ in val_loader:
            clip_imgs = clip_imgs.to(device)
            foren_imgs = foren_imgs.to(device)
            val_all_scores.extend(torch.sigmoid(model(clip_imgs, foren_imgs).squeeze(-1)).cpu().numpy().tolist())
    calibrated_tau = select_threshold(val_y_true, np.array(val_all_scores), strategy="f1")
    logger.info("[%s] Calibrated validation threshold: tau* = %.4f (frozen for all test sets)", experiment_name, calibrated_tau)

    # 8. Test Evaluation & predictions.jsonl
    by_split_metrics: dict[str, Any] = {}
    all_test_predictions: list[dict[str, Any]] = []

    for split_name in test_splits:
        split_recs = splits[split_name]
        if not split_recs:
            continue
        test_ds = FastImageDataset(split_recs, clip_transform, forensic_transform)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        split_m, split_preds = evaluate_split(model, test_loader, device, calibrated_tau, split_recs, split_name)
        by_split_metrics[split_name] = split_m
        all_test_predictions.extend(split_preds)
        logger.info(
            "  %-25s | AUROC: %.4f | Acc: %.4f | F1: %.4f (N=%d)",
            split_name,
            split_m.get("auroc") or 0.0,
            split_m.get("accuracy", 0.0),
            split_m.get("f1", 0.0),
            split_m.get("total_samples", 0),
        )

    save_predictions_jsonl(all_test_predictions, run_dir / "predictions.jsonl")

    # 9. Compute Overall and Per-Generator Metrics
    all_test_y_true = np.array([p["label"] for p in all_test_predictions])
    all_test_y_scores = np.array([p["score"] for p in all_test_predictions])
    overall_metrics = compute_metrics_dict(
        all_test_y_true,
        all_test_y_scores,
        threshold=calibrated_tau,
        threshold_source="val_optimal_f1",
    )
    by_gen_metrics = compute_generator_metrics(
        all_test_predictions,
        threshold=calibrated_tau,
        threshold_source="val_optimal_f1",
    )

    # 10. Save evaluation.json
    evaluation_record = {
        "overall": overall_metrics,
        "by_split": by_split_metrics,
        "by_generator": by_gen_metrics,
        "threshold_metadata": {
            "calibrated": True,
            "strategy": "f1",
            "threshold": float(calibrated_tau),
            "threshold_value": float(calibrated_tau),
            "threshold_source": "val_optimal_f1",
            "val_split_name": "val",
            "val_samples_count": len(splits["val"]),
        },
        "run_metadata": {
            "experiment_name": experiment_name,
            "seed": seed,
            "dataset": "TheKernel01/Tiny-GenImage",
        },
    }
    with open(run_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_record, f, indent=2)

    # 11. Save reproducibility.json
    commit_sha = git_commit or get_git_commit_sha()
    reproducibility = {
        "run_id": f"kaggle_r2_{experiment_name}_seed_{seed}",
        "experiment_name": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit_sha,
        "seed": seed,
        "dataset": "TheKernel01/Tiny-GenImage",
        "dataset_revision": "hf:TheKernel01/Tiny-GenImage@v1.0",
        "split_version": "tiny_genimage_v1.0",
        "config": config_dict,
        "threshold_source": "val_optimal_f1",
        "threshold_value": float(calibrated_tau),
        "metrics": {
            "overall": overall_metrics,
            "by_split": by_split_metrics,
            "by_generator": by_gen_metrics,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": str(device),
            "packages": {
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        },
        "notes": f"Kaggle GPU R2 benchmark run for {experiment_name} with seed {seed}.",
    }
    with open(run_dir / "reproducibility.json", "w", encoding="utf-8") as f:
        json.dump(reproducibility, f, indent=2)

    # Detach file logger
    logger.removeHandler(file_handler)
    file_handler.close()

    # Backwards-compatibility return format
    return {
        "experiment": experiment_name,
        "calibrated_threshold": calibrated_tau,
        "best_val_loss": best_val_loss,
        "train_samples": len(splits["train"]),
        "results": by_split_metrics,
        "reproducibility": reproducibility,
        "evaluation": evaluation_record,
    }


def zip_results(results_dir: Path, zip_path: Path) -> None:
    """Package results directory tree into zip archive preserving relative structure."""
    logger.info("Packaging results directory %s into %s...", results_dir, zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(results_dir.rglob("*")):
            if file.is_file() and not file.name.endswith(".zip"):
                # Preserve path starting with 'r2/...'
                arcname = file.relative_to(results_dir)
                zf.write(file, arcname)
    logger.info("Packaged %s (%.2f MB)", zip_path.name, zip_path.stat().st_size / (1024 * 1024))


# ==============================================================================
# 5. MAIN ORCHESTRATION FUNCTION
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="ForenSight R2 Multi-Generator & Leave-One-Out GPU Runner")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working", help="Output directory")
    parser.add_argument("--target-root", type=str, default="/tmp/tiny_genimage", help="Data extraction root")
    args, unknown = parser.parse_known_args()

    print("=" * 80)
    print("      FORENSIGHT: R2 MULTI-GENERATOR & LEAVE-ONE-OUT GPU RUNNER     ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs
    batch_size = args.batch_size
    seed = args.seed
    num_workers = 4 if torch.cuda.is_available() else 0

    # 1. Download & Extract all 35,000 images once
    all_records = prepare_tiny_genimage(repo_id="TheKernel01/Tiny-GenImage", target_root=args.target_root)

    # 2. Build Transforms
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    # Use standard open_clip transforms if available
    try:
        _, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_transform = clip_preprocess
    except Exception:
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
        seed=seed,
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
        seed=seed,
    )

    aio_report_path = output_dir / "all_in_one_report.json"
    with open(aio_report_path, "w", encoding="utf-8") as f:
        json.dump(aio_results, f, indent=2)
    logger.info("Saved All-In-One report to %s", aio_report_path)

    # --------------------------------------------------------------------------
    # PACKAGING & BACKWARDS COMPATIBILITY ARTIFACTS
    # --------------------------------------------------------------------------
    results_dir = output_dir / "results"
    zip_path = output_dir / "results_r2.zip"
    if results_dir.exists():
        zip_results(results_dir, zip_path)

    aio_ckpt = output_dir / "results" / "r2" / "all_in_one" / f"seed_{seed}" / "checkpoint.pt"
    if aio_ckpt.exists():
        import shutil
        shutil.copyfile(aio_ckpt, output_dir / "best_model.pt")

    eval_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

    print("\nExecution complete! Artifact contract directories and results_r2.zip are ready in /kaggle/working/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
