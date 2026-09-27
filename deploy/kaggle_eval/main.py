"""Evaluation script for ForenSight Milestone R2: LOGO Model at Threshold 0.50 vs 0.73.

WARNING -- NON-CANONICAL, DIAGNOSTIC ONLY.
    This script is a self-contained fork of the R2 pipeline (see the warning in
    `deploy/kaggle/main.py`) and it reuses its own model, preprocessing, and split logic.
    Threshold sweeps here are sensitivity diagnostics only: a threshold must never be
    chosen from test-set behaviour. The canonical pipeline calibrates tau* on the
    validation partition alone (`src/forensight/evaluation/runner.py`).

Evaluates pre-trained LOGO model (or trains if checkpoint not found in kernel inputs)
across multiple decision thresholds (0.50, 0.73, etc.) on held-out Midjourney and Seen In-domain.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

# Ensure third-party packages are installed
for pkg in ["open_clip_torch", "huggingface_hub", "pyarrow"]:
    try:
        __import__(pkg.replace("_torch", ""))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_threshold")


# ==============================================================================
# 1. MODEL ARCHITECTURE DEFINITIONS
# ==============================================================================

class NPRTransform(nn.Module):
    def __init__(self, scale: float = 0.5):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        down = nn.functional.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        up = nn.functional.interpolate(down, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return x - up


class ForensicEncoder(nn.Module):
    def __init__(self, projection_dim: int = 256):
        super().__init__()
        self.npr = NPRTransform(scale=0.5)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.projection = nn.Linear(512, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residuals = self.npr(image)
        feats = self.backbone(residuals)
        return self.norm(self.projection(feats))


class SemanticEncoder(nn.Module):
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


# ==============================================================================
# 2. DATASET EXTRACTION & SPLITS
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
) -> list[dict]:
    target_path = Path(target_root)
    cached_manifest = target_path / "all_images_manifest.json"
    if cached_manifest.exists():
        logger.info("Found cached manifest at %s", cached_manifest)
        with open(cached_manifest, "r", encoding="utf-8") as f:
            return json.load(f)

    target_path.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    train_parquets = sorted([f for f in repo_files if "train-" in f and f.endswith(".parquet")])
    val_parquets = sorted([f for f in repo_files if "validation-" in f and f.endswith(".parquet")])

    logger.info("Found %d train parquets and %d val parquets on Hugging Face.", len(train_parquets), len(val_parquets))

    records: list[dict] = []
    counter = 0

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

            try:
                os.remove(downloaded)
            except Exception:
                pass

            logger.info("  Extracted %s (%d images in %.1fs)", Path(rel_file).name, len(table), time.time() - t0)

    with open(cached_manifest, "w", encoding="utf-8") as f:
        json.dump(records, f)
    logger.info("Total images extracted & cached: %d", len(records))
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
        logger.info("  %-25s: %5d total (%5d real, %5d fake)", k, len(v), reals, fakes)

    return splits


class FastImageDataset(Dataset):
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
# 3. EVALUATION & THRESHOLD METRICS
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


def get_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for clip_imgs, foren_imgs, labels, _, _ in loader:
            clip_imgs = clip_imgs.to(device)
            foren_imgs = foren_imgs.to(device)
            logits = model(clip_imgs, foren_imgs).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy().tolist()
            all_scores.extend(scores)
            all_labels.extend(int(l) for l in labels.numpy())

    return np.array(all_labels), np.array(all_scores)


def compute_metrics_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_scores >= threshold).astype(int)
    auroc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else None
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "auroc": auroc,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "total_samples": len(y_true),
    }


# ==============================================================================
# 4. MAIN ORCHESTRATION
# ==============================================================================

def main():
    print("=" * 80)
    print("   FORENSIGHT: LOGO EVALUATION AT THRESHOLD 0.50 vs 0.73 (GPU INFERENCE)   ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path("/kaggle/working")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    all_records = prepare_tiny_genimage(target_root="/tmp/tiny_genimage")
    splits = build_logo_splits(all_records, leave_out_gen="midjourney", n_val_per_gen=100)

    clip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    forensic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 2. Locate or Train LOGO model
    found_ckpt = None
    if Path("/kaggle/input").exists():
        logger.info("Scanning /kaggle/input for pre-trained checkpoints...")
        for p in Path("/kaggle/input").rglob("*logo*midjourney*.pt"):
            logger.info("Found checkpoint in /kaggle/input: %s", p)
            found_ckpt = p
            break
        if not found_ckpt:
            for p in Path("/kaggle/input").rglob("*.pt"):
                logger.info("Found candidate .pt file: %s", p)
                if "logo" in p.name.lower():
                    found_ckpt = p
                    break

    if not found_ckpt:
        for c in [
            output_dir / "logo_midjourney_best.pt",
            Path("results/r2_kaggle/checkpoints/logo_midjourney_best.pt"),
        ]:
            if c.exists():
                found_ckpt = c
                break

    model = FusionDetector(hidden_dim=128, dropout=0.2).to(device)

    if found_ckpt:
        logger.info("Found pre-trained LOGO checkpoint at %s! Loading directly...", found_ckpt)
        model.load_state_dict(torch.load(found_ckpt, map_location=device))
    else:
        logger.info("Pre-trained checkpoint not found in inputs. Training LOGO model (5 epochs)...")
        train_ds = FastImageDataset(splits["train"], clip_transform, forensic_transform)
        val_ds = FastImageDataset(splits["val"], clip_transform, forensic_transform)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        best_val_loss = float("inf")
        ckpt_save_path = output_dir / "logo_midjourney_best.pt"

        for epoch in range(1, 6):
            model.train()
            model.semantic_encoder.backbone.eval()
            total_loss = 0.0
            n_batches = 0
            for c_img, f_img, labels, _, _ in train_loader:
                c_img = c_img.to(device)
                f_img = f_img.to(device)
                targets = labels.float().to(device)
                optimizer.zero_grad()
                logits = model(c_img, f_img).squeeze(-1)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()

            model.eval()
            v_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for c_img, f_img, labels, _, _ in val_loader:
                    c_img, f_img = c_img.to(device), f_img.to(device)
                    targets = labels.float().to(device)
                    v_loss += criterion(model(c_img, f_img).squeeze(-1), targets).item()
                    v_batches += 1
            avg_v_loss = v_loss / max(1, v_batches)
            logger.info("Epoch %d/5 | Train Loss: %.4f | Val Loss: %.4f", epoch, total_loss / n_batches, avg_v_loss)
            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                torch.save(model.state_dict(), ckpt_save_path)
                logger.info("  --> Saved best LOGO checkpoint to %s", ckpt_save_path)

        model.load_state_dict(torch.load(ckpt_save_path, map_location=device))

    # Also make sure the model is saved to /kaggle/working/logo_midjourney_best.pt for download
    dest_ckpt = output_dir / "logo_midjourney_best.pt"
    if not dest_ckpt.exists():
        torch.save(model.state_dict(), dest_ckpt)

    # 3. Validation Calibrated Threshold
    val_ds = FastImageDataset(splits["val"], clip_transform, forensic_transform)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    val_true, val_scores = get_predictions(model, val_loader, device)
    calibrated_tau = select_threshold(val_true, val_scores, strategy="f1")
    logger.info("Calibrated validation threshold on seen generators: tau* = %.4f", calibrated_tau)

    # 4. Run Inference on Test Sets
    thresholds_to_test = [0.50, round(float(calibrated_tau), 2)]
    eval_reports: dict[str, Any] = {
        "experiment": "logo_threshold_comparison",
        "calibrated_threshold": calibrated_tau,
        "results": {},
        "reproducibility": {
            "run_id": "kaggle_r2_logo_eval",
            "experiment_name": "logo_threshold_comparison",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": None,
            "seed": 42,
            "split_version": "tiny_genimage_logo_v1",
            "config": {
                "architecture": "FusionDetector (CLIP ViT-B/32 + ResNet18 NPR)",
                "batch_size": 64,
                "dataset": "TheKernel01/Tiny-GenImage",
                "thresholds": thresholds_to_test,
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
        },
    }

    test_splits = {
        "test_midjourney": splits["test_midjourney"],
        "test_in_domain_seen": splits["test_in_domain_seen"],
    }

    print("\n" + "=" * 96)
    print("                     EVALUATION RESULTS ACROSS THRESHOLDS (0.50 vs 0.73)                     ")
    print("=" * 96)

    for split_key, split_data in test_splits.items():
        ds = FastImageDataset(split_data, clip_transform, forensic_transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        y_true, y_scores = get_predictions(model, loader, device)

        eval_reports["results"][split_key] = {}
        for tau in thresholds_to_test:
            metrics = compute_metrics_at_threshold(y_true, y_scores, threshold=tau)
            eval_reports["results"][split_key][f"threshold_{tau:.2f}"] = metrics

            cm = metrics["confusion_matrix"]
            auroc_str = f"{metrics['auroc']:.4f}" if metrics.get("auroc") is not None else "N/A"
            print(f"[{split_key}] Threshold: {tau:.2f}")
            print(f"  AUROC:     {auroc_str}")
            print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision'] * 100:.2f}%")
            print(f"  Recall:    {metrics['recall'] * 100:.2f}%")
            print(f"  Confusion: TP={cm['tp']} | FP={cm['fp']} | TN={cm['tn']} | FN={cm['fn']} (Total={metrics['total_samples']})\n")

    # 5. Save Final Report
    report_path = output_dir / "logo_threshold_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_reports, f, indent=2)
    logger.info("Saved threshold comparison report to %s", report_path)
    print("=" * 96)
    print("EVALUATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
