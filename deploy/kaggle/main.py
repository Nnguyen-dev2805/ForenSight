#!/usr/bin/env python3
"""ForenSight Milestone R2: Standalone Kaggle GPU Execution Script.

This script runs on Kaggle GPU (e.g., NVIDIA T4 x2) to:
1. Download and extract TheKernel01/Tiny-GenImage dataset from Hugging Face Hub (pinned revision).
2. Build generator-disjoint train, validation, and OOD test manifests.
3. Train canonical ForenSight R2 architectures:
   - Semantic-only (frozen CLIP ViT-L/14 + projection + MLP)
   - Forensic-only (trainable ResNet18 weights=None + NPR residual transform + projection + MLP)
   - Concat Fusion (combining semantic and forensic representations)
4. Calibrate the decision threshold tau* strictly on the validation set.
5. Evaluate against In-Domain (in_domain_test) and Held-Out (cross_generator_ood) distributions.
6. Export the complete, strict ForenSight Artifact Contract to /kaggle/working/results/r2/
   and package into /kaggle/working/results_r2.zip for local reproduction.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
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
import zipfile

# Ensure required packages are present on Kaggle environment
REQUIRED_PACKAGES = {
    "open_clip": "open_clip_torch",
    "pyarrow": "pyarrow",
    "huggingface_hub": "huggingface_hub",
}
for mod_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(mod_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

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
from torchvision.models import resnet18

import open_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForenSight-Kaggle-R2")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducible data partitioning, shuffling, and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. MODEL ARCHITECTURES (CANONICAL R2 SPEC: ViT-L/14 + ResNet18 weights=None)
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
    """Forensic branch: NPR transform -> ResNet18 (weights=None) -> 256-d projection."""

    def __init__(self, projection_dim: int = 256, scale_factor: float = 0.5, mode: str = "bilinear"):
        super().__init__()
        self.npr = NPRTransform(scale_factor=scale_factor, mode=mode)
        # Canonical R2 constraint: weights=None (untrained backbone trained from scratch on residuals)
        backbone = resnet18(weights=None)
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
    """Semantic branch: Frozen CLIP (ViT-L/14 default) -> 256-d projection."""

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", projection_dim: int = 256):
        super().__init__()
        self.model_name = model_name
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.backbone = clip_model.visual
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        # OpenCLIP visual output_dim is 768 for ViT-L/14, 512 for ViT-B/32
        feature_dim = int(getattr(self.backbone, "output_dim", 768 if "ViT-L" in model_name or "L" in model_name else 512))
        self.projection = nn.Linear(feature_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(image)
        return self.norm(self.projection(feats.float()))


class FusionDetector(nn.Module):
    """Concat Fusion detector: [Semantic (256), Forensic (256)] -> MLP -> Logit."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.semantic_encoder = SemanticEncoder(model_name=model_name, pretrained=pretrained, projection_dim=256)
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

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = SemanticEncoder(model_name=model_name, pretrained=pretrained, projection_dim=256)
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
    """Forensic-only binary detector: NPR -> ResNet18 (weights=None) -> 256 -> MLP -> logit."""

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


def build_detector(
    variant: str,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> nn.Module:
    """Build detector for any of the 3 canonical R2 ablation variants."""
    v = variant.lower().strip()
    if v in ("fusion", "concat_fusion"):
        return FusionDetector(model_name=model_name, pretrained=pretrained, hidden_dim=hidden_dim, dropout=dropout)
    elif v in ("semantic", "semantic_only"):
        return SemanticOnlyDetector(model_name=model_name, pretrained=pretrained, hidden_dim=hidden_dim, dropout=dropout)
    elif v in ("forensic", "forensic_only"):
        return ForensicOnlyDetector(hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown detector variant '{variant}'. Supported: fusion, semantic_only, forensic_only")


# ==============================================================================
# 2. DATASET DOWNLOAD, PINNED REVISION & ZERO LEAKAGE PARTITIONING
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
    revision: str = "main",
) -> tuple[list[dict[str, Any]], str]:
    """Download Parquet shards with pinned revision, resolve HF commit SHA, and extract images to disk calculating SHA-256 hashes."""
    target_path = Path(target_root)
    target_path.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    resolved_commit_sha = revision
    try:
        info = api.dataset_info(repo_id=repo_id, revision=revision)
        if hasattr(info, "sha") and info.sha:
            resolved_commit_sha = str(info.sha)
            logger.info("Resolved HF revision tag '%s' -> commit SHA: %s", revision, resolved_commit_sha)
    except Exception as e:
        logger.warning("Could not resolve dataset commit SHA via HfApi (%s) for revision '%s'", e, revision)
        if revision != "main":
            try:
                info = api.dataset_info(repo_id=repo_id, revision="main")
                if hasattr(info, "sha") and info.sha:
                    resolved_commit_sha = str(info.sha)
                    logger.info("Fell back to 'main' branch -> commit SHA: %s", resolved_commit_sha)
            except Exception as e2:
                logger.warning("Fallback to 'main' branch also failed: %s", e2)

    repo_files = api.list_repo_files(repo_id=repo_id, revision=resolved_commit_sha, repo_type="dataset")

    train_parquets = sorted([f for f in repo_files if "train-" in f and f.endswith(".parquet")])
    val_parquets = sorted([f for f in repo_files if "validation-" in f and f.endswith(".parquet")])

    logger.info("Found %d train parquets and %d val parquets on HF (%s@%s -> SHA: %s).",
                len(train_parquets), len(val_parquets), repo_id, revision, resolved_commit_sha)

    records: list[dict[str, Any]] = []
    counter = 0

    for split_name, file_list in [("train", train_parquets), ("validation", val_parquets)]:
        logger.info("Downloading and extracting %d %s parquets...", len(file_list), split_name)
        for rel_file in file_list:
            t0 = time.time()
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=rel_file,
                revision=resolved_commit_sha,
                repo_type="dataset",
                local_dir="/tmp/hf_cache",
            )
            table = pq.read_table(downloaded)
            img_col = table.column("image")
            lbl_col = table.column("label")
            gen_col = table.column("generator")

            for i in range(len(table)):
                raw_bytes = extract_raw_bytes(img_col[i])
                content_hash = hashlib.sha256(raw_bytes).hexdigest()
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
                    "content_hash": content_hash,
                })
                counter += 1

            try:
                os.remove(downloaded)
            except Exception:
                pass

            logger.info("  Extracted %s (%d images in %.1fs)", Path(rel_file).name, len(table), time.time() - t0)

    logger.info("Total images extracted: %d (HF SHA: %s)", len(records), resolved_commit_sha)
    return records, resolved_commit_sha


def build_canonical_r2_splits(
    records: list[dict],
    n_val: int = 100,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build canonical ForenSight R2 protocol splits per docs/plans/r2-forensic-perception-v1.md.

    - Train: GenImage SD1.4 train fakes + Nature train reals (SD1.4-only training).
    - Val: GenImage SD1.4 validation fakes (n_val) + Nature validation reals (n_val) (for checkpointing & threshold tau*).
    - Test In-Domain: GenImage SD1.4 held-out validation fakes (remaining after n_val) + allocated Nature validation reals -> in_domain_test.
    - Test Near-OOD: GenImage SD1.5 validation fakes + allocated Nature validation reals -> near_ood.
    - Test Cross-Generator OOD: GenImage unseen generators (midjourney, adm, biggan, glide, vqdm) + allocated Nature validation reals -> cross_generator_ood.
    All test records are completely disjoint with zero duplicate sample_ids.
    """
    rng = random.Random(seed)

    train_reals = [r for r in records if r["raw_split"] == "train" and r["label"] == 0]
    val_reals = [r for r in records if r["raw_split"] == "validation" and r["label"] == 0]

    # 1. Train split: SD1.4 only
    sd14_train_fakes = [
        r for r in records if r["raw_split"] == "train" and r["generator"] == "sd14" and r["label"] == 1
    ]
    n_train = min(len(sd14_train_fakes), len(train_reals))
    selected_train_fakes = [copy.deepcopy(x) for x in sd14_train_fakes[:n_train]]
    selected_train_reals = [copy.deepcopy(x) for x in train_reals[:n_train]]
    train_set = selected_train_fakes + selected_train_reals
    rng.shuffle(train_set)
    for r in train_set:
        r["split"] = "train"

    # 2. Real allocation tracker for validation & test splits
    real_offset = 0

    def allocate_reals(n: int) -> list[dict]:
        nonlocal real_offset
        alloc: list[dict] = []
        rem_val = len(val_reals) - real_offset
        if rem_val > 0:
            take_val = min(n, rem_val)
            alloc.extend([copy.deepcopy(x) for x in val_reals[real_offset : real_offset + take_val]])
            real_offset += take_val
        if len(alloc) < n:
            needed = n - len(alloc)
            unused_train = train_reals[n_train:]
            alloc.extend([copy.deepcopy(x) for x in unused_train[:needed]])
        return alloc

    # 3. Val split: SD1.4 val fakes (n_val) + equal reals
    sd14_val_fakes = [
        r for r in records if r["raw_split"] == "validation" and r["generator"] == "sd14" and r["label"] == 1
    ]
    n_val_actual = min(n_val, len(sd14_val_fakes) // 2) if len(sd14_val_fakes) >= 2 else len(sd14_val_fakes)
    cur_val_fakes = [copy.deepcopy(x) for x in sd14_val_fakes[:n_val_actual]]
    cur_val_reals = allocate_reals(len(cur_val_fakes))
    if len(cur_val_reals) < len(cur_val_fakes):
        cur_val_fakes = cur_val_fakes[:len(cur_val_reals)]
    val_set = cur_val_fakes + cur_val_reals
    for r in val_set:
        r["split"] = "val"

    # 4. In-domain test split: remaining SD1.4 val fakes + equal reals
    remaining_sd14_fakes = [copy.deepcopy(x) for x in sd14_val_fakes[n_val_actual:]]
    cur_in_domain_reals = allocate_reals(len(remaining_sd14_fakes))
    if len(cur_in_domain_reals) < len(remaining_sd14_fakes):
        remaining_sd14_fakes = remaining_sd14_fakes[:len(cur_in_domain_reals)]
    in_domain_test_set = remaining_sd14_fakes + cur_in_domain_reals
    for r in in_domain_test_set:
        r["split"] = "in_domain_test"
        r["eval_slice"] = "in_domain_test"

    # 5. Near-OOD test split: SD1.5 fakes + equal reals
    sd15_fakes = [
        copy.deepcopy(r) for r in records if r["generator"] == "sd15" and r["label"] == 1
    ]
    cur_near_ood_reals = allocate_reals(len(sd15_fakes))
    if len(cur_near_ood_reals) < len(sd15_fakes):
        sd15_fakes = sd15_fakes[:len(cur_near_ood_reals)]
    near_ood_test_set = sd15_fakes + cur_near_ood_reals
    for r in near_ood_test_set:
        r["split"] = "near_ood"
        r["eval_slice"] = "near_ood"

    # 6. Cross-generator OOD test split: unseen generators (Midjourney, ADM, BigGAN, GLIDE, VQDM)
    cross_fakes = [
        copy.deepcopy(r)
        for r in records
        if r["generator"] not in ("nature", "sd14", "sd15") and r["label"] == 1
    ]
    cur_cross_reals = allocate_reals(len(cross_fakes))
    if len(cur_cross_reals) < len(cross_fakes):
        cross_fakes = cross_fakes[:len(cur_cross_reals)]
    cross_ood_test_set = cross_fakes + cur_cross_reals
    for r in cross_ood_test_set:
        r["split"] = "cross_generator_ood"
        r["eval_slice"] = "cross_generator_ood"

    splits = {
        "train": train_set,
        "val": val_set,
        "in_domain_test": in_domain_test_set,
        "near_ood": near_ood_test_set,
        "cross_generator_ood": cross_ood_test_set,
    }

    logger.info("Canonical R2 Protocol Splits created (seed: %d):", seed)
    for k, v in splits.items():
        reals = sum(1 for r in v if r["label"] == 0)
        fakes = sum(1 for r in v if r["label"] == 1)
        logger.info("  %s: %d total (%d real, %d fake)", k, len(v), reals, fakes)

    return splits


def load_splits_from_manifests(
    train_manifest: str | Path,
    val_manifest: str | Path,
    test_manifest: str | Path | None = None,
    eval_manifests: list[str | Path] | None = None,
    base_dir: Path | None = None,
    extracted_records: list[dict] | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Load canonical R2 splits from sealed R0 manifests per docs/plans/r2-forensic-perception-v1.md.

    Reconciles manifest entries with extracted image files and byte content hashes.
    Ensures zero leakage and respects sealed partition boundaries.
    """
    rec_lookup_by_id: dict[str, dict] = {}
    rec_lookup_by_path: dict[str, dict] = {}
    if extracted_records:
        for r in extracted_records:
            rec_lookup_by_id[r["sample_id"]] = r
            if r.get("image_path"):
                rec_lookup_by_path[r["image_path"]] = r
                rec_lookup_by_path[Path(r["image_path"]).name] = r

    def _resolve_record(raw_row: dict) -> dict:
        sample_id = str(raw_row["sample_id"])
        image_path = raw_row.get("image_path") or raw_row.get("rel_path") or raw_row.get("path")
        lookup = rec_lookup_by_id.get(sample_id)
        if not lookup and image_path:
            lookup = rec_lookup_by_path.get(image_path) or rec_lookup_by_path.get(Path(image_path).name)

        abs_path = None
        content_hash = raw_row.get("content_hash")
        if lookup:
            abs_path = lookup.get("abs_path")
            if not content_hash:
                content_hash = lookup.get("content_hash")

        if not abs_path and image_path:
            p = Path(image_path)
            if p.is_absolute() and p.exists():
                abs_path = str(p)
            elif base_dir is not None and (base_dir / image_path).exists():
                abs_path = str(base_dir / image_path)
            elif p.exists():
                abs_path = str(p.resolve())
            else:
                abs_path = str(base_dir / image_path) if base_dir else str(image_path)

        generator = raw_row.get("generator") or (lookup.get("generator") if lookup else "unknown")
        split = raw_row.get("split") or "train"
        label = int(raw_row["label"])
        dataset = raw_row.get("dataset") or "TheKernel01/Tiny-GenImage"
        meta = raw_row.get("metadata") if isinstance(raw_row.get("metadata"), dict) else {}
        eval_slice = meta.get("eval_slice") or raw_row.get("eval_slice") or split

        return {
            "sample_id": sample_id,
            "image_path": str(image_path) if image_path else "",
            "abs_path": abs_path,
            "label": label,
            "generator": generator,
            "split": split,
            "dataset": dataset,
            "eval_slice": eval_slice,
            "content_hash": content_hash,
        }

    def _load_file(path_str: str | Path) -> list[dict]:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Sealed manifest file not found: {path}")
        rows: list[dict] = []
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif path.suffix.lower() == ".csv":
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        else:
            raise ValueError(f"Unsupported manifest file extension: {path.suffix}")
        return [_resolve_record(r) for r in rows]

    train_recs = _load_file(train_manifest)
    val_recs = _load_file(val_manifest)

    splits: dict[str, list[dict]] = {
        "train": train_recs,
        "val": val_recs,
    }

    if eval_manifests:
        for p in eval_manifests:
            p_obj = Path(p)
            split_key = p_obj.stem.replace("_manifest", "")
            recs = _load_file(p)
            for r in recs:
                r["split"] = split_key
                r["eval_slice"] = split_key
            splits[split_key] = recs
    elif test_manifest:
        test_recs = _load_file(test_manifest)
        slices: dict[str, list[dict]] = {}
        for r in test_recs:
            s = r.get("eval_slice") or r.get("split") or "test"
            if s in ("train", "val"):
                s = "in_domain_test"
            slices.setdefault(s, []).append(r)

        if len(slices) == 1 and "test" in slices:
            splits["test"] = slices["test"]
        else:
            for s_name, s_recs in slices.items():
                splits[s_name] = s_recs

    logger.info("Loaded sealed splits from manifests:")
    for k, v in splits.items():
        reals = sum(1 for r in v if r["label"] == 0)
        fakes = sum(1 for r in v if r["label"] == 1)
        logger.info("  %s: %d total (%d real, %d fake)", k, len(v), reals, fakes)

    return splits


def build_logo_splits(
    records: list[dict],
    leave_out_gen: str = "midjourney",
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build Leave-One-Generator-Out (LOGO) splits controlled by seed.
    - Train: All train fakes from seen generators (6 gens x 2,000 = 12,000 fakes + 12,000 reals = 24,000 images).
    - Val: n_val_per_gen fakes from seen generators (6 x 100 = 600 fakes + 600 reals = 1,200 images).
    - Test Held-Out (leave_out_gen): All available fakes (2,500 fakes + 2,500 reals = 5,000 images) -> cross_generator_ood.
    - Test In-Domain Seen: Remaining validation fakes from seen generators (2,400 fakes + 2,400 reals = 4,800 images) -> in_domain_test.
    """
    rng = random.Random(seed)
    all_fake_gens = sorted(list({r["generator"] for r in records if r["label"] == 1}))
    seen_gens = [g for g in all_fake_gens if g != leave_out_gen]

    train_reals = [r for r in records if r["raw_split"] == "train" and r["label"] == 0]
    val_reals = [r for r in records if r["raw_split"] == "validation" and r["label"] == 0]

    # 1. Train split
    train_fakes = [r for r in records if r["raw_split"] == "train" and r["generator"] in seen_gens and r["label"] == 1]
    n_train = min(len(train_fakes), len(train_reals))
    selected_train_fakes = train_fakes[:n_train]
    selected_train_reals = train_reals[:n_train]
    train_set = selected_train_fakes + selected_train_reals
    rng.shuffle(train_set)
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
        r["split"] = "in_domain_test"
        r["eval_slice"] = "test_in_domain_seen"

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
        r["split"] = "cross_generator_ood"
        r["eval_slice"] = f"test_{leave_out_gen}"

    splits = {
        "train": train_set,
        "val": val_set,
        "cross_generator_ood": test_held_out,
        "in_domain_test": test_seen_set,
    }

    logger.info("LOGO Splits created (held-out: %s, seed: %d):", leave_out_gen, seed)
    for k, v in splits.items():
        reals = sum(1 for r in v if r["label"] == 0)
        fakes = sum(1 for r in v if r["label"] == 1)
        logger.info("  %s: %d total (%d real, %d fake)", k, len(v), reals, fakes)

    return splits


def build_all_in_one_splits(
    records: list[dict],
    n_val_per_gen: int = 100,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build All-In-One multi-generator splits controlled by seed with ZERO duplicate test samples.
    - Train: All train fakes from all 7 generators (7 x 2,000 = 14,000 fakes + 14,000 reals = 28,000 images).
    - Val: 100 fakes from each generator (7 x 100 = 700 fakes + 700 reals = 1,400 images).
    - Disjoint per-generator test sets (400 fakes + 400 reals = 800 images per gen).
    NOTE: Does NOT return a redundant 'test_all_combined' key so each sample is evaluated exactly once.
    """
    rng = random.Random(seed)
    all_fake_gens = sorted(list({r["generator"] for r in records if r["label"] == 1}))

    train_reals = [r for r in records if r["raw_split"] == "train" and r["label"] == 0]
    val_reals = [r for r in records if r["raw_split"] == "validation" and r["label"] == 0]

    train_fakes = [r for r in records if r["raw_split"] == "train" and r["label"] == 1]
    n_train = min(len(train_fakes), len(train_reals))
    selected_train_fakes = train_fakes[:n_train]
    selected_train_reals = train_reals[:n_train]
    train_set = selected_train_fakes + selected_train_reals
    rng.shuffle(train_set)
    for r in train_set:
        r["split"] = "train"

    real_offset = 0

    def allocate_val_reals(n: int) -> list[dict]:
        nonlocal real_offset
        alloc = val_reals[real_offset : real_offset + n]
        real_offset += len(alloc)
        return alloc

    val_set: list[dict] = []
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
            r["split"] = "in_domain_test"
            r["eval_slice"] = f"test_{gen}"
        per_gen_test[f"test_{gen}"] = cur_test

    for r in val_set:
        r["split"] = "val"

    splits = {
        "train": train_set,
        "val": val_set,
    }
    splits.update(per_gen_test)

    logger.info("All-In-One Splits created (seed: %d):", seed)
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

try:
    from forensight.evaluation.metrics import select_threshold as r0_select_threshold
except ImportError:
    r0_select_threshold = None


def select_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    strategy: str = "f1",
    default_threshold: float = 0.5,
) -> float:
    """Select optimal decision threshold using R0 score boundaries, midpoints, and deterministic tie-breaking."""
    if r0_select_threshold is not None:
        try:
            return float(r0_select_threshold(y_true, y_scores, strategy=strategy, default_threshold=default_threshold))
        except Exception:
            pass

    strat = strategy.lower().strip()
    if strat not in ("f1", "accuracy", "youden"):
        raise ValueError(f"Unknown threshold selection strategy '{strategy}'. Supported: 'f1', 'accuracy', 'youden'")

    y_t = np.asarray(y_true, dtype=np.int64)
    y_s = np.asarray(y_scores, dtype=np.float64)
    if len(np.unique(y_t)) < 2:
        return default_threshold

    order = np.argsort(y_s)
    y_s_sorted = y_s[order]
    y_t_sorted = y_t[order]
    n_samples = len(y_s_sorted)

    unique_scores = np.unique(y_s_sorted)
    upper_bound = np.nextafter(unique_scores.max(), np.inf)
    if len(unique_scores) > 1:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidate_list = [unique_scores, midpoints, [upper_bound]]
        if unique_scores.min() <= default_threshold <= upper_bound:
            candidate_list.append([default_threshold])
        candidates = np.unique(np.concatenate(candidate_list))
    else:
        candidates = np.unique(np.concatenate([unique_scores, [upper_bound]]))

    idx = np.searchsorted(y_s_sorted, candidates, side="left")
    cum_pos = np.cumsum(y_t_sorted)
    total_pos = cum_pos[-1]
    total_neg = n_samples - total_pos

    pos_before = np.where(idx > 0, cum_pos[idx - 1], 0)
    tp = total_pos - pos_before
    fp = (n_samples - idx) - tp
    fn = total_pos - tp
    tn = total_neg - fp

    if strat == "f1":
        denom_f1 = total_pos + tp + fp
        metric_vals = (2.0 * tp) / np.maximum(denom_f1, 1e-12)
    elif strat == "accuracy":
        metric_vals = (tp + tn) / n_samples
    elif strat == "youden":
        tpr = tp / total_pos if total_pos > 0 else np.zeros_like(tp, dtype=float)
        fpr = fp / total_neg if total_neg > 0 else np.zeros_like(fp, dtype=float)
        metric_vals = tpr - fpr
    else:
        raise ValueError(f"Unhandled strategy: {strat}")

    max_val = np.max(metric_vals)
    best_mask = np.isclose(metric_vals, max_val, rtol=0.0, atol=1e-12)
    tied_candidates = candidates[best_mask]

    best_threshold = min(
        tied_candidates,
        key=lambda c: (abs(c - default_threshold), -c),
    )
    return float(best_threshold)


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
    dataset_revision: str = "hf:TheKernel01/Tiny-GenImage@v1.0",
) -> dict[str, Any]:
    """Perform a strict 3-way partition audit with true SHA-256 byte content hashing."""
    train_ids = {r["sample_id"] for r in train_records}
    val_ids = {r["sample_id"] for r in val_records}
    test_ids = {r["sample_id"] for r in test_records}

    train_val_id_overlap = len(train_ids & val_ids)
    train_test_id_overlap = len(train_ids & test_ids)
    val_test_id_overlap = len(val_ids & test_ids)

    # Content-hash based duplicate detection (honest verification)
    has_hashes = (
        all("content_hash" in r for r in train_records)
        and all("content_hash" in r for r in val_records)
        and all("content_hash" in r for r in test_records)
    )

    if has_hashes:
        train_hashes = {r["content_hash"] for r in train_records}
        val_hashes = {r["content_hash"] for r in val_records}
        test_hashes = {r["content_hash"] for r in test_records}

        dup_train_val = len(train_hashes & val_hashes)
        dup_train_test = len(train_hashes & test_hashes)
        dup_val_test = len(val_hashes & test_hashes)
        exact_duplicates = dup_train_val + dup_train_test + dup_val_test
        content_hash_verified = True
    else:
        dup_train_val = train_val_id_overlap
        dup_train_test = train_test_id_overlap
        dup_val_test = val_test_id_overlap
        exact_duplicates = dup_train_val + dup_train_test + dup_val_test
        content_hash_verified = False

    leakage_detected = (dup_train_val > 0) or (dup_train_test > 0) or (dup_val_test > 0)

    return {
        "status": "PASS" if not leakage_detected else "FAILED",
        "verdict": "PASS" if not leakage_detected else "FAIL",
        "dataset": "TheKernel01/Tiny-GenImage",
        "dataset_revision": dataset_revision,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "leakage": {
            "train_val_overlap": dup_train_val,
            "train_test_overlap": dup_train_test,
            "val_test_overlap": dup_val_test,
            "exact_duplicates": exact_duplicates,
            "content_hash_method": "sha256" if content_hash_verified else "sample_id_fallback",
            "content_hash_verified": content_hash_verified,
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


def save_manifest_jsonl(records: list[dict], path: Path, default_split: str = "train") -> None:
    """Save manifest records strictly ensuring split is in VALID_SPLIT_NAMES while preserving eval_slice."""
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
                "metadata": {
                    "eval_slice": r.get("eval_slice") or r.get("split"),
                },
            }
            f.write(json.dumps(manifest_row) + "\n")


def save_predictions_jsonl(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")


def get_git_commit_sha() -> str:
    """Resolve genuine git commit provenance: local git HEAD, commit_info.json, or env var."""
    # 1. Live git repository has ultimate authority when available
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        sha = res.stdout.strip()
        if res.returncode == 0 and sha and len(sha) == 40:
            return sha
    except Exception:
        pass

    # 2. Check injected commit_info.json (used when running on Kaggle without .git)
    commit_file = Path(__file__).resolve().parent / "commit_info.json"
    if commit_file.exists():
        try:
            with open(commit_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                sha = data.get("git_commit")
                if sha and sha.strip() and sha != "unversioned":
                    return sha.strip()
        except Exception:
            pass

    # 3. Check environment variable
    env_sha = os.environ.get("GIT_COMMIT")
    if env_sha and env_sha.strip():
        return env_sha.strip()

    return "unversioned_kaggle_run"


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

            # Route input depending on model architecture
            if isinstance(model, ForensicOnlyDetector):
                logits = model(forensic_img=foren_imgs).squeeze(-1)
            elif isinstance(model, SemanticOnlyDetector):
                logits = model(clip_img=clip_imgs).squeeze(-1)
            else:
                logits = model(clip_imgs, foren_imgs).squeeze(-1)

            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            all_scores.extend(probs)
            all_labels.extend(int(l) for l in labels.numpy())
            all_gens.extend(gens)
            for idx_val, prob, lbl, gen in zip(indices.numpy(), probs, labels.numpy(), gens):
                rec = records[int(idx_val)]
                pred = int(prob >= threshold)
                canonical_split = rec.get("split") or (split_name.replace("test_", "") if split_name.startswith("test_") and split_name != "test" else split_name)
                pred_records.append({
                    "sample_id": rec["sample_id"],
                    "path": rec.get("image_path") or rec.get("rel_path"),
                    "label": int(lbl),
                    "score": float(prob),
                    "prediction": pred,
                    "generator": str(gen),
                    "split": canonical_split,
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
    dataset_revision: str = "hf:TheKernel01/Tiny-GenImage@v1.0",
    variant: str = "fusion",
    clip_model: str = "ViT-L-14",
    hidden_dim: int = 128,
    dropout: float = 0.2,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    # Enforce random seed globally
    set_seed(seed)

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

    # 1. Aggregate Test Splits and assert ZERO sample duplication
    test_splits = sorted([k for k in splits.keys() if k not in ("train", "val")])
    all_test_records: list[dict] = []
    for split_name in test_splits:
        all_test_records.extend(splits[split_name])

    test_ids = [r["sample_id"] for r in all_test_records]
    if len(test_ids) != len(set(test_ids)):
        dup_count = len(test_ids) - len(set(test_ids))
        raise ValueError(f"CRITICAL ERROR: test splits contain {dup_count} duplicate samples!")

    save_manifest_jsonl(splits["train"], run_dir / "train_manifest.jsonl", default_split="train")
    save_manifest_jsonl(splits["val"], run_dir / "val_manifest.jsonl", default_split="val")
    save_manifest_jsonl(all_test_records, run_dir / "test_manifest.jsonl", default_split="in_domain_test")
    logger.info("Saved train, val, and test manifests to %s (%d test records, 0 duplicates)", run_dir, len(all_test_records))

    # 2. Strict 3-way Dataset Audit with content-hash verification
    audit_data = generate_dataset_audit(splits["train"], splits["val"], all_test_records, dataset_revision=dataset_revision)
    with open(run_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    logger.info("Dataset audit completed: %s (exact_duplicates=%s, hash_verified=%s)",
                audit_data["status"], audit_data["leakage"]["exact_duplicates"], audit_data["leakage"]["content_hash_verified"])

    # 3. Save config.json with canonical variant name accepted by load_r2_config
    v_norm = "semantic" if variant in ("semantic", "semantic_only") else ("forensic" if variant in ("forensic", "forensic_only") else "fusion")
    if learning_rate is not None:
        lr = float(learning_rate)
    else:
        # Canonical R2 configs: semantic uses 1e-3, forensic and fusion use 1e-4
        lr = 1e-3 if v_norm == "semantic" else 1e-4

    config_dict = {
        "variant": v_norm,
        "model_variant": variant,
        "seed": seed,
        "model": {
            "architecture": model.__class__.__name__,
            "clip_model": clip_model,
            "clip_pretrained": "openai",
            "projection_dim": 256,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "image_size": 224,
            "npr_scale_factor": 0.5,
            "npr_mode": "bilinear",
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": 1e-4,
            "num_workers": num_workers,
        },
        "evaluation": {
            "threshold_strategy": "f1",
        },
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 4. Data Loaders with seeded shuffling
    train_ds = FastImageDataset(splits["train"], clip_transform, forensic_transform)
    val_ds = FastImageDataset(splits["val"], clip_transform, forensic_transform)

    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_generator, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
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
            if isinstance(model, ForensicOnlyDetector):
                logits = model(forensic_img=foren_imgs).squeeze(-1)
            elif isinstance(model, SemanticOnlyDetector):
                logits = model(clip_img=clip_imgs).squeeze(-1)
            else:
                logits = model(clip_imgs, foren_imgs).squeeze(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

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

                if isinstance(model, ForensicOnlyDetector):
                    logits = model(forensic_img=foren_imgs).squeeze(-1)
                elif isinstance(model, SemanticOnlyDetector):
                    logits = model(clip_img=clip_imgs).squeeze(-1)
                else:
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
    val_y_true = np.array([r["label"] for r in splits["val"]])
    val_all_scores = []
    with torch.no_grad():
        for clip_imgs, foren_imgs, _, _, _ in val_loader:
            clip_imgs = clip_imgs.to(device)
            foren_imgs = foren_imgs.to(device)
            if isinstance(model, ForensicOnlyDetector):
                logits = model(forensic_img=foren_imgs).squeeze(-1)
            elif isinstance(model, SemanticOnlyDetector):
                logits = model(clip_img=clip_imgs).squeeze(-1)
            else:
                logits = model(clip_imgs, foren_imgs).squeeze(-1)
            val_all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())

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
        canonical_split_key = split_name.replace("test_", "") if split_name.startswith("test_") and split_name != "test" else split_name
        by_split_metrics[canonical_split_key] = split_m
        all_test_predictions.extend(split_preds)
        logger.info(
            "  %-25s | AUROC: %.4f | Acc: %.4f | F1: %.4f (N=%d)",
            canonical_split_key,
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

    # Calculate Mean Cross-Generator AUROC across unseen generators
    cross_gens = [g for g in by_gen_metrics.keys() if g not in ("nature", "sd14", "sd15")]
    cross_aurocs = [
        by_gen_metrics[g]["auroc"]
        for g in cross_gens
        if by_gen_metrics[g].get("auroc") is not None
    ]
    mean_cross_gen_auroc = float(np.mean(cross_aurocs)) if cross_aurocs else None

    # 10. Save evaluation.json
    evaluation_record = {
        "overall": overall_metrics,
        "by_split": by_split_metrics,
        "by_generator": by_gen_metrics,
        "mean_cross_generator_auroc": mean_cross_gen_auroc,
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

    # 11. Save reproducibility.json with verified git commit
    commit_sha = git_commit or get_git_commit_sha()
    reproducibility = {
        "run_id": f"kaggle_r2_{experiment_name}_seed_{seed}",
        "experiment_name": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit_sha,
        "seed": seed,
        "dataset": "TheKernel01/Tiny-GenImage",
        "dataset_revision": dataset_revision,
        "split_version": "tiny_genimage_v1.0",
        "config": config_dict,
        "threshold_source": "val_optimal_f1",
        "threshold_value": float(calibrated_tau),
        "metrics": {
            "overall": overall_metrics,
            "by_split": by_split_metrics,
            "by_generator": by_gen_metrics,
            "mean_cross_generator_auroc": mean_cross_gen_auroc,
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
                arcname = file.relative_to(results_dir)
                zf.write(file, arcname)
    logger.info("Packaged %s (%.2f MB)", zip_path.name, zip_path.stat().st_size / (1024 * 1024))


# ==============================================================================
# 5. MAIN ORCHESTRATION FUNCTION
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="ForenSight R2 Canonical GPU Runner & Ablation Suite")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working", help="Output directory")
    parser.add_argument("--target-root", type=str, default="/tmp/tiny_genimage", help="Data extraction root")
    parser.add_argument("--clip-model", type=str, default="ViT-L-14", help="OpenCLIP model name (default: ViT-L-14)")
    parser.add_argument("--dataset-revision", type=str, default="main", help="Hugging Face dataset revision tag or branch (default: main)")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "fusion", "semantic_only", "forensic_only"],
        help="Ablation variants to run ('all' runs semantic-only, forensic-only, and concat fusion)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="canonical",
        choices=["all", "canonical", "logo", "all_in_one"],
        help="Experiment paradigms to execute ('canonical', 'logo', 'all_in_one', or 'all')",
    )
    parser.add_argument("--leave-out", type=str, default="midjourney", help="Held-out generator for LOGO")
    parser.add_argument("--train-manifest", type=str, default=None, help="Path to sealed training manifest (JSONL/CSV)")
    parser.add_argument("--val-manifest", type=str, default=None, help="Path to sealed validation manifest (JSONL/CSV)")
    parser.add_argument("--test-manifest", type=str, default=None, help="Path to sealed test manifest (JSONL/CSV)")
    parser.add_argument("--eval-manifests", type=str, nargs="*", default=None, help="Paths to sealed evaluation manifests (JSONL/CSV)")
    parser.add_argument("--manifest-dir", type=str, default=None, help="Directory containing sealed manifests (train_manifest.jsonl, val_manifest.jsonl, etc.)")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory for resolving relative image paths in sealed manifests")
    parser.add_argument("--learning-rate", type=float, default=None, help="Explicit learning rate override (defaults to 1e-3 for semantic, 1e-4 for forensic/fusion)")

    run_config_path = Path(__file__).resolve().parent / "run_config.json"
    if run_config_path.exists():
        try:
            with open(run_config_path, "r", encoding="utf-8") as f:
                cfg_overrides = json.load(f)
            parser.set_defaults(**cfg_overrides)
            logger.info("Loaded run configuration overrides from %s: %s", run_config_path.name, cfg_overrides)
        except Exception as e:
            logger.warning("Could not read run_config.json: %s", e)

    args, unknown = parser.parse_known_args()

    print("=" * 80)
    print("      FORENSIGHT: R2 CANONICAL GPU RUNNER & ABLATION SUITE     ")
    print("=" * 80)

    # Set seed immediately
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs
    batch_size = args.batch_size
    seed = args.seed
    num_workers = 4 if torch.cuda.is_available() else 0

    # 1. Download & Extract all images once with pinned revision and resolve commit SHA
    all_records, resolved_hf_sha = prepare_tiny_genimage(
        repo_id="TheKernel01/Tiny-GenImage",
        target_root=args.target_root,
        revision=args.dataset_revision,
    )
    full_dataset_rev = f"hf:TheKernel01/Tiny-GenImage@{resolved_hf_sha}"

    # 2. Build Transforms
    try:
        _, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained="openai")
        clip_transform = clip_preprocess
    except Exception as e:
        logger.warning("Failed to load OpenCLIP preprocess for %s (%s), falling back to standard resize.", args.clip_model, e)
        clip_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    forensic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    resolved_git_commit = get_git_commit_sha()
    logger.info("Provenance git commit: %s | Dataset revision: %s", resolved_git_commit, full_dataset_rev)

    # Determine which model variants to execute
    if args.variant == "all":
        variants_to_run = ["semantic_only", "forensic_only", "fusion"]
    else:
        variants_to_run = [args.variant]

    all_experiment_reports = {}

    # --------------------------------------------------------------------------
    # PARADIGM 0: CANONICAL R2 SPEC (SD1.4-Only Train/Val -> In-Domain, Near-OOD, Cross-Gen)
    # --------------------------------------------------------------------------
    if args.experiment in ("all", "canonical"):
        print("\n" + "#" * 80)
        print(f"# PARADIGM 0: CANONICAL R2 SPEC (SD1.4-Only Train/Val, Variants: {variants_to_run})")
        print("#" * 80)

        # Check for sealed manifests per docs/plans/r2-forensic-perception-v1.md
        train_manifest = args.train_manifest
        val_manifest = args.val_manifest
        test_manifest = args.test_manifest
        eval_manifests = args.eval_manifests

        search_dirs: list[Path] = []
        if args.manifest_dir:
            search_dirs.append(Path(args.manifest_dir))
        search_dirs.append(Path(__file__).resolve().parent / "manifests")
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for sub in kaggle_input.iterdir():
                if sub.is_dir():
                    search_dirs.append(sub)

        if not train_manifest or not val_manifest:
            for sdir in search_dirs:
                if not sdir.exists():
                    continue
                t_cand = [sdir / "train_manifest.jsonl", sdir / "train.jsonl", sdir / "train.csv"]
                v_cand = [sdir / "val_manifest.jsonl", sdir / "val.jsonl", sdir / "val.csv"]
                found_t = next((str(p) for p in t_cand if p.exists()), None)
                found_v = next((str(p) for p in v_cand if p.exists()), None)
                if found_t and found_v:
                    train_manifest = found_t
                    val_manifest = found_v
                    if not test_manifest and not eval_manifests:
                        test_cand = [sdir / "test_manifest.jsonl", sdir / "test.jsonl", sdir / "test.csv"]
                        test_manifest = next((str(p) for p in test_cand if p.exists()), None)
                    break

        if train_manifest and val_manifest:
            logger.info("Using sealed R0 manifests:")
            logger.info("  Train: %s", train_manifest)
            logger.info("  Val:   %s", val_manifest)
            if test_manifest:
                logger.info("  Test:  %s", test_manifest)
            canonical_splits = load_splits_from_manifests(
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                test_manifest=test_manifest,
                eval_manifests=eval_manifests,
                base_dir=Path(args.base_dir) if args.base_dir else None,
                extracted_records=all_records,
                seed=seed,
            )
        else:
            logger.info("No sealed manifest flags or files detected. Building canonical R2 splits on-the-fly (pilot mode).")
            canonical_splits = build_canonical_r2_splits(all_records, n_val=100, seed=seed)

        for v in variants_to_run:
            set_seed(seed)
            model = build_detector(variant=v, model_name=args.clip_model, hidden_dim=128, dropout=0.2)
            exp_name = f"canonical_r2_{v}"
            report = train_and_eval_experiment(
                experiment_name=exp_name,
                model=model,
                splits=canonical_splits,
                clip_transform=clip_transform,
                forensic_transform=forensic_transform,
                device=device,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
                git_commit=resolved_git_commit,
                dataset_revision=full_dataset_rev,
                variant=v,
                clip_model=args.clip_model,
                learning_rate=args.learning_rate,
            )
            all_experiment_reports[exp_name] = report
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # PARADIGM 1: LEAVE-ONE-GENERATOR-OUT (LOGO)
    # --------------------------------------------------------------------------
    if args.experiment in ("all", "logo"):
        print("\n" + "#" * 80)
        print(f"# PARADIGM 1: LEAVE-ONE-OUT (LOGO: Held-out {args.leave_out}, Variants: {variants_to_run})")
        print("#" * 80)

        logo_splits = build_logo_splits(all_records, leave_out_gen=args.leave_out, n_val_per_gen=100, seed=seed)

        for v in variants_to_run:
            set_seed(seed)
            model = build_detector(variant=v, model_name=args.clip_model, hidden_dim=128, dropout=0.2)
            exp_name = f"logo_{args.leave_out}_{v}"
            report = train_and_eval_experiment(
                experiment_name=exp_name,
                model=model,
                splits=logo_splits,
                clip_transform=clip_transform,
                forensic_transform=forensic_transform,
                device=device,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
                git_commit=resolved_git_commit,
                dataset_revision=full_dataset_rev,
                variant=v,
                clip_model=args.clip_model,
                learning_rate=args.learning_rate,
            )
            all_experiment_reports[exp_name] = report
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # PARADIGM 2: ALL-IN-ONE MULTI-GENERATOR (Upper Bound)
    # --------------------------------------------------------------------------
    if args.experiment in ("all", "all_in_one"):
        print("\n" + "#" * 80)
        print("# PARADIGM 2: ALL-IN-ONE MULTI-GENERATOR (Upper Bound on All 7 Generators)")
        print("#" * 80)

        aio_splits = build_all_in_one_splits(all_records, n_val_per_gen=100, seed=seed)
        aio_variant = "fusion" if "fusion" in variants_to_run else variants_to_run[0]
        aio_exp_name = f"all_in_one_{aio_variant}"
        set_seed(seed)
        aio_model = build_detector(variant=aio_variant, model_name=args.clip_model, hidden_dim=128, dropout=0.2)
        aio_report = train_and_eval_experiment(
            experiment_name=aio_exp_name,
            model=aio_model,
            splits=aio_splits,
            clip_transform=clip_transform,
            forensic_transform=forensic_transform,
            device=device,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            git_commit=resolved_git_commit,
            dataset_revision=full_dataset_rev,
            variant=aio_variant,
            clip_model=args.clip_model,
            learning_rate=args.learning_rate,
        )
        all_experiment_reports[aio_exp_name] = aio_report
        del aio_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # PACKAGING & BACKWARDS COMPATIBILITY ARTIFACTS
    # --------------------------------------------------------------------------
    results_dir = output_dir / "results"
    zip_path = output_dir / "results_r2.zip"
    if results_dir.exists():
        zip_results(results_dir, zip_path)

    # Save summary eval_report.json
    eval_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolved_git_commit,
        "dataset_revision": full_dataset_rev,
        "seed": seed,
        "device": str(device),
        "experiments": all_experiment_reports,
    }
    with open(output_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Backwards compatibility legacy files
    for exp_name, rep in all_experiment_reports.items():
        if "logo" in exp_name and "fusion" in exp_name:
            with open(output_dir / "logo_report.json", "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
        if "all_in_one" in exp_name:
            with open(output_dir / "all_in_one_report.json", "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
            ckpt = output_dir / "results" / "r2" / exp_name / f"seed_{seed}" / "checkpoint.pt"
            if ckpt.exists():
                import shutil
                shutil.copyfile(ckpt, output_dir / "best_model.pt")

    # --------------------------------------------------------------------------
    # PRINT FINAL COMPARATIVE BENCHMARK SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 108)
    print("                     FINAL R2 BENCHMARK EVALUATION SUMMARY                     ")
    print("=" * 108)
    for exp_name, rep in all_experiment_reports.items():
        print(f"\n--- EXPERIMENT: {exp_name.upper()} ---")
        overall = rep["evaluation"]["overall"]
        mean_cross = rep["evaluation"].get("mean_cross_generator_auroc")
        mean_cross_str = f"{mean_cross:.4f}" if mean_cross is not None else "N/A"
        print(f"  Overall: AUROC: {overall.get('auroc') or 0.0:.4f} | Accuracy: {overall.get('accuracy', 0.0):.4f} | F1: {overall.get('f1', 0.0):.4f} | Mean Cross-Gen AUROC: {mean_cross_str} (N={overall.get('total_samples', 0)})")
        for s, m in rep["results"].items():
            print(f"  Split: {s:<26} | AUROC: {m['auroc'] or 0.0:.4f} | Accuracy: {m['accuracy']:.4f} | F1: {m['f1']:.4f} (N={m['total_samples']})")
        by_gen = rep["evaluation"].get("by_generator", {})
        if by_gen:
            print("  Per-Generator Breakdown:")
            for g, gm in sorted(by_gen.items()):
                print(f"    - {g:<24} | AUROC: {gm.get('auroc') or 0.0:.4f} | Accuracy: {gm.get('accuracy', 0.0):.4f} | F1: {gm.get('f1', 0.0):.4f} (N={gm.get('total_samples', 0)})")
    print("=" * 108)

    print("\nExecution complete! Strict Artifact Contract runs and results_r2.zip are ready in /kaggle/working/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
