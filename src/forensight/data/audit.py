"""Leakage, bias, and distribution audit utilities for ForenSight dataset manifests.

This module implements Task 0.3 of R0:
- Image attribute extraction (dimensions, aspect ratio, format, file size, JPEG quality, SHA256, dHash).
- Exact duplicate detection (SHA256 collisions, cross-label, cross-split).
- Near-duplicate detection using perceptual difference hashing (dHash).
- Generator leakage and sample collision checks between train and evaluation partitions.
- Distribution quantification for GenImage literature confounds:
  * JPEG compression quality disparity between real and fake.
  * Resolution and aspect ratio disparity between real and fake.
- Format and class balance quantification.
- Machine-readable JSON serialization and comprehensive Markdown audit report generation.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image

from forensight.data.split import (
    NON_GENERATOR_LABELS,
    Manifest,
    ManifestRecord,
    validate_no_leakage,
)

# Standard JPEG luminance quantization table (Annex K of ITU-T T.81 / ISO 10918-1)
STD_LUMINANCE_QUANT_TBL = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    """Estimate JPEG quality factor (1-100) from image luminance quantization table.

    Uses the standard IJG (Independent JPEG Group) / ImageMagick formula:
    - Measures scaling factor relative to the baseline ITU-T T.81 luminance quantization table.
    - If average scale <= 100: quality = (200 - scale) / 2
    - If average scale > 100: quality = 5000 / scale
    - Clamped to integer [1, 100].

    Returns None if the image is not in JPEG format or has no quantization tables.
    """
    if getattr(img, "format", None) != "JPEG":
        return None
    quant = getattr(img, "quantization", None)
    if not quant or 0 not in quant:
        return None
    tbl = quant[0]
    if len(tbl) < 64:
        return None

    diffs = [100.0 * float(tbl[i]) / float(STD_LUMINANCE_QUANT_TBL[i]) for i in range(64)]
    avg_scale = sum(diffs) / len(diffs)
    if avg_scale <= 0.0:
        return 100
    if avg_scale <= 100.0:
        q = (200.0 - avg_scale) / 2.0
    else:
        q = 5000.0 / avg_scale
    return int(round(max(1.0, min(100.0, q))))


def compute_file_sha256(path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA256 hex digest for an image file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute 64-bit difference hash (dHash) as a 16-character hex string.

    Resizes image to (hash_size + 1, hash_size) in grayscale, compares adjacent
    horizontal pixels, and returns a 16-character hexadecimal representation.
    """
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(resized.tobytes())
    diff = []
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            p_left = pixels[row_offset + col]
            p_right = pixels[row_offset + col + 1]
            diff.append("1" if p_left > p_right else "0")
    val = int("".join(diff), 2)
    return f"{val:016x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance (bit differences) between two hex hash strings."""
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def detect_watermark_shortcuts(img: Image.Image) -> list[str]:
    """Lightweight heuristic check for watermarks, copyright text, and border shortcuts.

    Inspects:
    1. EXIF metadata and image info dictionary for known watermark / generator strings.
    2. Image borders for uniform solid letterbox bands when image interior contains texture.

    Returns:
        List of detected indicator tags (empty if clean).
    """
    indicators: list[str] = []

    # 1. Check metadata and EXIF
    meta_strings: list[str] = []
    if hasattr(img, "info") and isinstance(img.info, dict):
        for k, v in img.info.items():
            if isinstance(v, str):
                meta_strings.append(f"{k}:{v}")
            elif isinstance(v, bytes):
                try:
                    meta_strings.append(f"{k}:{v.decode('latin-1', errors='ignore')}")
                except Exception:
                    pass

    if hasattr(img, "getexif"):
        try:
            exif = img.getexif()
            if exif:
                for tag_id, val in exif.items():
                    if isinstance(val, (str, bytes)):
                        s_val = (
                            val.decode("latin-1", errors="ignore")
                            if isinstance(val, bytes)
                            else str(val)
                        )
                        meta_strings.append(f"exif_{tag_id}:{s_val}")
        except Exception:
            pass

    keywords = [
        "watermark",
        "copyright",
        "dall-e",
        "midjourney",
        "stable diffusion",
        "civitai",
        "novelai",
        "shutterstock",
        "getty",
        "photoshop",
        "stock",
    ]
    meta_blob = " ".join(meta_strings).lower()
    for kw in keywords:
        if kw in meta_blob:
            indicators.append(f"metadata:{kw}")

    # 2. Check for solid letterbox border bands (top or bottom)
    w, h = img.size
    if w >= 32 and h >= 32:
        try:
            gray = img.convert("L")
            # Interior region (20% to 80%)
            center_box = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
            center_crop = gray.crop(center_box)
            c_ext = center_crop.getextrema()
            # If interior has meaningful contrast/texture
            if c_ext and (c_ext[1] - c_ext[0]) > 20:
                band_h = max(2, int(h * 0.05))
                top_ext = gray.crop((0, 0, w, band_h)).getextrema()
                bottom_ext = gray.crop((0, h - band_h, w, h)).getextrema()
                has_top = top_ext and (top_ext[1] - top_ext[0]) <= 1
                has_bottom = bottom_ext and (bottom_ext[1] - bottom_ext[0]) <= 1
                if has_top and has_bottom:
                    indicators.append("letterbox_border")
                elif has_top:
                    indicators.append("letterbox_top_border")
                elif has_bottom:
                    indicators.append("letterbox_bottom_border")
        except Exception:
            pass

    return indicators


@dataclass
class ImageAttributes:
    """Extracted physical and forensic attributes of an image file."""

    sample_id: str
    image_path: str
    label: int
    dataset: str
    generator: str
    split: str
    class_id: str | None
    width: int
    height: int
    aspect_ratio: float
    format: str
    file_size_bytes: int
    jpeg_quality: int | None
    sha256: str
    phash: str | None = None
    watermark_indicators: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_image_attributes(
    record: ManifestRecord | dict[str, Any],
    base_dir: Path | str | None = None,
    compute_phash: bool = True,
) -> ImageAttributes:
    """Extract physical and forensic attributes from a manifest image record.

    Args:
        record: ManifestRecord instance or dictionary with manifest fields.
        base_dir: Optional base directory prefix if image_path is relative.
        compute_phash: Whether to compute perceptual dHash for near-duplicate checks.

    Returns:
        ImageAttributes containing dimension, format, quality, and hash properties.
    """
    if isinstance(record, ManifestRecord):
        rec_dict = record.to_dict()
    else:
        rec_dict = dict(record)

    raw_path = rec_dict.get("image_path", "")
    p = Path(raw_path)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p

    sample_id = str(rec_dict.get("sample_id", ""))
    label = int(rec_dict.get("label", 0))
    dataset = str(rec_dict.get("dataset", ""))
    generator = str(rec_dict.get("generator", ""))
    split = str(rec_dict.get("split", ""))
    class_id = rec_dict.get("class_id")

    if not p.exists() or not p.is_file():
        return ImageAttributes(
            sample_id=sample_id,
            image_path=str(p),
            label=label,
            dataset=dataset,
            generator=generator,
            split=split,
            class_id=class_id,
            width=0,
            height=0,
            aspect_ratio=0.0,
            format="UNKNOWN",
            file_size_bytes=0,
            jpeg_quality=None,
            sha256="",
            phash=None,
            error=f"File not found: {p}",
        )

    try:
        file_size_bytes = p.stat().st_size
        sha256_hash = compute_file_sha256(p)

        with Image.open(p) as img:
            width, height = img.size
            aspect_ratio = round(width / height, 4) if height > 0 else 0.0
            fmt = (img.format or "UNKNOWN").upper()
            jpeg_quality = estimate_jpeg_quality(img)
            phash_val = compute_dhash(img) if compute_phash else None
            watermark_indicators = detect_watermark_shortcuts(img)

        return ImageAttributes(
            sample_id=sample_id,
            image_path=str(p),
            label=label,
            dataset=dataset,
            generator=generator,
            split=split,
            class_id=class_id,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            format=fmt,
            file_size_bytes=file_size_bytes,
            jpeg_quality=jpeg_quality,
            sha256=sha256_hash,
            phash=phash_val,
            watermark_indicators=watermark_indicators,
            error=None,
        )
    except Exception as e:
        return ImageAttributes(
            sample_id=sample_id,
            image_path=str(p),
            label=label,
            dataset=dataset,
            generator=generator,
            split=split,
            class_id=class_id,
            width=0,
            height=0,
            aspect_ratio=0.0,
            format="CORRUPT",
            file_size_bytes=0,
            jpeg_quality=None,
            sha256="",
            phash=None,
            error=f"Error reading image: {e}",
        )


def calculate_distribution_stats(values: Sequence[float | int]) -> dict[str, Any]:
    """Calculate descriptive summary statistics for a sequence of numbers."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "p25": None,
            "p75": None,
            "iqr": None,
        }

    arr = np.asarray(values, dtype=float)
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr, ddof=1 if len(arr) > 1 else 0)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p25": round(p25, 2),
        "p75": round(p75, 2),
        "iqr": round(p75 - p25, 2),
    }


def calculate_jpeg_quality_buckets(qualities: Sequence[int]) -> dict[str, int]:
    """Bucket JPEG quality factors into standard bins."""
    buckets = {
        "<50": 0,
        "50-70": 0,
        "71-80": 0,
        "81-90": 0,
        "91-95": 0,
        "96-100": 0,
    }
    for q in qualities:
        if q < 50:
            buckets["<50"] += 1
        elif q <= 70:
            buckets["50-70"] += 1
        elif q <= 80:
            buckets["71-80"] += 1
        elif q <= 90:
            buckets["81-90"] += 1
        elif q <= 95:
            buckets["91-95"] += 1
        else:
            buckets["96-100"] += 1
    return buckets


def calculate_resolution_counts(attrs: Sequence[ImageAttributes]) -> list[dict[str, Any]]:
    """Determine top resolution occurrences and proportions."""
    counts: dict[str, int] = {}
    for a in attrs:
        if a.width > 0 and a.height > 0:
            res = f"{a.width}x{a.height}"
            counts[res] = counts.get(res, 0) + 1
    total = len(attrs) if attrs else 1
    sorted_res = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [
        {"resolution": res, "count": cnt, "proportion": round(cnt / total, 4)}
        for res, cnt in sorted_res[:10]
    ]


def calculate_format_stats(attrs: Sequence[ImageAttributes]) -> dict[str, Any]:
    """Calculate format counts and proportions overall and grouped by label."""
    reals = [a for a in attrs if a.label == 0]
    fakes = [a for a in attrs if a.label == 1]

    def _get_counts(group: Sequence[ImageAttributes]) -> dict[str, dict[str, Any]]:
        counts: dict[str, int] = {}
        for a in group:
            counts[a.format] = counts.get(a.format, 0) + 1
        total = len(group) if group else 1
        return {
            fmt: {"count": cnt, "proportion": round(cnt / total, 4)}
            for fmt, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        }

    return {
        "real": _get_counts(reals),
        "fake": _get_counts(fakes),
        "overall": _get_counts(attrs),
    }


def calculate_class_balance(attrs: Sequence[ManifestRecord | ImageAttributes]) -> dict[str, Any]:
    """Calculate real and fake sample distribution across semantic classes."""
    classes: dict[str, dict[str, int]] = {}
    for a in attrs:
        cid = a.class_id or "__unassigned__"
        if cid not in classes:
            classes[cid] = {"real": 0, "fake": 0, "total": 0}
        if a.label == 0:
            classes[cid]["real"] += 1
        elif a.label == 1:
            classes[cid]["fake"] += 1
        classes[cid]["total"] += 1

    by_class: dict[str, dict[str, Any]] = {}
    for cid, counts in sorted(classes.items()):
        ratio = (
            round(counts["real"] / counts["fake"], 3)
            if counts["fake"] > 0
            else None
        )
        by_class[cid] = {
            "real": counts["real"],
            "fake": counts["fake"],
            "total": counts["total"],
            "ratio_real_to_fake": ratio,
        }

    reals_per_class = [c["real"] for c in classes.values()]
    fakes_per_class = [c["fake"] for c in classes.values()]
    missing_real = [cid for cid, c in classes.items() if c["real"] == 0]
    missing_fake = [cid for cid, c in classes.items() if c["fake"] == 0]

    return {
        "num_classes": len(classes),
        "min_per_class_real": min(reals_per_class) if reals_per_class else 0,
        "max_per_class_real": max(reals_per_class) if reals_per_class else 0,
        "min_per_class_fake": min(fakes_per_class) if fakes_per_class else 0,
        "max_per_class_fake": max(fakes_per_class) if fakes_per_class else 0,
        "classes_missing_real": missing_real,
        "classes_missing_fake": missing_fake,
        "by_class": by_class,
    }


def calculate_source_balance(
    attrs: Sequence[ManifestRecord | ImageAttributes],
) -> dict[str, Any]:
    """Calculate sample distribution and correlation across dataset sources per label.

    Identifies dataset/source confounds where real and fake images originate from
    disjoint or heavily skewed data sources (e.g., ImageNet vs GenImage).
    """
    sources: dict[str, dict[str, int]] = {}
    for a in attrs:
        src = getattr(a, "dataset", None) or "__unassigned__"
        if src not in sources:
            sources[src] = {"real": 0, "fake": 0, "total": 0}
        if a.label == 0:
            sources[src]["real"] += 1
        elif a.label == 1:
            sources[src]["fake"] += 1
        sources[src]["total"] += 1

    total_real = sum(s["real"] for s in sources.values())
    total_fake = sum(s["fake"] for s in sources.values())

    by_source: dict[str, dict[str, Any]] = {}
    max_disparity = 0.0
    for src, counts in sorted(sources.items()):
        p_real = round(counts["real"] / total_real, 4) if total_real > 0 else 0.0
        p_fake = round(counts["fake"] / total_fake, 4) if total_fake > 0 else 0.0
        disparity = abs(p_real - p_fake)
        max_disparity = max(max_disparity, disparity)
        by_source[src] = {
            "real": counts["real"],
            "fake": counts["fake"],
            "total": counts["total"],
            "prop_of_reals": p_real,
            "prop_of_fakes": p_fake,
            "disparity": round(disparity, 4),
        }

    # Severe correlation occurs when real and fake originate from disjoint sources
    # or the proportion disparity between real and fake across sources exceeds 50%.
    is_strongly_correlated = False
    if total_real > 0 and total_fake > 0 and len(sources) > 1:
        if max_disparity >= 0.50:
            is_strongly_correlated = True

    return {
        "num_sources": len(sources),
        "total_real": total_real,
        "total_fake": total_fake,
        "max_source_disparity": round(max_disparity, 4),
        "is_strongly_correlated": is_strongly_correlated,
        "by_source": by_source,
    }


@dataclass
class AuditReport:
    """Comprehensive dataset audit report containing distribution metrics, leakage flags, and bias findings."""

    manifest_name: str
    total_samples: int
    num_real: int
    num_fake: int
    duplicate_check: list[dict[str, Any]]
    near_duplicate_check: list[dict[str, Any]]
    generator_leakage: list[dict[str, Any]]
    sample_leakage: list[dict[str, Any]]
    resolution_stats: dict[str, Any]
    format_stats: dict[str, Any]
    compression_stats: dict[str, Any]
    class_balance: dict[str, Any]
    source_balance: dict[str, Any] = field(default_factory=dict)
    watermark_stats: dict[str, Any] = field(default_factory=dict)
    summary_findings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize audit report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path) -> None:
        """Save report to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(self.to_json())

    def generate_markdown(self) -> str:
        """Render report as human-readable Markdown documentation."""
        return generate_audit_markdown(self)

    def save_markdown(self, path: str | Path) -> None:
        """Save report as Markdown file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(self.generate_markdown())

    def passed(self) -> bool:
        """Return True if no critical failure conditions were triggered."""
        return not any(f.get("status") == "FAIL" for f in self.summary_findings)

    def has_critical_findings(self) -> bool:
        """Return True if any finding has CRITICAL severity."""
        return any(f.get("severity") == "CRITICAL" for f in self.summary_findings)


def audit_manifest_leakage(
    train_manifest: Manifest | Sequence[ManifestRecord],
    eval_manifests: (
        Manifest
        | Sequence[ManifestRecord]
        | dict[str, Manifest | Sequence[ManifestRecord]]
        | list[Manifest]
    ),
    check_generators: bool = True,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect manifests for sample ID collisions, image path collisions, and generator leakage.

    Args:
        train_manifest: Training manifest.
        eval_manifests: Evaluation manifest(s) as a single Manifest, list of Manifests,
            or dict mapping split names to Manifests.
        check_generators: If True, checks that fake generator sets do not overlap.
        base_dir: Optional base directory prefix to resolve relative image paths.

    Returns:
        Dictionary detailing detected collisions and generator leakage.
    """
    train_m = (
        train_manifest
        if isinstance(train_manifest, Manifest)
        else Manifest(list(train_manifest))
    )

    if isinstance(eval_manifests, Manifest):
        named_evals = {"eval": eval_manifests}
    elif isinstance(eval_manifests, dict):
        named_evals = {
            k: v if isinstance(v, Manifest) else Manifest(list(v))
            for k, v in eval_manifests.items()
        }
    elif isinstance(eval_manifests, (list, tuple)):
        named_evals = {}
        for idx, item in enumerate(eval_manifests):
            m_inst = item if isinstance(item, Manifest) else Manifest(list(item))
            split_label = m_inst[0].split if len(m_inst) > 0 else f"eval_{idx}"
            named_evals[split_label] = m_inst
    else:
        raise TypeError(f"Unsupported eval_manifests type: {type(eval_manifests)}")

    def _resolve_path(raw_path: str) -> Path:
        p = Path(raw_path)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        return p.resolve()

    train_ids = {r.sample_id for r in train_m}
    train_paths = {_resolve_path(r.image_path) for r in train_m}
    train_fake_gens = {
        r.generator.lower().strip()
        for r in train_m
        if r.label == 1 and r.generator.lower().strip() not in NON_GENERATOR_LABELS
    }

    sample_id_collisions: list[dict[str, Any]] = []
    image_path_collisions: list[dict[str, Any]] = []
    generator_overlaps: list[dict[str, Any]] = []
    violations: list[str] = []

    for split_name, eval_m in named_evals.items():
        eval_ids = {r.sample_id for r in eval_m}
        id_overlap = train_ids & eval_ids
        if id_overlap:
            sample_id_collisions.append({
                "eval_split": split_name,
                "count": len(id_overlap),
                "colliding_sample_ids": sorted(list(id_overlap))[:10],
            })
            violations.append(
                f"Sample ID leakage between train and {split_name}: {len(id_overlap)} shared IDs."
            )

        eval_paths = {_resolve_path(r.image_path) for r in eval_m}
        path_overlap = train_paths & eval_paths
        if path_overlap:
            image_path_collisions.append({
                "eval_split": split_name,
                "count": len(path_overlap),
                "colliding_paths": [str(p) for p in sorted(list(path_overlap))[:10]],
            })
            violations.append(
                f"Image path leakage between train and {split_name}: {len(path_overlap)} shared paths."
            )

        if check_generators:
            eval_fake_gens = {
                r.generator.lower().strip()
                for r in eval_m
                if r.label == 1 and r.generator.lower().strip() not in NON_GENERATOR_LABELS
            }
            gen_overlap = train_fake_gens & eval_fake_gens
            test_split_tags = {r.split.lower().strip() for r in eval_m}
            is_ood_split = (
                any(
                    s in ("cross_generator_ood", "cross_generator_test", "near_ood")
                    or "ood" in s
                    or "cross_generator" in s
                    for s in test_split_tags
                )
                or "ood" in split_name.lower()
                or "cross_gen" in split_name.lower()
            )
            if gen_overlap and is_ood_split:
                generator_overlaps.append({
                    "eval_split": split_name,
                    "train_generators": sorted(list(train_fake_gens)),
                    "eval_generators": sorted(list(eval_fake_gens)),
                    "overlapping_generators": sorted(list(gen_overlap)),
                })
                violations.append(
                    f"Generator leakage between train and {split_name}: overlapping fake generator(s) {sorted(gen_overlap)}."
                )

    return {
        "has_leakage": len(violations) > 0,
        "sample_id_collisions": sample_id_collisions,
        "image_path_collisions": image_path_collisions,
        "generator_overlaps": generator_overlaps,
        "violations": violations,
    }


def audit_manifest_images(
    manifest: Manifest | Sequence[ManifestRecord],
    base_dir: Path | str | None = None,
    manifest_name: str = "dataset_audit",
    max_workers: int = 4,
    compute_phash: bool = True,
    near_duplicate_threshold: int = 2,
    max_near_duplicate_candidates: int = 2500,
    leakage_check_results: dict[str, Any] | None = None,
) -> AuditReport:
    """Run comprehensive image inspection and bias quantification on a manifest.

    Args:
        manifest: Manifest or sequence of ManifestRecords to audit.
        base_dir: Optional base directory prefix for relative image paths.
        manifest_name: Name identifier for the audit report.
        max_workers: Number of concurrent threads for image attribute extraction.
        compute_phash: Whether to compute perceptual dHash for near-duplicate analysis.
        near_duplicate_threshold: Hamming distance threshold for flagging near duplicates.
        max_near_duplicate_candidates: Maximum candidate images for pairwise dHash comparison.
        leakage_check_results: Optional leakage check results from audit_manifest_leakage.

    Returns:
        AuditReport containing distribution statistics, duplicate groups, and bias findings.
    """
    records = list(manifest)
    total_samples = len(records)
    num_real = sum(1 for r in records if r.label == 0)
    num_fake = sum(1 for r in records if r.label == 1)

    # 1. Parallel image attribute extraction
    attrs: list[ImageAttributes] = []
    if records:
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        extract_image_attributes,
                        record=rec,
                        base_dir=base_dir,
                        compute_phash=compute_phash,
                    )
                    for rec in records
                ]
                for f in concurrent.futures.as_completed(futures):
                    attrs.append(f.result())
        else:
            for rec in records:
                attrs.append(
                    extract_image_attributes(
                        record=rec,
                        base_dir=base_dir,
                        compute_phash=compute_phash,
                    )
                )

    # Maintain deterministic ordering by sample_id
    attrs.sort(key=lambda a: a.sample_id)

    # Separate real and fake attributes
    real_attrs = [a for a in attrs if a.label == 0 and a.error is None]
    fake_attrs = [a for a in attrs if a.label == 1 and a.error is None]
    valid_attrs = [a for a in attrs if a.error is None]

    # 2. Exact duplicate check (SHA256 collisions)
    sha_groups: dict[str, list[ImageAttributes]] = {}
    for a in valid_attrs:
        if a.sha256:
            sha_groups.setdefault(a.sha256, []).append(a)

    duplicate_check: list[dict[str, Any]] = []
    for sha, group in sorted(sha_groups.items()):
        if len(group) > 1:
            labels = [g.label for g in group]
            splits = [g.split for g in group]
            duplicate_check.append({
                "sha256": sha,
                "count": len(group),
                "sample_ids": [g.sample_id for g in group],
                "image_paths": [g.image_path for g in group],
                "labels": labels,
                "splits": splits,
                "cross_label": len(set(labels)) > 1,
                "cross_split": len(set(splits)) > 1,
            })

    # 3. Near-duplicate check (dHash Hamming distance)
    near_duplicate_check: list[dict[str, Any]] = []
    near_dup_skipped = False
    if compute_phash:
        phash_candidates = [a for a in valid_attrs if a.phash is not None]
        if len(phash_candidates) <= max_near_duplicate_candidates:
            for i in range(len(phash_candidates)):
                a1 = phash_candidates[i]
                h1 = int(a1.phash, 16)
                for j in range(i + 1, len(phash_candidates)):
                    a2 = phash_candidates[j]
                    h2 = int(a2.phash, 16)
                    dist = (h1 ^ h2).bit_count()
                    if dist <= near_duplicate_threshold:
                        if a1.sha256 and a1.sha256 == a2.sha256:
                            continue
                        near_duplicate_check.append({
                            "sample_ids": [a1.sample_id, a2.sample_id],
                            "image_paths": [a1.image_path, a2.image_path],
                            "distance": dist,
                            "labels": [a1.label, a2.label],
                            "splits": [a1.split, a2.split],
                            "cross_label": a1.label != a2.label,
                            "cross_split": a1.split != a2.split,
                        })
        else:
            near_dup_skipped = True

    # 4. Resolution statistics (literature confound: GenImage resolution disparity)
    resolution_stats = {
        "real": {
            "width": calculate_distribution_stats([a.width for a in real_attrs]),
            "height": calculate_distribution_stats([a.height for a in real_attrs]),
            "aspect_ratio": calculate_distribution_stats([a.aspect_ratio for a in real_attrs]),
            "top_resolutions": calculate_resolution_counts(real_attrs),
        },
        "fake": {
            "width": calculate_distribution_stats([a.width for a in fake_attrs]),
            "height": calculate_distribution_stats([a.height for a in fake_attrs]),
            "aspect_ratio": calculate_distribution_stats([a.aspect_ratio for a in fake_attrs]),
            "top_resolutions": calculate_resolution_counts(fake_attrs),
        },
        "overall": {
            "width": calculate_distribution_stats([a.width for a in valid_attrs]),
            "height": calculate_distribution_stats([a.height for a in valid_attrs]),
            "aspect_ratio": calculate_distribution_stats([a.aspect_ratio for a in valid_attrs]),
            "top_resolutions": calculate_resolution_counts(valid_attrs),
        },
    }

    # 5. Format statistics
    format_stats = calculate_format_stats(valid_attrs)

    # 6. Compression & JPEG Quality statistics (literature confound: GenImage compression disparity)
    real_qualities = [a.jpeg_quality for a in real_attrs if a.jpeg_quality is not None]
    fake_qualities = [a.jpeg_quality for a in fake_attrs if a.jpeg_quality is not None]
    valid_qualities = [a.jpeg_quality for a in valid_attrs if a.jpeg_quality is not None]

    compression_stats = {
        "real": {
            "file_size_bytes": calculate_distribution_stats([a.file_size_bytes for a in real_attrs]),
            "jpeg_quality": calculate_distribution_stats(real_qualities),
            "jpeg_quality_buckets": calculate_jpeg_quality_buckets(real_qualities),
            "missing_jpeg_quality_count": len(real_attrs) - len(real_qualities),
        },
        "fake": {
            "file_size_bytes": calculate_distribution_stats([a.file_size_bytes for a in fake_attrs]),
            "jpeg_quality": calculate_distribution_stats(fake_qualities),
            "jpeg_quality_buckets": calculate_jpeg_quality_buckets(fake_qualities),
            "missing_jpeg_quality_count": len(fake_attrs) - len(fake_qualities),
        },
        "overall": {
            "file_size_bytes": calculate_distribution_stats([a.file_size_bytes for a in valid_attrs]),
            "jpeg_quality": calculate_distribution_stats(valid_qualities),
            "jpeg_quality_buckets": calculate_jpeg_quality_buckets(valid_qualities),
            "missing_jpeg_quality_count": len(valid_attrs) - len(valid_qualities),
        },
    }

    # 7. Class balance
    class_balance = calculate_class_balance(records)

    # 8. Source balance & correlation
    source_balance = calculate_source_balance(records)

    # 9. Watermark statistics & shortcuts
    real_wm = [a for a in real_attrs if a.watermark_indicators]
    fake_wm = [a for a in fake_attrs if a.watermark_indicators]
    valid_wm = [a for a in valid_attrs if a.watermark_indicators]

    watermark_stats = {
        "total_flagged": len(valid_wm),
        "real_flagged": len(real_wm),
        "fake_flagged": len(fake_wm),
        "sample_indicators": [
            {"sample_id": a.sample_id, "label": a.label, "indicators": a.watermark_indicators}
            for a in valid_wm[:10]
        ],
    }

    # 10. Leakage summaries
    generator_leakage: list[dict[str, Any]] = []
    sample_leakage: list[dict[str, Any]] = []
    if leakage_check_results:
        generator_leakage = leakage_check_results.get("generator_overlaps", [])
        sample_leakage = (
            leakage_check_results.get("sample_id_collisions", [])
            + leakage_check_results.get("image_path_collisions", [])
        )

    # 11. Formulate summary findings & flags
    summary_findings: list[dict[str, Any]] = []

    # Finding 1: Exact Duplicates
    if not duplicate_check:
        summary_findings.append({
            "check": "exact_duplicates",
            "status": "PASS",
            "severity": "INFO",
            "message": "Zero exact duplicate images detected (SHA256 collisions = 0).",
        })
    else:
        cross_label = [d for d in duplicate_check if d.get("cross_label")]
        cross_split = [d for d in duplicate_check if d.get("cross_split")]
        if cross_label:
            summary_findings.append({
                "check": "exact_duplicates",
                "status": "FAIL",
                "severity": "CRITICAL",
                "message": (
                    f"Critical label contradiction: {len(cross_label)} duplicate group(s) have "
                    "identical byte hashes labeled as both real and fake."
                ),
                "details": {"cross_label_groups": cross_label},
            })
        elif cross_split:
            summary_findings.append({
                "check": "exact_duplicates",
                "status": "FAIL",
                "severity": "CRITICAL",
                "message": (
                    f"Cross-split duplicate leakage: {len(cross_split)} duplicate group(s) "
                    "shared across partitions."
                ),
                "details": {"cross_split_groups": cross_split},
            })
        else:
            summary_findings.append({
                "check": "exact_duplicates",
                "status": "WARNING",
                "severity": "WARNING",
                "message": f"Found {len(duplicate_check)} duplicate group(s) within the same split and label.",
                "details": {"duplicate_count": len(duplicate_check)},
            })

    # Finding 2: Near Duplicates
    if not compute_phash:
        summary_findings.append({
            "check": "near_duplicates",
            "status": "SKIPPED",
            "severity": "INFO",
            "message": "Near-duplicate check disabled (compute_phash=False).",
        })
    elif near_dup_skipped:
        summary_findings.append({
            "check": "near_duplicates",
            "status": "SKIPPED",
            "severity": "INFO",
            "message": (
                f"Near-duplicate check skipped: sample size ({len(phash_candidates)}) "
                f"exceeds pairwise limit ({max_near_duplicate_candidates}). "
                "Use subsampling or indexed search."
            ),
            "details": {
                "candidate_count": len(phash_candidates),
                "limit": max_near_duplicate_candidates,
            },
        })
    elif not near_duplicate_check:
        summary_findings.append({
            "check": "near_duplicates",
            "status": "PASS",
            "severity": "INFO",
            "message": f"Zero near-duplicate image pairs detected (Hamming distance <= {near_duplicate_threshold}).",
        })
    else:
        cross_split_near = [d for d in near_duplicate_check if d.get("cross_split")]
        if cross_split_near:
            summary_findings.append({
                "check": "near_duplicates",
                "status": "WARNING",
                "severity": "WARNING",
                "message": f"Detected {len(cross_split_near)} near-duplicate pair(s) across splits.",
                "details": {"near_duplicate_count": len(cross_split_near)},
            })
        else:
            summary_findings.append({
                "check": "near_duplicates",
                "status": "INFO",
                "severity": "INFO",
                "message": f"Detected {len(near_duplicate_check)} near-duplicate pair(s) within partitions.",
                "details": {"near_duplicate_count": len(near_duplicate_check)},
            })

    # Finding 3: Generator Leakage
    if not generator_leakage:
        summary_findings.append({
            "check": "generator_leakage",
            "status": "PASS",
            "severity": "INFO",
            "message": "No generator leakage detected between train and evaluation partitions.",
        })
    else:
        summary_findings.append({
            "check": "generator_leakage",
            "status": "FAIL",
            "severity": "CRITICAL",
            "message": f"Generator leakage detected: {generator_leakage}",
            "details": {"leaks": generator_leakage},
        })

    # Finding 4: Sample ID and Path Leakage
    if not sample_leakage:
        summary_findings.append({
            "check": "sample_leakage",
            "status": "PASS",
            "severity": "INFO",
            "message": "No sample ID or image path collisions detected between splits.",
        })
    else:
        summary_findings.append({
            "check": "sample_leakage",
            "status": "FAIL",
            "severity": "CRITICAL",
            "message": f"Cross-split sample collision detected: {len(sample_leakage)} collision(s).",
            "details": {"collisions": sample_leakage},
        })

    # Finding 5: Resolution Disparity (GenImage literature confound)
    real_w_median = resolution_stats["real"]["width"]["median"]
    fake_w_median = resolution_stats["fake"]["width"]["median"]
    real_w_std = resolution_stats["real"]["width"]["std"]
    fake_w_std = resolution_stats["fake"]["width"]["std"]

    if real_w_median is not None and fake_w_median is not None:
        w_diff = abs(real_w_median - fake_w_median) / max(real_w_median, fake_w_median, 1.0)
        is_fixed_fake_vs_variable_real = (fake_w_std == 0.0 and (real_w_std or 0.0) > 10.0)
        if w_diff > 0.10 or is_fixed_fake_vs_variable_real:
            summary_findings.append({
                "check": "resolution_disparity",
                "status": "WARNING",
                "severity": "WARNING",
                "message": (
                    f"Resolution confound detected (literature confound): Real median width ({real_w_median}px, std={real_w_std}) "
                    f"differs substantially from fake median width ({fake_w_median}px, std={fake_w_std}). "
                    "Detectors risk exploiting dimension shortcuts."
                ),
                "details": {
                    "real_median": real_w_median,
                    "fake_median": fake_w_median,
                    "real_std": real_w_std,
                    "fake_std": fake_w_std,
                },
            })
        else:
            summary_findings.append({
                "check": "resolution_disparity",
                "status": "PASS",
                "severity": "INFO",
                "message": f"Real and fake resolution distributions are reasonably aligned (median width gap: {w_diff:.1%}).",
            })

    # Finding 6: JPEG Quality Disparity (GenImage literature confound)
    real_q_mean = compression_stats["real"]["jpeg_quality"]["mean"]
    fake_q_mean = compression_stats["fake"]["jpeg_quality"]["mean"]
    if real_q_mean is not None and fake_q_mean is not None:
        q_gap = abs(real_q_mean - fake_q_mean)
        if q_gap > 5.0:
            summary_findings.append({
                "check": "jpeg_quality_disparity",
                "status": "WARNING",
                "severity": "WARNING",
                "message": (
                    f"JPEG quality confound detected (literature confound): Real images (mean Q={real_q_mean}) "
                    f"and fake images (mean Q={fake_q_mean}) exhibit a quality gap of {q_gap:.1f}. "
                    "Detectors risk learning compression artifacts."
                ),
                "details": {"real_mean_q": real_q_mean, "fake_mean_q": fake_q_mean, "gap": round(q_gap, 2)},
            })
        else:
            summary_findings.append({
                "check": "jpeg_quality_disparity",
                "status": "PASS",
                "severity": "INFO",
                "message": f"Real and fake JPEG quality factors are balanced (gap: {q_gap:.1f} points).",
            })
    elif (real_q_mean is None) ^ (fake_q_mean is None):
        summary_findings.append({
            "check": "jpeg_quality_disparity",
            "status": "WARNING",
            "severity": "WARNING",
            "message": "One class has valid JPEG quality factors while the other is missing/non-JPEG.",
        })

    # Finding 7: Format Disparity
    real_fmts = set(format_stats.get("real", {}).keys())
    fake_fmts = set(format_stats.get("fake", {}).keys())
    if real_fmts and fake_fmts:
        all_fmts = real_fmts | fake_fmts
        max_prop_diff = 0.0
        for fmt in all_fmts:
            p_real = format_stats.get("real", {}).get(fmt, {}).get("proportion", 0.0)
            p_fake = format_stats.get("fake", {}).get(fmt, {}).get("proportion", 0.0)
            max_prop_diff = max(max_prop_diff, abs(p_real - p_fake))
        if max_prop_diff > 0.10:
            summary_findings.append({
                "check": "format_disparity",
                "status": "WARNING",
                "severity": "WARNING",
                "message": f"Format disparity detected: Real and fake format proportions differ by up to {max_prop_diff:.1%}.",
                "details": {"max_proportion_difference": round(max_prop_diff, 4)},
            })
        else:
            summary_findings.append({
                "check": "format_disparity",
                "status": "PASS",
                "severity": "INFO",
                "message": "Real and fake format proportions are well balanced.",
            })

    # Finding 8: Class Balance
    if class_balance.get("classes_missing_real") or class_balance.get("classes_missing_fake"):
        summary_findings.append({
            "check": "class_balance",
            "status": "WARNING",
            "severity": "WARNING",
            "message": (
                f"Class balance warning: {len(class_balance.get('classes_missing_real', []))} class(es) missing real samples, "
                f"{len(class_balance.get('classes_missing_fake', []))} class(es) missing fake samples."
            ),
            "details": {
                "classes_missing_real": class_balance.get("classes_missing_real"),
                "classes_missing_fake": class_balance.get("classes_missing_fake"),
            },
        })
    else:
        summary_findings.append({
            "check": "class_balance",
            "status": "PASS",
            "severity": "INFO",
            "message": f"All {class_balance.get('num_classes', 0)} classes have balanced real and fake representation.",
        })

    # Finding 9: Source-Label Correlation
    if source_balance.get("is_strongly_correlated"):
        summary_findings.append({
            "check": "source_label_correlation",
            "status": "WARNING",
            "severity": "WARNING",
            "message": (
                f"Strong source-label correlation detected: Real and fake distributions across dataset sources "
                f"differ substantially (max disparity: {source_balance['max_source_disparity']:.1%}). "
                "Detectors risk exploiting dataset/source shortcuts."
            ),
            "details": source_balance,
        })
    else:
        summary_findings.append({
            "check": "source_label_correlation",
            "status": "PASS",
            "severity": "INFO",
            "message": "Real and fake distributions across dataset sources are balanced or single-source.",
            "details": source_balance,
        })

    # Finding 10: Watermark Shortcuts
    if not valid_wm:
        summary_findings.append({
            "check": "watermark_shortcuts",
            "status": "PASS",
            "severity": "INFO",
            "message": "No obvious watermark, metadata copyright, or letterbox shortcuts detected.",
        })
    else:
        prop_real_wm = len(real_wm) / len(real_attrs) if real_attrs else 0.0
        prop_fake_wm = len(fake_wm) / len(fake_attrs) if fake_attrs else 0.0
        wm_disparity = abs(prop_real_wm - prop_fake_wm)
        if wm_disparity > 0.05 or len(real_wm) != len(fake_wm):
            summary_findings.append({
                "check": "watermark_shortcuts",
                "status": "WARNING",
                "severity": "WARNING",
                "message": (
                    f"Watermark shortcut alert: Detected {len(valid_wm)} image(s) with watermark indicators "
                    f"(real: {len(real_wm)}, fake: {len(fake_wm)}). Detectors risk exploiting watermark shortcuts."
                ),
                "details": watermark_stats,
            })
        else:
            summary_findings.append({
                "check": "watermark_shortcuts",
                "status": "INFO",
                "severity": "INFO",
                "message": f"Detected {len(valid_wm)} image(s) with watermark indicators, evenly balanced.",
                "details": watermark_stats,
            })

    # Finding 11: File read errors
    read_errors = [a for a in attrs if a.error is not None]
    if read_errors:
        summary_findings.append({
            "check": "file_integrity",
            "status": "FAIL",
            "severity": "CRITICAL",
            "message": f"{len(read_errors)} image file(s) could not be read or were corrupt.",
            "details": {"errors": [f"{a.image_path}: {a.error}" for a in read_errors[:5]]},
        })
    else:
        summary_findings.append({
            "check": "file_integrity",
            "status": "PASS",
            "severity": "INFO",
            "message": "All manifest images were verified readable and uncorrupted.",
        })

    return AuditReport(
        manifest_name=manifest_name,
        total_samples=total_samples,
        num_real=num_real,
        num_fake=num_fake,
        duplicate_check=duplicate_check,
        near_duplicate_check=near_duplicate_check,
        generator_leakage=generator_leakage,
        sample_leakage=sample_leakage,
        resolution_stats=resolution_stats,
        format_stats=format_stats,
        compression_stats=compression_stats,
        class_balance=class_balance,
        source_balance=source_balance,
        watermark_stats=watermark_stats,
        summary_findings=summary_findings,
        metadata={
            "num_read_errors": len(read_errors),
            "near_duplicate_threshold": near_duplicate_threshold,
            "near_duplicate_skipped": near_dup_skipped,
            "max_near_duplicate_candidates": max_near_duplicate_candidates,
        },
    )


def generate_audit_markdown(report: AuditReport) -> str:
    """Generate a clean, structured Markdown report from an AuditReport instance."""
    lines: list[str] = []

    # Overall Status determination
    if report.has_critical_findings():
        status_badge = "CRITICAL / FAILED"
    elif any(f.get("status") == "WARNING" for f in report.summary_findings):
        status_badge = "WARNINGS IDENTIFIED"
    else:
        status_badge = "PASSED"

    lines.append(f"# ForenSight Dataset Audit Report: {report.manifest_name}")
    lines.append("")
    lines.append(f"**Overall Status:** `{status_badge}`  ")
    lines.append(f"**Total Samples:** {report.total_samples} (Real: {report.num_real}, Fake: {report.num_fake})  ")
    lines.append("")

    # Protocol Guidance Alert
    lines.append("> [!IMPORTANT]")
    lines.append("> **Research Invariant (AGENTS.md & Plan R0):**")
    lines.append("> Do NOT automatically resize or re-encode raw data to 'fix' bias before benchmarking.")
    lines.append("> Raw dataset distributions must be accurately quantified. Any bias-mitigation preprocessing")
    lines.append("> must be tested as an explicit, versioned experimental hypothesis.")
    lines.append("")

    # Section 1: Executive Summary of Findings
    lines.append("## 1. Executive Summary & Findings")
    lines.append("")
    lines.append("| Check | Status | Severity | Summary Finding |")
    lines.append("| :--- | :---: | :---: | :--- |")
    for f in report.summary_findings:
        st = f.get("status", "INFO")
        sev = f.get("severity", "INFO")
        chk = f.get("check", "")
        msg = f.get("message", "")
        lines.append(f"| `{chk}` | `{st}` | {sev} | {msg} |")
    lines.append("")

    # Section 2: Duplication & Leakage Analysis
    lines.append("## 2. Integrity, Duplicates & Leakage")
    lines.append("")
    lines.append(f"- **Exact Duplicates (SHA256 Collisions):** {len(report.duplicate_check)} duplicate group(s)")
    near_status = (
        "SKIPPED (sample size exceeds pairwise limit)"
        if report.metadata.get("near_duplicate_skipped")
        else f"{len(report.near_duplicate_check)} near-duplicate pair(s)"
    )
    lines.append(f"- **Near Duplicates (dHash):** {near_status}")
    lines.append(f"- **Cross-Split Sample Collisions:** {len(report.sample_leakage)} collision(s)")
    lines.append(f"- **Generator Overlaps (Train vs Eval):** {len(report.generator_leakage)} leak(s)")
    lines.append("")

    if report.duplicate_check:
        lines.append("### Exact Duplicate Groups")
        lines.append("")
        for d in report.duplicate_check[:5]:
            lines.append(f"- **SHA256:** `{d['sha256'][:16]}...` (Count: {d['count']}, Labels: {d['labels']}, Splits: {d['splits']})")
            for p in d["image_paths"][:3]:
                lines.append(f"  * `{p}`")
        if len(report.duplicate_check) > 5:
            lines.append(f"  * *(and {len(report.duplicate_check) - 5} more duplicate groups...)*")
        lines.append("")

    # Section 3: Literature Confound 1 - Resolution Disparity
    lines.append("## 3. Resolution & Dimension Distribution (Literature Confound)")
    lines.append("")
    lines.append("Literature (Unbiased GenImage) documents that AI generators typically produce fixed dimensions (e.g. 512x512),")
    lines.append("whereas real-world datasets (ImageNet) exhibit high variance in width, height, and aspect ratio.")
    lines.append("")

    lines.append("| Partition | Width Mean±Std | Width Median (IQR) | Height Mean±Std | Height Median (IQR) | Aspect Ratio Mean |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for group in ("real", "fake", "overall"):
        w = report.resolution_stats.get(group, {}).get("width", {})
        h = report.resolution_stats.get(group, {}).get("height", {})
        ar = report.resolution_stats.get(group, {}).get("aspect_ratio", {})
        w_m_s = f"{w.get('mean')}±{w.get('std')}" if w.get("mean") is not None else "N/A"
        w_med = f"{w.get('median')} ({w.get('iqr')})" if w.get("median") is not None else "N/A"
        h_m_s = f"{h.get('mean')}±{h.get('std')}" if h.get("mean") is not None else "N/A"
        h_med = f"{h.get('median')} ({h.get('iqr')})" if h.get("median") is not None else "N/A"
        ar_mean = f"{ar.get('mean')}" if ar.get("mean") is not None else "N/A"
        lines.append(f"| **{group.capitalize()}** | {w_m_s} | {w_med} | {h_m_s} | {h_med} | {ar_mean} |")
    lines.append("")

    lines.append("### Top Occurring Resolutions")
    lines.append("")
    lines.append("| Real Top Resolutions | Fake Top Resolutions |")
    lines.append("| :--- | :--- |")
    real_res = report.resolution_stats.get("real", {}).get("top_resolutions", [])
    fake_res = report.resolution_stats.get("fake", {}).get("top_resolutions", [])
    max_res_len = max(len(real_res), len(fake_res), 1)
    for i in range(min(max_res_len, 5)):
        r_str = f"{real_res[i]['resolution']} ({real_res[i]['count']}, {real_res[i]['proportion']:.1%})" if i < len(real_res) else "-"
        f_str = f"{fake_res[i]['resolution']} ({fake_res[i]['count']}, {fake_res[i]['proportion']:.1%})" if i < len(fake_res) else "-"
        lines.append(f"| {r_str} | {f_str} |")
    lines.append("")

    # Section 4: Literature Confound 2 - Compression & JPEG Quality Disparity
    lines.append("## 4. JPEG Compression & Quality Distribution (Literature Confound)")
    lines.append("")
    lines.append("Disparity in JPEG quality factors (derived from luminance quantization tables) can serve as an artificial shortcut")
    lines.append("where detectors learn to classify quantization table profiles rather than generative semantic artifacts.")
    lines.append("")

    lines.append("| Partition | JPEG Quality Mean±Std | JPEG Quality Median | File Size Mean±Std (KB) | Missing Quality Count |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for group in ("real", "fake", "overall"):
        q = report.compression_stats.get(group, {}).get("jpeg_quality", {})
        sz = report.compression_stats.get(group, {}).get("file_size_bytes", {})
        missing = report.compression_stats.get(group, {}).get("missing_jpeg_quality_count", 0)
        q_m_s = f"{q.get('mean')}±{q.get('std')}" if q.get("mean") is not None else "N/A"
        q_med = f"{q.get('median')}" if q.get("median") is not None else "N/A"
        if sz.get("mean") is not None and sz.get("std") is not None:
            sz_m_s = f"{sz['mean']/1024:.1f}±{sz['std']/1024:.1f}"
        else:
            sz_m_s = "N/A"
        lines.append(f"| **{group.capitalize()}** | {q_m_s} | {q_med} | {sz_m_s} | {missing} |")
    lines.append("")

    # Quality Buckets Table
    lines.append("### JPEG Quality Factor Histogram Buckets")
    lines.append("")
    lines.append("| Quality Range | Real Count | Real % | Fake Count | Fake % |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    r_buckets = report.compression_stats.get("real", {}).get("jpeg_quality_buckets", {})
    f_buckets = report.compression_stats.get("fake", {}).get("jpeg_quality_buckets", {})
    r_sum = sum(r_buckets.values()) if r_buckets else 0
    f_sum = sum(f_buckets.values()) if f_buckets else 0
    for b_name in ("<50", "50-70", "71-80", "81-90", "91-95", "96-100"):
        rc = r_buckets.get(b_name, 0)
        fc = f_buckets.get(b_name, 0)
        rc_str = f"{rc/r_sum:.1%}" if r_sum > 0 else "N/A"
        fc_str = f"{fc/f_sum:.1%}" if f_sum > 0 else "N/A"
        lines.append(f"| `{b_name}` | {rc} | {rc_str} | {fc} | {fc_str} |")
    lines.append("")

    # Section 5: Format Distribution
    lines.append("## 5. File Format Distribution")
    lines.append("")
    lines.append("| Format | Real Count (Prop) | Fake Count (Prop) | Overall Count (Prop) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    fmts = set(report.format_stats.get("overall", {}).keys())
    for fmt in sorted(fmts):
        r_f = report.format_stats.get("real", {}).get(fmt, {"count": 0, "proportion": 0.0})
        f_f = report.format_stats.get("fake", {}).get(fmt, {"count": 0, "proportion": 0.0})
        o_f = report.format_stats.get("overall", {}).get(fmt, {"count": 0, "proportion": 0.0})
        lines.append(
            f"| `{fmt}` | {r_f['count']} ({r_f['proportion']:.1%}) | "
            f"{f_f['count']} ({f_f['proportion']:.1%}) | {o_f['count']} ({o_f['proportion']:.1%}) |"
        )
    lines.append("")

    # Section 6: Semantic Class Balance
    lines.append("## 6. Semantic Class Representation")
    lines.append("")
    cb = report.class_balance
    lines.append(f"- **Distinct Classes / Synsets:** {cb.get('num_classes', 0)}")
    lines.append(f"- **Real Samples per Class:** Min {cb.get('min_per_class_real', 0)}, Max {cb.get('max_per_class_real', 0)}")
    lines.append(f"- **Fake Samples per Class:** Min {cb.get('min_per_class_fake', 0)}, Max {cb.get('max_per_class_fake', 0)}")
    if cb.get("classes_missing_real"):
        lines.append(f"- **Classes Missing Real Samples:** {cb['classes_missing_real'][:5]}")
    if cb.get("classes_missing_fake"):
        lines.append(f"- **Classes Missing Fake Samples:** {cb['classes_missing_fake'][:5]}")
    lines.append("")

    # Section 7: Dataset Source Distribution & Correlation
    lines.append("## 7. Dataset Source Distribution & Correlation")
    lines.append("")
    sb = report.source_balance
    lines.append(f"- **Distinct Sources:** {sb.get('num_sources', 0)} ({', '.join(sb.get('sources', []))})")
    lines.append(f"- **Strong Source-Label Correlation:** {'YES (WARNING)' if sb.get('is_strongly_correlated') else 'NO (PASS)'}")
    lines.append(f"- **Max Source Disparity:** {sb.get('max_source_disparity', 0.0):.1%}")
    lines.append("")
    lines.append("| Source | Real Count (Prop) | Fake Count (Prop) | Overall Count (Prop) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for src in sb.get("sources", []):
        r_s = sb.get("real_distribution", {}).get(src, {"count": 0, "proportion": 0.0})
        f_s = sb.get("fake_distribution", {}).get(src, {"count": 0, "proportion": 0.0})
        o_s = sb.get("overall_distribution", {}).get(src, {"count": 0, "proportion": 0.0})
        lines.append(
            f"| `{src}` | {r_s['count']} ({r_s['proportion']:.1%}) | "
            f"{f_s['count']} ({f_s['proportion']:.1%}) | {o_s['count']} ({o_s['proportion']:.1%}) |"
        )
    lines.append("")

    # Section 8: Watermark & Shortcut Analysis
    lines.append("## 8. Watermark & Artifact Shortcut Analysis")
    lines.append("")
    wm = report.watermark_stats
    lines.append(f"- **Images with Watermark/Border Indicators:** {wm.get('total_flagged', 0)} "
                 f"(Real: {wm.get('real_flagged', 0)}, Fake: {wm.get('fake_flagged', 0)})")
    lines.append("")
    if wm.get("sample_indicators"):
        lines.append("### Flagged Sample Indicators (up to 10)")
        lines.append("")
        for item in wm["sample_indicators"]:
            lines.append(f"- **Sample:** `{item['sample_id']}` ({item['label']}): {', '.join(item['indicators'])}")
        lines.append("")

    lines.append("---")
    lines.append("*Report generated by ForenSight Dataset Audit Toolkit (Task 0.3).*")
    lines.append("")

    return "\n".join(lines)


__all__ = [
    "AuditReport",
    "ImageAttributes",
    "STD_LUMINANCE_QUANT_TBL",
    "audit_manifest_images",
    "audit_manifest_leakage",
    "calculate_class_balance",
    "calculate_distribution_stats",
    "calculate_format_stats",
    "calculate_jpeg_quality_buckets",
    "calculate_resolution_counts",
    "calculate_source_balance",
    "compute_dhash",
    "compute_file_sha256",
    "detect_watermark_shortcuts",
    "estimate_jpeg_quality",
    "extract_image_attributes",
    "generate_audit_markdown",
    "hamming_distance",
]
