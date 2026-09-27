"""Fast, lightweight synthetic dataset generation and end-to-end smoke test pipeline.

This module implements Task 0.7 of ForenSight Milestone R0:
- Deterministic synthetic dataset generator producing valid JPEG images in standard
  Stage-1 hierarchy (SD1.5 train/val and Midjourney cross-generator OOD).
- End-to-end smoke-test pipeline executing all 7 R0 stages in seconds:
  1. Inventory verification (load_inventory, validation check).
  2. Synthetic dataset & manifest generation.
  3. Leakage & generator-disjointness assertions.
  4. Dataset audit execution (audit_manifest_images, audit_manifest_leakage).
  5. Synthetic model predictions simulation across 3 distinct seeds.
  6. Evaluation runner on each seed with validation threshold calibration.
  7. Multi-seed repeated run aggregation (aggregate_reports).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from PIL import Image

from forensight.data.audit import (
    AuditReport,
    audit_manifest_images,
    audit_manifest_leakage,
)
from forensight.data.inventory import (
    DatasetInventory,
    build_default_inventory,
    load_inventory,
)
from forensight.data.split import (
    Manifest,
    ManifestRecord,
    assert_generator_disjoint,
    validate_no_leakage,
)
from forensight.evaluation.aggregate import (
    AggregatedEvaluationReport,
    aggregate_reports,
)
from forensight.evaluation.runner import (
    EvaluationReport,
    PredictionRecord,
    PredictionSet,
    evaluate_predictions,
)


DEFAULT_SYNSET_IDS: list[str] = [
    "n01440764",  # tench
    "n01443537",  # goldfish
    "n01484850",  # great white shark
    "n01491361",  # tiger shark
    "n01494475",  # hammerhead shark
    "n01514668",  # cock
    "n01514859",  # hen
    "n01518878",  # ostrich
    "n01530575",  # brambling
    "n01531178",  # goldfinch
]


def _resolve_classes(num_classes: int) -> list[str]:
    """Return a deterministic list of unique ImageNet-style synset IDs."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive integer, got: {num_classes}")
    if num_classes <= len(DEFAULT_SYNSET_IDS):
        return DEFAULT_SYNSET_IDS[:num_classes]
    classes = list(DEFAULT_SYNSET_IDS)
    seen = set(classes)
    candidate_id = 1440764 + len(DEFAULT_SYNSET_IDS)
    while len(classes) < num_classes:
        candidate = f"n{candidate_id:08d}"
        if candidate not in seen:
            seen.add(candidate)
            classes.append(candidate)
        candidate_id += 1
    return classes


def _generate_synthetic_image(
    file_path: Path,
    seed: int,
    split: str,
    class_id: str,
    label: int,
    image_size: tuple[int, int] = (64, 64),
    jpeg_quality: int = 85,
) -> None:
    """Generate and save a deterministic lightweight synthetic JPEG image with Pillow."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive deterministic seed from parameters
    seed_str = f"{seed}_{split}_{class_id}_{label}_{file_path.name}"
    seed_int = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed_int)

    # Base background color depends on label and class to provide realistic variation
    base_color = rng.randint(40, 215, size=(3,), dtype=np.uint8)
    noise = rng.randint(-30, 31, size=(image_size[0], image_size[1], 3), dtype=np.int16)
    arr = np.clip(base_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save as JPEG with valid quantization table
    img = Image.fromarray(arr, mode="RGB")
    img.save(file_path, format="JPEG", quality=jpeg_quality)


def generate_smoke_dataset(
    output_dir: Path | str,
    num_classes: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Path], dict[str, Manifest]]:
    """Generate a lightweight synthetic JPEG dataset organized in standard GenImage layouts.

    Creates standard directory layouts:
    - genimage/sdv5/train/nature, genimage/sdv5/train/ai
    - genimage/sdv5/val/nature, genimage/sdv5/val/ai
    - genimage/midjourney/val/nature, genimage/midjourney/val/ai

    Generates corresponding deterministic Manifest objects and writes JSONL manifests:
    - 'train': In-domain SD1.5 train split
    - 'val': In-domain SD1.5 validation split
    - 'cross_generator_ood': Midjourney cross-generator OOD validation split

    Args:
        output_dir: Root directory where images and manifests will be generated.
        num_classes: Number of ImageNet-style classes to generate per split (default: 5).
        seed: Random seed for deterministic image synthesis and sampling (default: 42).

    Returns:
        tuple[dict[str, Path], dict[str, Manifest]]:
            - Dictionary mapping split names to saved JSONL manifest file paths.
            - Dictionary mapping split names to Manifest container objects.
    """
    out_dir = Path(output_dir).resolve()
    classes = _resolve_classes(num_classes)

    # Split specifications: (split_name, generator_folder, subsplit, fake_generator_id)
    split_configs = [
        ("train", "sdv5", "train", "sd15"),
        ("val", "sdv5", "val", "sd15"),
        ("cross_generator_ood", "midjourney", "val", "midjourney"),
    ]

    manifests: dict[str, Manifest] = {}

    for split_name, gen_folder, subsplit, fake_gen in split_configs:
        records: list[ManifestRecord] = []
        gen_root = out_dir / "genimage" / gen_folder / subsplit

        nature_dir = gen_root / "nature"
        ai_dir = gen_root / "ai"

        nature_dir.mkdir(parents=True, exist_ok=True)
        ai_dir.mkdir(parents=True, exist_ok=True)

        for class_id in classes:
            # 1. Real image (nature)
            real_img_path = nature_dir / class_id / f"real_{class_id}.jpg"
            _generate_synthetic_image(
                file_path=real_img_path,
                seed=seed,
                split=split_name,
                class_id=class_id,
                label=0,
            )
            real_sample_id = f"genimage_{gen_folder}_{subsplit}_nature_{class_id}"
            records.append(
                ManifestRecord(
                    sample_id=real_sample_id,
                    image_path=str(real_img_path.resolve()),
                    label=0,
                    dataset="genimage",
                    generator="nature",
                    split=split_name,
                    class_id=class_id,
                    metadata={
                        "subset": gen_folder,
                        "subsplit": subsplit,
                        "role": split_name,
                        "is_synthetic_smoke": True,
                    },
                )
            )

            # 2. Fake image (ai)
            fake_img_path = ai_dir / class_id / f"fake_{class_id}.jpg"
            _generate_synthetic_image(
                file_path=fake_img_path,
                seed=seed,
                split=split_name,
                class_id=class_id,
                label=1,
            )
            fake_sample_id = f"genimage_{gen_folder}_{subsplit}_ai_{class_id}"
            records.append(
                ManifestRecord(
                    sample_id=fake_sample_id,
                    image_path=str(fake_img_path.resolve()),
                    label=1,
                    dataset="genimage",
                    generator=fake_gen,
                    split=split_name,
                    class_id=class_id,
                    metadata={
                        "subset": gen_folder,
                        "subsplit": subsplit,
                        "role": split_name,
                        "is_synthetic_smoke": True,
                    },
                )
            )

        manifests[split_name] = Manifest(records)

    # Save manifests to output_dir / manifests
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: dict[str, Path] = {}
    for split_name, manifest in manifests.items():
        m_path = manifests_dir / f"{split_name}.jsonl"
        manifest.to_jsonl(m_path)
        manifest_paths[split_name] = m_path

    return manifest_paths, manifests


def _assert_audit_leakage_clean(leakage_audit: dict[str, Any]) -> None:
    """Raise ValueError if audit_manifest_leakage reported any leakage.

    Reads the actual contract key returned by
    forensight.data.audit.audit_manifest_leakage, which is 'has_leakage'
    (with detail in 'violations'). Earlier code read non-existent keys
    ('leakage_detected'/'generator_leakage_detected'), so the guard never fired.
    """
    if leakage_audit.get("has_leakage"):
        violations = leakage_audit.get("violations") or []
        detail = "; ".join(violations) if violations else str(leakage_audit)
        raise ValueError(
            f"Dataset audit detected unexpected leakage in smoke dataset: {detail}"
        )


def _locate_or_load_inventory() -> DatasetInventory:
    """Find and load project dataset inventory, falling back to default inventory."""
    candidate_paths = [
        Path("data/dataset_inventory.json"),
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "dataset_inventory.json",
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            return load_inventory(p)
    return build_default_inventory()


def run_smoke_pipeline(
    output_dir: Path | str | None = None,
    seed: int = 42,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Execute the full ForenSight R0 pipeline end-to-end.

    Executes all 7 stages:
    1. Inventory verification (`load_inventory`).
    2. Smoke dataset and manifest generation.
    3. Leakage and generator-disjointness assertions (`assert_generator_disjoint`, `validate_no_leakage`).
    4. Dataset audit execution (`audit_manifest_images`, `audit_manifest_leakage`).
    5. Synthetic model predictions simulation across 3 distinct seeds.
    6. Evaluation runner on each seed (`evaluate_predictions` with val calibration).
    7. Multi-seed repeated run aggregation (`aggregate_reports`).

    Args:
        output_dir: Directory where smoke test artifacts are saved. If None,
            defaults to 'data/derived/smoke_test'.
        seed: Base random seed for data synthesis, auditing, and repeated evaluations.
        num_classes: Number of synthetic classes per split (default: 5).

    Returns:
        dict[str, Any]: Execution summary confirming success and performance metrics across all 7 stages.
    """
    start_time = time.perf_counter()
    out_dir = Path(output_dir or "data/derived/smoke_test").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    execution_stages: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Stage 1: Inventory verification
    # -------------------------------------------------------------------------
    inventory = _locate_or_load_inventory()
    inv_errors = inventory.validate(strict=False)
    if inv_errors:
        raise ValueError(
            f"Dataset inventory validation failed during smoke test: {inv_errors}"
        )
    execution_stages["1_inventory"] = {
        "status": "passed",
        "num_datasets": len(inventory.datasets),
        "dataset_names": inventory.list_dataset_names(),
    }

    # -------------------------------------------------------------------------
    # Stage 2: Smoke dataset and manifest generation
    # -------------------------------------------------------------------------
    manifest_paths, manifests = generate_smoke_dataset(
        output_dir=out_dir,
        num_classes=num_classes,
        seed=seed,
    )
    total_images = sum(len(m) for m in manifests.values())
    execution_stages["2_data_generation"] = {
        "status": "passed",
        "splits": list(manifests.keys()),
        "num_classes": num_classes,
        "total_images": total_images,
        "samples_per_split": {k: len(v) for k, v in manifests.items()},
    }

    # -------------------------------------------------------------------------
    # Stage 3: Leakage & generator-disjointness assertions
    # -------------------------------------------------------------------------
    train_manifest = manifests["train"]
    val_manifest = manifests["val"]
    cross_gen_manifest = manifests["cross_generator_ood"]

    # 3.1 Strict generator-disjointness between train and OOD splits
    assert_generator_disjoint(
        train_manifest=train_manifest,
        ood_manifests=[cross_gen_manifest],
    )

    # 3.2 Leakage validation between train and evaluation partitions
    validate_no_leakage(train_manifest, val_manifest, check_generators=False, strict=True)
    validate_no_leakage(train_manifest, cross_gen_manifest, check_generators=True, strict=True)

    execution_stages["3_disjointness_and_leakage"] = {
        "status": "passed",
        "generator_disjoint": True,
        "leakage_clean": True,
    }

    # -------------------------------------------------------------------------
    # Stage 4: Dataset audit execution
    # -------------------------------------------------------------------------
    leakage_audit = audit_manifest_leakage(
        train_manifest=train_manifest,
        eval_manifests={
            "val": val_manifest,
            "cross_generator_ood": cross_gen_manifest,
        },
        check_generators=True,
    )
    _assert_audit_leakage_clean(leakage_audit)

    # Image inspection audit on train manifest
    train_audit_report: AuditReport = audit_manifest_images(
        manifest=train_manifest,
        manifest_name="smoke_train_audit",
        max_workers=2,
        leakage_check_results=leakage_audit,
    )

    # Save audit artifacts
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    train_audit_report.save_json(audit_dir / "train_audit.json")
    train_audit_report.save_markdown(audit_dir / "train_audit.md")

    execution_stages["4_audit"] = {
        "status": "passed",
        "has_critical_findings": train_audit_report.has_critical_findings(),
        "summary_findings_count": len(train_audit_report.summary_findings),
        "leakage_detected": leakage_audit.get("has_leakage", False),
    }

    # -------------------------------------------------------------------------
    # Stage 5: Synthetic model predictions simulation across 3 distinct seeds
    # -------------------------------------------------------------------------
    eval_seeds = [seed, seed + 1, seed + 2]
    eval_splits = ["val", "cross_generator_ood"]

    prediction_sets: list[tuple[int, PredictionSet]] = []
    for s in eval_seeds:
        pred_records: list[PredictionRecord] = []
        for split_name in eval_splits:
            m = manifests[split_name]
            for idx, rec in enumerate(m):
                rng = np.random.RandomState(s * 1000 + idx + (100 if rec.label == 1 else 0))
                # Realistic detector: high scores for fake (mean ~0.85), low for real (mean ~0.15)
                if rec.label == 1:
                    score = float(np.clip(rng.normal(0.85, 0.08), 0.01, 0.99))
                else:
                    score = float(np.clip(rng.normal(0.15, 0.08), 0.01, 0.99))

                pred_records.append(
                    PredictionRecord(
                        sample_id=rec.sample_id,
                        label=rec.label,
                        score=score,
                        split=rec.split,
                        generator=rec.generator,
                        dataset=rec.dataset,
                        metadata={"seed": s, "simulated": True},
                    )
                )
        prediction_sets.append((s, PredictionSet(pred_records)))

    execution_stages["5_predictions"] = {
        "status": "passed",
        "seeds": eval_seeds,
        "predictions_per_seed": len(prediction_sets[0][1]),
    }

    # -------------------------------------------------------------------------
    # Stage 6: Evaluation runner on each seed (validation threshold calibration)
    # -------------------------------------------------------------------------
    eval_reports: list[EvaluationReport] = []
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for s, p_set in prediction_sets:
        report = evaluate_predictions(
            predictions=p_set,
            val_split_name="val",
            threshold_strategy="f1",
            seed=s,
            run_metadata={
                "seed": s,
                "pipeline": "smoke_test",
                "split_version": "smoke-v1",
                "model_name": "smoke_simulated",
                "variant": "smoke_simulated",
                "experiment_name": "smoke_pipeline",
            },
        )
        report.save_json(eval_dir / f"report_seed_{s}.json")
        report.save_markdown(eval_dir / f"report_seed_{s}.md")
        eval_reports.append(report)

    execution_stages["6_evaluation"] = {
        "status": "passed",
        "num_reports": len(eval_reports),
        "all_calibrated": all(r.threshold_metadata.get("calibrated", False) for r in eval_reports),
        "calibrated_thresholds": [r.threshold_metadata.get("threshold") for r in eval_reports],
    }

    # -------------------------------------------------------------------------
    # Stage 7: Multi-seed repeated run aggregation
    # -------------------------------------------------------------------------
    agg_report: AggregatedEvaluationReport = aggregate_reports(
        reports=eval_reports,
        metadata={
            "pipeline": "smoke_test",
            "base_seed": seed,
            "num_classes": num_classes,
        },
    )
    agg_report.save_json(eval_dir / "aggregated_report.json")
    agg_report.save_markdown(eval_dir / "aggregated_report.md")

    execution_stages["7_aggregation"] = {
        "status": "passed",
        "num_runs": agg_report.num_runs,
        "is_preliminary": agg_report.is_preliminary,
        "overall_auroc_mean": agg_report.overall.auroc.mean,
        "overall_accuracy_mean": agg_report.overall.accuracy.mean,
        "overall_f1_mean": agg_report.overall.f1.mean,
    }

    duration = time.perf_counter() - start_time

    summary = {
        "status": "success",
        "output_dir": str(out_dir),
        "base_seed": seed,
        "duration_seconds": round(duration, 4),
        "stages": execution_stages,
        "manifest_paths": {k: str(v) for k, v in manifest_paths.items()},
        "aggregated_report": agg_report.to_dict(),
    }

    # Save complete pipeline summary to output directory
    summary_path = out_dir / "smoke_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
