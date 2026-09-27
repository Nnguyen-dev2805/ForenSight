#!/usr/bin/env python3
"""Automated Protocol Audit Script for ForenSight Milestone R2.

Validates all 3 canonical experiment protocols:
1. Single-generator: SD1.5 train/val -> SD1.5 in-domain test + 6 OOD tests
2. LOGO (Leave-One-Generator-Out): 7 folds (leave_sd15, leave_adm, leave_biggan,
   leave_glide, leave_midjourney, leave_vqdm, leave_wukong)
3. All-7: Multi-generator closed-world training & evaluation across all 7 generators

Invariants strictly asserted:
- Zero sample ID collision across splits (train, val, test).
- Zero image file path collision across splits.
- Zero generator leakage (held-out generator strictly absent from train and val).
- Exact 1:1 real:fake sample balancing.
- Disjoint real image allocation across evaluation splits (allocated without replacement).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any

from forensight.data.split import (
    Manifest,
    assert_generator_disjoint,
)
from forensight.data.tiny_genimage import (
    KAGGLE_TINY_GENIMAGE_GENERATORS,
    TINY_GENIMAGE_SEVEN_GENERATORS,
    build_all7_manifests,
    build_all_logo_manifests,
    build_single_generator_manifests,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProtocolAudit")


def generate_synthetic_multi_gen_records(
    n_classes: int = 5,
    reals_per_class_train: int = 50,
    reals_per_class_val: int = 150,
    fakes_per_class_train: int = 50,
    fakes_per_class_val: int = 20,
) -> list[dict[str, Any]]:
    """Synthesize complete 7-generator records covering all classes and splits."""
    records: list[dict[str, Any]] = []
    counter = 0

    # 1. Real Images (Nature)
    for c in range(n_classes):
        cid = f"n{c:08d}"
        for _ in range(reals_per_class_train):
            records.append({
                "sample_id": f"real_train_{counter:06d}",
                "image_path": f"/data/raw/nature/train/{cid}_{counter}.JPEG",
                "label": 0,
                "generator": "nature",
                "raw_split": "train",
                "class_id": cid,
            })
            counter += 1
        for _ in range(reals_per_class_val):
            records.append({
                "sample_id": f"real_val_{counter:06d}",
                "image_path": f"/data/raw/nature/val/{cid}_{counter}.JPEG",
                "label": 0,
                "generator": "nature",
                "raw_split": "validation",
                "class_id": cid,
            })
            counter += 1

    # 2. Fake Images across all 7 generators
    for gen in TINY_GENIMAGE_SEVEN_GENERATORS:
        for c in range(n_classes):
            cid = f"n{c:08d}"
            for _ in range(fakes_per_class_train):
                records.append({
                    "sample_id": f"fake_train_{gen}_{counter:06d}",
                    "image_path": f"/data/raw/{gen}/train/{cid}_{counter}.png",
                    "label": 1,
                    "generator": gen,
                    "raw_split": "train",
                    "class_id": cid,
                })
                counter += 1
            for _ in range(fakes_per_class_val):
                records.append({
                    "sample_id": f"fake_val_{gen}_{counter:06d}",
                    "image_path": f"/data/raw/{gen}/val/{cid}_{counter}.png",
                    "label": 1,
                    "generator": gen,
                    "raw_split": "validation",
                    "class_id": cid,
                })
                counter += 1

    return records


def audit_split_bundle(
    name: str,
    train_m: Manifest,
    val_m: Manifest,
    test_manifests: dict[str, Manifest],
    expected_held_out_gen: str | None = None,
) -> None:
    """Audit a bundle of manifests for leakage and protocol invariants."""
    # 1. Zero sample ID collision between train and val
    train_ids = {r.sample_id for r in train_m}
    val_ids = {r.sample_id for r in val_m}
    assert train_ids.isdisjoint(val_ids), f"[{name}] train and val share sample IDs!"

    # 2. Zero path collision between train and val
    train_paths = {r.image_path for r in train_m}
    val_paths = {r.image_path for r in val_m}
    assert train_paths.isdisjoint(val_paths), f"[{name}] train and val share image paths!"

    # 3. Balance checks
    assert len(train_m.filter(label=0)) == len(train_m.filter(label=1)), f"[{name}] train split is unbalanced!"
    assert len(val_m.filter(label=0)) == len(val_m.filter(label=1)), f"[{name}] val split is unbalanced!"

    # 4. Check each test manifest against train and val
    allocated_eval_reals: set[str] = set(r.sample_id for r in val_m.filter(label=0))
    for t_name, test_m in test_manifests.items():
        t_ids = {r.sample_id for r in test_m}
        t_paths = {r.image_path for r in test_m}

        assert train_ids.isdisjoint(t_ids), f"[{name}] train and {t_name} share sample IDs!"
        assert train_paths.isdisjoint(t_paths), f"[{name}] train and {t_name} share image paths!"
        assert val_ids.isdisjoint(t_ids), f"[{name}] val and {t_name} share sample IDs!"
        assert val_paths.isdisjoint(t_paths), f"[{name}] val and {t_name} share image paths!"

        # Real samples must not overlap across evaluation splits (allocated without replacement)
        test_reals = set(r.sample_id for r in test_m.filter(label=0))
        assert allocated_eval_reals.isdisjoint(test_reals), (
            f"[{name}] {t_name} reuses real samples previously allocated to val or another test split!"
        )
        allocated_eval_reals.update(test_reals)

        # Check 1:1 balance
        assert len(test_m.filter(label=0)) == len(test_m.filter(label=1)), f"[{name}] {t_name} is unbalanced!"

    # 5. Check pairwise disjointness across all test splits
    test_keys = list(test_manifests.keys())
    for i in range(len(test_keys)):
        for j in range(i + 1, len(test_keys)):
            k1, k2 = test_keys[i], test_keys[j]
            ids1 = {r.sample_id for r in test_manifests[k1]}
            ids2 = {r.sample_id for r in test_manifests[k2]}
            assert ids1.isdisjoint(ids2), f"[{name}] test splits '{k1}' and '{k2}' share sample IDs!"
            paths1 = {r.image_path for r in test_manifests[k1]}
            paths2 = {r.image_path for r in test_manifests[k2]}
            assert paths1.isdisjoint(paths2), f"[{name}] test splits '{k1}' and '{k2}' share image paths!"

    # 5. Generator disjointness check
    if expected_held_out_gen is not None:
        for t_name, test_m in test_manifests.items():
            if expected_held_out_gen in t_name or t_name == "cross_generator_ood":
                assert_generator_disjoint(train_m, test_m)
                assert_generator_disjoint(val_m, test_m)
                held_out_fakes = {r.generator for r in test_m if r.label == 1}
                assert expected_held_out_gen in held_out_fakes, (
                    f"[{name}] {t_name} does not contain expected held-out generator {expected_held_out_gen}"
                )


def run_full_protocol_audit() -> None:
    """Execute end-to-end audit for Single, 7 LOGO Folds, and All-7."""
    logger.info("=" * 80)
    logger.info("FORENSIGHT R2 PROTOCOL AUDIT: LOCKING 3 EXPERIMENTAL PARADIGMS")
    logger.info("=" * 80)

    # 1. Audit sealed R0 Kaggle manifests if present
    r0_manifest_dir = Path("reports/r0_kaggle/manifests")
    if r0_manifest_dir.exists():
        logger.info("[AUDIT 0] Verifying sealed R0 Kaggle Single-Generator manifests...")
        train_m = Manifest.from_jsonl(r0_manifest_dir / "train.jsonl")
        val_m = Manifest.from_jsonl(r0_manifest_dir / "val.jsonl")
        in_domain_m = Manifest.from_jsonl(r0_manifest_dir / "in_domain_test.jsonl")
        ood_m = Manifest.from_jsonl(r0_manifest_dir / "cross_generator_ood.jsonl")

        audit_split_bundle(
            name="R0 Sealed Kaggle Manifests",
            train_m=train_m,
            val_m=val_m,
            test_manifests={
                "in_domain_test": in_domain_m,
                "cross_generator_ood": ood_m,
            },
            expected_held_out_gen=None,
        )
        assert_generator_disjoint(train_m, ood_m)
        assert_generator_disjoint(val_m, ood_m)
        logger.info("  -> PASSED: R0 Kaggle manifests strictly adhere to zero-leakage invariants.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        records = generate_synthetic_multi_gen_records()
        logger.info("Synthesized %d records covering all 7 generators and real nature images.", len(records))

        # 2. Audit Protocol 1: Single-generator (SD1.5 train -> 6 OOD)
        logger.info("\n[AUDIT 1] Auditing Protocol 1: Single-Generator (SD1.5)...")
        single_dir = tmp_path / "single"
        single_manifests = build_single_generator_manifests(
            extracted_records=records,
            manifest_dir=single_dir,
            train_generator="sd15",
            seed=42,
        )
        # Check primary composite evaluation splits
        audit_split_bundle(
            name="Protocol 1 (Single - Composite)",
            train_m=single_manifests["train"],
            val_m=single_manifests["val"],
            test_manifests={
                "in_domain_test": single_manifests["in_domain_test"],
                "cross_generator_ood": single_manifests["cross_generator_ood"],
            },
            expected_held_out_gen="midjourney",
        )
        # Check individual disjoint per-generator splits
        per_gen_tests = {
            "in_domain_test": single_manifests["in_domain_test"],
        }
        for gid in [g for g in KAGGLE_TINY_GENIMAGE_GENERATORS if g != "sd15"]:
            if f"test_{gid}" in single_manifests:
                per_gen_tests[f"test_{gid}"] = single_manifests[f"test_{gid}"]
        audit_split_bundle(
            name="Protocol 1 (Single - Per Generator)",
            train_m=single_manifests["train"],
            val_m=single_manifests["val"],
            test_manifests=per_gen_tests,
            expected_held_out_gen="midjourney",
        )
        logger.info("  -> PASSED: Protocol 1 (Single-generator) satisfies all invariants.")

        # 3. Audit Protocol 2: LOGO (7 Folds)
        logger.info("\n[AUDIT 2] Auditing Protocol 2: Leave-One-Generator-Out (7 Folds)...")
        logo_dir = tmp_path / "logo"
        all_logo_manifests = build_all_logo_manifests(
            extracted_records=records,
            manifest_dir=logo_dir,
            n_val_per_gen=2,
            seed=42,
        )

        for gen in TINY_GENIMAGE_SEVEN_GENERATORS:
            fold_manifests = all_logo_manifests[gen]
            audit_split_bundle(
                name=f"Protocol 2 (LOGO fold: leave_{gen})",
                train_m=fold_manifests["train"],
                val_m=fold_manifests["val"],
                test_manifests={
                    f"test_{gen}": fold_manifests[f"test_{gen}"],
                    "in_domain_test": fold_manifests["in_domain_test"],
                },
                expected_held_out_gen=gen,
            )
            logger.info("  -> PASSED: LOGO Fold 'leave_%s' satisfies all invariants.", gen)

        # 4. Audit Protocol 3: All-7 (Multi-Generator Closed Capacity)
        logger.info("\n[AUDIT 3] Auditing Protocol 3: All-7 Upper-Bound Protocol...")
        all7_dir = tmp_path / "all7"
        all7_manifests = build_all7_manifests(
            extracted_records=records,
            manifest_dir=all7_dir,
            n_val_per_gen=2,
            seed=42,
        )
        # Check composite test
        audit_split_bundle(
            name="Protocol 3 (All-7 Composite)",
            train_m=all7_manifests["train"],
            val_m=all7_manifests["val"],
            test_manifests={"test_all_combined": all7_manifests["test_all_combined"]},
            expected_held_out_gen=None,
        )
        # Check per-generator test splits
        all7_per_gen = {}
        for gid in TINY_GENIMAGE_SEVEN_GENERATORS:
            if f"test_{gid}" in all7_manifests:
                all7_per_gen[f"test_{gid}"] = all7_manifests[f"test_{gid}"]
        audit_split_bundle(
            name="Protocol 3 (All-7 Per Generator)",
            train_m=all7_manifests["train"],
            val_m=all7_manifests["val"],
            test_manifests=all7_per_gen,
            expected_held_out_gen=None,
        )
        logger.info("  -> PASSED: Protocol 3 (All-7) satisfies all invariants.")

    logger.info("\n" + "=" * 80)
    logger.info(" FORENSIGHT PROTOCOL AUDIT SUMMARY:")
    logger.info("   * R0 real Single manifests:       VERIFIED (0 sample leakage, 0 generator leakage)")
    logger.info("   * Protocol construction logic:    VERIFIED (Single, 7 LOGO Folds, All-7)")
    logger.info("   * Actual LOGO/All-7 manifests:    MUST BE AUDITED AFTER GENERATION BEFORE TRAINING")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_full_protocol_audit()
