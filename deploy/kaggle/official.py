#!/usr/bin/env python3
"""Kaggle entrypoint for sealed ForenSight R2 experiments.

The local bridge stages this file as main.py together with the canonical
ForenSight package, configs, and R0 manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT).startswith("/kaggle/src"):
    ROOT = Path("/kaggle/working")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

INJECTED_CONFIG: dict = {}
INJECTED_BUNDLE_B64: str = ""


def bootstrap_environment(root: Path) -> Path:
    """Unpack self-contained bundle if executing in Kaggle's single-file script sandbox."""
    if INJECTED_BUNDLE_B64 and not (root / "src" / "forensight").exists():
        import base64
        import io
        import tarfile

        data = base64.b64decode(INJECTED_BUNDLE_B64)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(root)
    if INJECTED_CONFIG and not (root / "run_config.json").exists():
        (root / "run_config.json").write_text(json.dumps(INJECTED_CONFIG, indent=2), encoding="utf-8")
    sys.path_importer_cache.clear()
    for p in [str(root), str(root / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def resolve_manifests(root: Path, config: dict) -> tuple[Path, Path, list[Path]]:
    """Choose disjoint sealed partitions; aggregate and child tests never mix."""
    base = root / "data" / "manifests"
    experiment = config["experiment"]
    if experiment == "single":
        folder = base / "single"
        tests = [folder / "in_domain_test.jsonl", folder / "cross_generator_ood.jsonl"]
    elif experiment == "logo":
        from forensight.data.tiny_genimage import KAGGLE_TINY_GENIMAGE_GENERATORS

        leave_out = config.get("leave_out")
        if leave_out not in KAGGLE_TINY_GENIMAGE_GENERATORS:
            raise ValueError("Official LOGO run requires a valid --leave-out generator")
        folder = base / "logo" / f"leave_{leave_out}"
        tests = [folder / "test_in_domain_seen.jsonl", folder / f"test_{leave_out}.jsonl"]
    elif experiment == "all7":
        folder = base / "all7"
        tests = [folder / "test_all_combined.jsonl"]
    else:
        raise ValueError("Official experiment must be single, logo, or all7")

    train, val = folder / "train.jsonl", folder / "val.jsonl"
    for path in (train, val, *tests):
        if not path.is_file():
            raise FileNotFoundError(f"Sealed manifest missing: {path}")
    return train, val, tests


def run_official(root: Path, raw_root: Path) -> Path:
    """Audit real images, then train and evaluate with the shared local code."""
    from scripts import check_dataset, evaluate_r2, train_r2

    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    train, val, tests = resolve_manifests(root, config)
    dataset_ref = config["dataset_ref"]
    if "/versions/" not in dataset_ref:
        raise ValueError("Official dataset_ref must pin a Kaggle dataset version")

    variant = {
        "semantic_only": "semantic", "forensic_only": "forensic",
    }.get(config["variant"], config["variant"])
    if variant not in {"semantic", "forensic", "fusion"}:
        raise ValueError(f"Unsupported R2 variant: {variant}")
    seeds = config.get("seeds")
    if seeds is None:
        seed_val = config.get("seed")
        seeds = [int(seed_val)] if seed_val is not None else [42]
    else:
        seeds = [int(s) for s in seeds]
    experiment_name = config["experiment"]
    if experiment_name == "logo":
        experiment_name = f"logo_leave_{config['leave_out']}"

    exp_dir = root / "results" / "r2" / f"{experiment_name}_{variant}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    audit_json = exp_dir / "raw_dataset_audit.json"
    audit_args = [
        "--train-manifest", str(train), "--eval-manifest", str(val),
        *(str(path) for path in tests), "--base-dir", str(raw_root),
        "--output-json", str(audit_json),
        "--output-md", str(audit_json.with_suffix(".md")), "--strict",
    ]
    if check_dataset.main(audit_args) != 0:
        raise RuntimeError("R0 raw-data audit failed; training was not started")

    eval_reports = []
    last_run_dir = None
    for seed in seeds:
        run_dir = exp_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "data" / "manifests" / "manifest_provenance.json", run_dir)
        if audit_json.is_file():
            shutil.copy2(audit_json, run_dir / "raw_dataset_audit.json")
        if audit_json.with_suffix(".md").is_file():
            shutil.copy2(audit_json.with_suffix(".md"), run_dir / "raw_dataset_audit.md")

        base_config = json.loads((root / "configs" / "r2" / f"{variant}.json").read_text(encoding="utf-8"))
        base_config["seed"] = seed
        base_config["experiment_mode"] = config["experiment"]
        base_config["dataset_revision"] = f"kaggle:{dataset_ref}"
        if config.get("prepare_manifests"):
            base_config["split_seed"] = 42
        for key in (
            "source_payload_sha256", "code_payload_sha256",
            "git_commit", "base_commit", "git_dirty",
        ):
            base_config[key] = config.get(key)
        for key in ("epochs", "batch_size", "learning_rate"):
            if config.get(key) is not None:
                base_config["training"][key] = config[key]
        active_config = root / "active_config.json"
        active_config.write_text(json.dumps(base_config, indent=2), encoding="utf-8")

        train_args = [
            "--config", str(active_config), "--train-manifest", str(train),
            "--val-manifest", str(val), "--base-dir", str(raw_root),
            "--output-dir", str(run_dir),
        ]
        if train_r2.main(train_args) != 0:
            raise RuntimeError(f"R2 training failed for seed {seed}")

        split_digest = hashlib.sha256()
        for path in (train, val, *tests):
            split_digest.update(path.read_bytes())
        # Name the report explicitly instead of relying on evaluate_r2's default:
        # the aggregator below reads this exact path, so the producer and the
        # consumer must agree on it by construction, not by coincidence.
        eval_report = run_dir / "evaluation.json"
        eval_args = [
            "--config", str(run_dir / "config.json"),
            "--checkpoint", str(run_dir / "checkpoint.pt"),
            "--train-manifest", str(train), "--val-manifest", str(val),
            "--eval-manifest", *(str(path) for path in tests),
            "--base-dir", str(raw_root), "--output-dir", str(run_dir),
            "--report-json", str(eval_report),
            "--dataset", "yangsangtai/tiny-genimage",
            "--dataset-revision", f"kaggle:{dataset_ref}",
            "--split-version", split_digest.hexdigest()[:16],
        ]
        if evaluate_r2.main(eval_args) != 0:
            raise RuntimeError(f"R2 evaluation failed for seed {seed}")
        if not eval_report.is_file():
            raise RuntimeError(
                f"R2 evaluation reported success for seed {seed} but wrote no "
                f"report at {eval_report}."
            )
        eval_reports.append(eval_report)
        last_run_dir = run_dir

    if len(eval_reports) > 1:
        from scripts import aggregate_runs
        # A failed aggregation must abort: otherwise the run directory looks
        # complete while silently missing the aggregated_report.json that the
        # multi-seed protocol requires.
        if aggregate_runs.main([
            "--reports", *(str(p) for p in eval_reports),
            "--output-json", str(exp_dir / "aggregated_report.json"),
            "--output-md", str(exp_dir / "aggregated_report.md"),
        ]) != 0:
            raise RuntimeError(
                f"R2 aggregation failed for {len(eval_reports)} seeds in {exp_dir}"
            )
        return exp_dir
    return last_run_dir


def prepare_manifests_in_kernel(root: Path, raw_root: Path, config: dict) -> dict:
    """Build sealed manifests from the already-downloaded dataset inside the kernel.

    Uses the same canonical builders, seed, and parameters as local preparation, so the
    resulting splits are identical to a locally prepared bundle built from the same pinned
    dataset version. No raw dataset ever has to be stored on a development machine.
    """
    from forensight.data.tiny_genimage import build_protocol_manifests, scan_kaggle_tiny_genimage

    records = scan_kaggle_tiny_genimage(raw_root)
    if not records:
        raise RuntimeError(
            f"No images recognised under {raw_root}; refusing to build empty manifests."
        )
    print(f"Scanned {len(records)} images from {raw_root}")

    manifest_root = root / "data" / "manifests"
    split_seed = 42
    manifests = build_protocol_manifests(
        extracted_records=records,
        manifest_dir=manifest_root,
        experiment=config["experiment"],
        seed=split_seed,
        leave_out_gen=config.get("leave_out"),
    )

    provenance = {
        "generated_in_kernel": True,
        "dataset_ref": config.get("dataset_ref"),
        "scan_root": str(raw_root),
        "experiment": config["experiment"],
        "leave_out": config.get("leave_out"),
        "split_seed": split_seed,
        "training_seed": int(config.get("seed", config.get("seeds", [42])[0])),
        "training_seeds": [int(s) for s in config.get("seeds", [config.get("seed", 42)])],
        "scanned_images": len(records),
        "split_counts": {name: len(manifest) for name, manifest in manifests.items()},
    }
    (manifest_root / "manifest_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("In-kernel manifest provenance:", json.dumps(provenance))
    return provenance


def main(root: Path = ROOT) -> int:
    bootstrap_environment(root)
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    prepare_manifests = bool(config.get("prepare_manifests"))
    if not prepare_manifests:
        resolve_manifests(root, config)  # Reject missing sealed inputs before any download.

    for module, package in (("kagglehub", "kagglehub>=1.0"), ("open_clip", "open_clip_torch")):
        try:
            __import__(module)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    import kagglehub

    dataset_ref = config["dataset_ref"]
    dataset_name = dataset_ref.split("/")[1] if "/" in dataset_ref else dataset_ref
    mounted_path = Path("/kaggle/input") / dataset_name
    if (mounted_path / "versions" / "1").exists():
        raw_root = mounted_path / "versions" / "1"
    elif mounted_path.exists():
        raw_root = mounted_path
    else:
        raw_root = Path(kagglehub.dataset_download(dataset_ref))
    if prepare_manifests:
        prepare_manifests_in_kernel(root, raw_root, config)
    run_official(root, raw_root)
    shutil.make_archive(str(root / "results_r2"), "zip", root_dir=root, base_dir="results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
