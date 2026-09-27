"""Offline checks for the sealed-manifest Kaggle entrypoint."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forensight.data.split import Manifest
from scripts.run_on_kaggle import prepare_official_bundle

GENIMAGE_STYLE_FOLDERS = (
    "stable_diffusion_v_1_5",
    "imagenet_ai_0508_adm",
    "imagenet_ai_0419_biggan",
    "imagenet_glide",
    "imagenet_midjourney",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0424_wukong",
)


def _genimage_style_tree(root: Path, n_real: int = 40, n_fake: int = 20) -> Path:
    """Build a minimal tree using the real generator folder names the scanner recognises."""
    from PIL import Image

    raw = root / "raw"
    counter = 0
    for folder in GENIMAGE_STYLE_FOLDERS:
        for split in ("train", "val"):
            for i in range(n_real):
                target = raw / folder / split / "nature" / f"n{i:04d}_0.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(counter % 255, (counter * 3) % 255, (counter * 7) % 255)).save(target)
                counter += 1
            for i in range(n_fake):
                target = raw / folder / split / "ai" / f"ai{i:03d}_0.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=((counter + 11) % 255, (counter * 5) % 255, (counter * 2) % 255)).save(target)
                counter += 1
    return raw


def _repo_with_single_manifests(root: Path) -> Path:
    kernel = root / "deploy" / "kaggle"
    kernel.mkdir(parents=True)
    (kernel / "kernel-metadata.json").write_text('{"code_file": "main.py"}')
    (kernel / "official.py").write_text("# canonical kernel\n")
    for folder, filename in (("src/forensight", "__init__.py"), ("scripts", "train_r2.py"), ("configs/r2", "fusion.json")):
        target = root / folder
        target.mkdir(parents=True)
        (target / filename).write_text("{}\n")
    (root / "configs" / "r2" / "fusion.json").write_text(json.dumps({
        "variant": "fusion", "seed": 42, "model": {},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.0001},
        "evaluation": {},
    }))
    manifests = root / "data" / "manifests" / "single"
    manifests.mkdir(parents=True)
    for name in ("train", "val", "in_domain_test", "cross_generator_ood"):
        (manifests / f"{name}.jsonl").write_text("{}\n")
    (manifests.parent / "manifest_provenance.json").write_text(json.dumps({
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
        "skip_download": False,
        "split_seed": 42,
    }))
    return kernel


def test_kaggle_bundle_carries_exact_source_and_sealed_manifests(tmp_path):
    repo = tmp_path / "repo"
    kernel = _repo_with_single_manifests(repo)
    config = {
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
    }

    prepared = prepare_official_bundle(repo, kernel, tmp_path / "bundle", config)

    bundle = tmp_path / "bundle"
    assert (bundle / "main.py").read_text() == (kernel / "official.py").read_text()
    assert (bundle / "src" / "forensight" / "__init__.py").exists()
    assert (bundle / "data" / "manifests" / "single" / "train.jsonl").exists()
    assert len(prepared["source_payload_sha256"]) == 64
    assert json.loads((bundle / "run_config.json").read_text()) == prepared


def test_kaggle_bundle_code_digest_is_stable_across_kernel_identity(tmp_path):
    """Variant kernels must agree on the code digest but not the full payload.

    `source_payload_sha256` covers the whole staged bundle, so it necessarily
    differs between variant kernels (their kernel-metadata.json differs). The
    code digest covers only canonical source, so it must be identical -- that is
    what makes cross-variant results comparable.
    """
    repo = tmp_path / "repo"
    kernel = _repo_with_single_manifests(repo)
    other = repo / "deploy" / "kaggle_semantic"
    other.mkdir(parents=True)
    (other / "kernel-metadata.json").write_text(
        '{"id": "user/forensight-r2-semantic-training", "code_file": "main.py"}'
    )

    config = {
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
    }
    fusion = prepare_official_bundle(repo, kernel, tmp_path / "b_fusion", dict(config))
    semantic = prepare_official_bundle(repo, other, tmp_path / "b_semantic", dict(config))

    assert len(fusion["code_payload_sha256"]) == 64
    assert fusion["code_payload_sha256"] == semantic["code_payload_sha256"]
    assert fusion["source_payload_sha256"] != semantic["source_payload_sha256"]


def test_kaggle_bundle_code_digest_ignores_run_specific_manifests(tmp_path):
    """The code digest must not move when only run inputs change."""
    repo = tmp_path / "repo"
    kernel = _repo_with_single_manifests(repo)
    config = {
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
    }
    first = prepare_official_bundle(repo, kernel, tmp_path / "b1", dict(config))

    # Change a sealed manifest: the full payload changes, canonical code does not.
    (repo / "data" / "manifests" / "single" / "train.jsonl").write_text('{"changed": true}\n')
    second = prepare_official_bundle(repo, kernel, tmp_path / "b2", dict(config))

    assert first["code_payload_sha256"] == second["code_payload_sha256"]
    assert first["source_payload_sha256"] != second["source_payload_sha256"]


def test_kaggle_bundle_rejects_unpinned_source_or_missing_manifests(tmp_path):
    repo = tmp_path / "repo"
    kernel = _repo_with_single_manifests(repo)
    config = {"experiment": "single", "variant": "fusion", "dataset_ref": "yangsangtai/tiny-genimage"}
    with pytest.raises(ValueError, match="version"):
        prepare_official_bundle(repo, kernel, tmp_path / "bundle", config)

    config["dataset_ref"] = "yangsangtai/tiny-genimage/versions/1"
    (repo / "data" / "manifests" / "single" / "val.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="val.jsonl"):
        prepare_official_bundle(repo, kernel, tmp_path / "bundle", config)


def test_kaggle_bundle_rejects_manifests_from_another_dataset_version(tmp_path):
    repo = tmp_path / "repo"
    kernel = _repo_with_single_manifests(repo)
    config = {
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/2",
    }
    with pytest.raises(ValueError, match="dataset_ref"):
        prepare_official_bundle(repo, kernel, tmp_path / "bundle", config)


def test_local_manifest_preparation_records_dataset_source(tmp_path):
    from scripts.download_tiny_genimage import run_download_and_prep

    raw_root = _genimage_style_tree(tmp_path)
    manifests = tmp_path / "manifests"
    run_download_and_prep(
        dataset_ref="yangsangtai/tiny-genimage/versions/1",
        output_dir=raw_root,
        manifest_dir=manifests,
        skip_download=True,
        experiment="single",
    )
    provenance = json.loads((manifests / "manifest_provenance.json").read_text())
    assert provenance["dataset_ref"] == "yangsangtai/tiny-genimage/versions/1"
    assert provenance["split_seed"] == 42
    assert provenance["skip_download"] is True


def test_official_kernel_uses_shared_code_and_staged_manifests(tmp_path):
    from deploy.kaggle import official

    _repo_with_single_manifests(tmp_path)
    (tmp_path / "run_config.json").write_text(json.dumps({
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
        "source_payload_sha256": "payload-sha", "git_dirty": True,
        "git_commit": None,
    }))
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    with patch("scripts.check_dataset.main", return_value=0) as audit, \
         patch("scripts.train_r2.main", return_value=0) as train, \
         patch("scripts.evaluate_r2.main", side_effect=_fake_evaluate_writing_canonical_report) as evaluate:
        run_dir = official.run_official(tmp_path, raw_root)

    assert "--train-manifest" in audit.call_args.args[0]
    train_args = train.call_args.args[0]
    assert str(tmp_path / "data" / "manifests" / "single" / "train.jsonl") in train_args
    assert "--base-dir" in train_args and str(raw_root) in train_args
    eval_args = evaluate.call_args.args[0]
    assert "--dataset-revision" in eval_args
    assert "kaggle:yangsangtai/tiny-genimage/versions/1" in eval_args
    active = json.loads((tmp_path / "active_config.json").read_text())
    assert active["source_payload_sha256"] == "payload-sha"
    assert active["git_commit"] is None
    assert (run_dir / "manifest_provenance.json").is_file()


def test_official_kernel_checks_manifests_before_downloading(tmp_path):
    from deploy.kaggle import official

    _repo_with_single_manifests(tmp_path)
    (tmp_path / "data" / "manifests" / "single" / "val.jsonl").unlink()
    (tmp_path / "run_config.json").write_text(json.dumps({
        "experiment": "single", "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
    }))
    with patch("sys.modules", {"kagglehub": MagicMock(dataset_download=MagicMock(side_effect=AssertionError("downloaded")))}):
        with pytest.raises(FileNotFoundError, match="val.jsonl"):
            official.main(tmp_path)


def _empty_manifest_bundle_repo(root: Path) -> Path:
    """A repository that has the kernel code but no local manifests at all."""
    kernel = root / "deploy" / "kaggle"
    kernel.mkdir(parents=True)
    (kernel / "kernel-metadata.json").write_text('{"code_file": "main.py"}')
    (kernel / "official.py").write_text("# canonical kernel\n")
    for folder, filename in (("src/forensight", "__init__.py"), ("scripts", "train_r2.py"), ("configs/r2", "fusion.json")):
        target = root / folder
        target.mkdir(parents=True)
        (target / filename).write_text("{}\n")
    return kernel


def test_kaggle_bundle_can_prepare_manifests_in_kernel(tmp_path):
    repo = tmp_path / "repo"
    kernel = _empty_manifest_bundle_repo(repo)
    config = {
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
        "prepare_manifests": True,
    }

    prepared = prepare_official_bundle(repo, kernel, tmp_path / "bundle", config)

    bundle = tmp_path / "bundle"
    assert prepared["prepare_manifests"] is True
    assert json.loads((bundle / "run_config.json").read_text())["prepare_manifests"] is True
    assert (bundle / "src" / "forensight" / "__init__.py").exists()
    # Nothing to ship: the kernel builds the splits from the pinned dataset.
    assert not (bundle / "data" / "manifests").exists()


def test_prepare_manifests_in_kernel_builds_sealed_splits(tmp_path):
    from deploy.kaggle import official

    raw_root = _genimage_style_tree(tmp_path)
    config = {
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
    }

    provenance = official.prepare_manifests_in_kernel(tmp_path, raw_root, config)

    manifests = tmp_path / "data" / "manifests" / "single"
    assert (manifests / "train.jsonl").is_file()
    assert (manifests / "cross_generator_ood.jsonl").is_file()
    assert provenance["generated_in_kernel"] is True
    assert provenance["scanned_images"] > 0
    assert json.loads((tmp_path / "data" / "manifests" / "manifest_provenance.json").read_text()) == provenance

    # Manifests reference images relatively, so `--base-dir <raw_root>` resolves them
    # on the Kaggle side without shipping any raw data.
    for record in list(Manifest.from_jsonl(manifests / "train.jsonl"))[:5]:
        assert not Path(record.image_path).is_absolute()
        assert (raw_root / record.image_path).is_file()


def test_official_kernel_prepare_mode_skips_manifest_precheck(tmp_path):
    from deploy.kaggle import official

    (tmp_path / "run_config.json").write_text(json.dumps({
        "experiment": "single", "variant": "fusion", "seed": 42,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
        "prepare_manifests": True,
    }))
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    seen: dict = {}

    with patch.dict(sys.modules, {"kagglehub": MagicMock(dataset_download=MagicMock(return_value=str(raw_root)))}), \
         patch.object(official, "prepare_manifests_in_kernel", side_effect=lambda root, raw, config: seen.update(raw=raw)), \
         patch.object(official, "run_official", return_value=tmp_path), \
         patch.object(official.shutil, "make_archive", return_value=None):
        assert official.main(tmp_path) == 0

    assert seen["raw"] == raw_root


def test_logo_folds_use_distinct_output_directories(tmp_path):
    from deploy.kaggle import official

    _repo_with_single_manifests(tmp_path)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for leave_out in ("adm", "biggan"):
        folder = tmp_path / "data" / "manifests" / "logo" / f"leave_{leave_out}"
        folder.mkdir(parents=True)
        for name in ("train", "val", "test_in_domain_seen", f"test_{leave_out}"):
            (folder / f"{name}.jsonl").write_text("{}\n")

    outputs = []
    with patch("scripts.check_dataset.main", return_value=0), \
         patch("scripts.train_r2.main", return_value=0), \
         patch("scripts.evaluate_r2.main", side_effect=_fake_evaluate_writing_canonical_report):
        for leave_out in ("adm", "biggan"):
            (tmp_path / "run_config.json").write_text(json.dumps({
                "experiment": "logo", "leave_out": leave_out,
                "variant": "fusion", "seed": 42,
                "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
            }))
            outputs.append(official.run_official(tmp_path, raw_root))

    assert outputs[0] != outputs[1]
    assert "leave_adm" in str(outputs[0])
    assert "leave_biggan" in str(outputs[1])


def _multi_seed_repo(tmp_path: Path, seeds: list[int]) -> Path:
    """Repo + run_config for a multi-seed single run, ready for run_official."""
    _repo_with_single_manifests(tmp_path)
    (tmp_path / "run_config.json").write_text(json.dumps({
        "experiment": "single", "variant": "fusion", "seeds": seeds,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
    }))
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    return raw_root


def _fake_evaluate_writing_canonical_report(args: list[str]) -> int:
    """Stand in for evaluate_r2: write the report at the path it was asked for.

    `scripts/evaluate_r2.py` writes to `--report-json` when given, and otherwise
    defaults to `evaluation.json` inside `--output-dir`. Mirroring both keeps
    these tests honest about what the real entrypoint produces.
    """
    if "--report-json" in args:
        target = Path(args[args.index("--report-json") + 1])
    else:
        target = Path(args[args.index("--output-dir") + 1]) / "evaluation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "overall": {
            "auroc": 0.5, "accuracy": 0.5, "f1": 0.5, "precision": 0.5, "recall": 0.5,
            "threshold": 0.5, "threshold_source": "val_optimal_f1",
            "confusion_matrix": {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        },
    }))
    return 0


def test_official_multi_seed_aggregates_reports_that_evaluate_r2_writes(tmp_path):
    """Aggregation must receive the report path evaluate_r2 actually produces.

    Regression: official.py collected `run_dir/"eval_report.json"` while
    evaluate_r2 writes `run_dir/"evaluation.json"`, so every multi-seed run
    handed aggregate_runs non-existent paths and produced no aggregated report.
    """
    from deploy.kaggle import official

    raw_root = _multi_seed_repo(tmp_path, [42, 43])
    seen: dict = {}

    def fake_aggregate(args: list[str]) -> int:
        # Collect exactly the operands of --reports (stops at the next flag).
        rest = args[args.index("--reports") + 1:]
        reports = []
        for token in rest:
            if token.startswith("--"):
                break
            reports.append(token)
        seen["reports"] = reports
        seen["all_exist"] = all(Path(a).is_file() for a in reports)
        return 0

    with patch("scripts.check_dataset.main", return_value=0), \
         patch("scripts.train_r2.main", return_value=0), \
         patch("scripts.evaluate_r2.main", side_effect=_fake_evaluate_writing_canonical_report), \
         patch("scripts.aggregate_runs.main", side_effect=fake_aggregate):
        official.run_official(tmp_path, raw_root)

    assert seen["reports"], "aggregate_runs was never invoked for a multi-seed run"
    assert seen["all_exist"], (
        f"aggregate_runs was handed reports that do not exist: {seen['reports']}"
    )


def test_official_multi_seed_fails_loudly_when_aggregation_fails(tmp_path):
    """A failed aggregation must abort, not silently return a partial run.

    Regression: official.py ignored aggregate_runs' return code, so a broken
    aggregation produced a run directory with no aggregated_report.json and no
    error signal.
    """
    from deploy.kaggle import official

    raw_root = _multi_seed_repo(tmp_path, [42, 43])

    with patch("scripts.check_dataset.main", return_value=0), \
         patch("scripts.train_r2.main", return_value=0), \
         patch("scripts.evaluate_r2.main", side_effect=_fake_evaluate_writing_canonical_report), \
         patch("scripts.aggregate_runs.main", return_value=1):
        with pytest.raises(RuntimeError, match="aggregat"):
            official.run_official(tmp_path, raw_root)


def test_prepare_mode_keeps_same_split_across_training_seeds(tmp_path):
    from deploy.kaggle import official

    raw_root = _genimage_style_tree(tmp_path)
    cohorts = []
    provenance_records = []
    for training_seed in (42, 43):
        run_root = tmp_path / f"seed_{training_seed}"
        provenance = official.prepare_manifests_in_kernel(run_root, raw_root, {
            "experiment": "single", "variant": "fusion", "seed": training_seed,
            "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
        })
        val = Manifest.from_jsonl(run_root / "data" / "manifests" / "single" / "val.jsonl")
        cohorts.append({record.sample_id for record in val})
        provenance_records.append(provenance)

    assert cohorts[0] == cohorts[1]
    assert [p["split_seed"] for p in provenance_records] == [42, 42]
    assert [p["training_seed"] for p in provenance_records] == [42, 43]


def test_official_run_exports_generated_manifest_provenance(tmp_path):
    from deploy.kaggle import official

    _repo_with_single_manifests(tmp_path)
    source = tmp_path / "data" / "manifests" / "manifest_provenance.json"
    source.write_text(json.dumps({"split_seed": 42, "dataset_ref": "yangsangtai/tiny-genimage/versions/1"}))
    (tmp_path / "run_config.json").write_text(json.dumps({
        "experiment": "single", "variant": "fusion", "seed": 43,
        "prepare_manifests": True,
        "dataset_ref": "yangsangtai/tiny-genimage/versions/1",
    }))
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    with patch("scripts.check_dataset.main", return_value=0), \
         patch("scripts.train_r2.main", return_value=0), \
         patch("scripts.evaluate_r2.main", side_effect=_fake_evaluate_writing_canonical_report):
        run_dir = official.run_official(tmp_path, raw_root)

    assert json.loads((run_dir / "manifest_provenance.json").read_text()) == json.loads(source.read_text())
