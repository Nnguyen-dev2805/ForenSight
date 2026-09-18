"""Tests for split policy, deterministic manifests, leakage detection, and scaling tiers."""

from pathlib import Path
import pytest
import pandas as pd

from forensight.data.split import (
    ManifestRecord,
    Manifest,
    assert_generator_disjoint,
    subsample_manifest_by_class,
    create_scale_manifest,
    get_generator_membership_summary,
    validate_no_leakage,
    PROTOCOL_V1_SPLITS,
    SCALE_TIERS,
)


def _make_dummy_records(n: int = 10, split: str = "train", generator: str = "sd14", dataset: str = "genimage") -> list[ManifestRecord]:
    """Helper to generate dummy records for testing."""
    records = []
    for i in range(n):
        label = i % 2
        class_id = f"n0144076{i % 3}"
        record = ManifestRecord(
            sample_id=f"{dataset}_{generator}_{split}_{i}",
            image_path=f"/path/to/{dataset}/{generator}/{split}/{i}.jpg",
            label=label,
            dataset=dataset,
            generator=generator,
            split=split,
            class_id=class_id,
            metadata={"orig_idx": i},
        )
        records.append(record)
    return records


class TestManifestRecord:
    """Unit tests for ManifestRecord dataclass and its validations."""

    def test_valid_record(self):
        rec = ManifestRecord(
            sample_id="s1",
            image_path="/data/img1.jpg",
            label=0,
            dataset="genimage",
            generator="sd14",
            split="train",
            class_id="n01440764",
            metadata={"width": 512, "height": 512},
        )
        assert rec.sample_id == "s1"
        assert rec.label == 0
        assert rec.class_id == "n01440764"
        assert rec.metadata["width"] == 512

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="label must be 0 .* or 1"):
            ManifestRecord(
                sample_id="s1",
                image_path="/data/img1.jpg",
                label=2,  # Invalid
                dataset="genimage",
                generator="sd14",
                split="train",
            )

    def test_empty_required_field_raises(self):
        with pytest.raises(ValueError, match="sample_id must be non-empty"):
            ManifestRecord(
                sample_id="",
                image_path="/data/img1.jpg",
                label=1,
                dataset="genimage",
                generator="sd14",
                split="train",
            )

        with pytest.raises(ValueError, match="image_path must be non-empty"):
            ManifestRecord(
                sample_id="s1",
                image_path="",
                label=1,
                dataset="genimage",
                generator="sd14",
                split="train",
            )

    def test_record_dict_roundtrip(self):
        rec = ManifestRecord(
            sample_id="rec_100",
            image_path="genimage/sd14/train/img_100.png",
            label=1,
            dataset="genimage",
            generator="sd14",
            split="train",
            class_id="n02123045",
            metadata={"resolution": [512, 512]},
        )
        data = rec.to_dict()
        assert data["sample_id"] == "rec_100"
        assert data["label"] == 1
        reconstructed = ManifestRecord.from_dict(data)
        assert reconstructed == rec


class TestManifestCollectionAndIO:
    """Unit tests for Manifest collection, filtering, and serialization."""

    def test_len_indexing_and_iter(self):
        records = _make_dummy_records(5)
        manifest = Manifest(records)
        assert len(manifest) == 5
        assert manifest[0].sample_id == records[0].sample_id
        assert list(manifest) == records

    def test_filter_by_kwargs_and_predicate(self):
        records = _make_dummy_records(10)
        manifest = Manifest(records)

        # Filter by kwargs
        reals = manifest.filter(label=0)
        assert len(reals) == 5
        assert all(r.label == 0 for r in reals)

        fakes = manifest.filter(label=1)
        assert len(fakes) == 5
        assert all(r.label == 1 for r in fakes)

        # Filter by predicate
        c0 = manifest.filter(predicate=lambda r: r.class_id == "n01440760")
        assert len(c0) > 0
        assert all(r.class_id == "n01440760" for r in c0)

    def test_dataframe_conversion(self):
        records = _make_dummy_records(4)
        manifest = Manifest(records)
        df = manifest.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert list(df.columns) == [
            "sample_id",
            "image_path",
            "label",
            "dataset",
            "generator",
            "split",
            "class_id",
            "metadata",
        ]
        assert df.iloc[0]["sample_id"] == records[0].sample_id

    def test_jsonl_roundtrip(self, tmp_path):
        records = _make_dummy_records(6)
        manifest = Manifest(records)
        jsonl_path = tmp_path / "manifest.jsonl"

        manifest.to_jsonl(jsonl_path)
        assert jsonl_path.exists()

        loaded = Manifest.from_jsonl(jsonl_path)
        assert len(loaded) == len(manifest)
        assert [r.sample_id for r in loaded] == [r.sample_id for r in manifest]
        assert loaded[0] == manifest[0]

    def test_csv_roundtrip(self, tmp_path):
        records = _make_dummy_records(6)
        manifest = Manifest(records)
        csv_path = tmp_path / "manifest.csv"

        manifest.to_csv(csv_path)
        assert csv_path.exists()

        loaded = Manifest.from_csv(csv_path)
        assert len(loaded) == len(manifest)
        assert [r.sample_id for r in loaded] == [r.sample_id for r in manifest]
        assert loaded[0].metadata == manifest[0].metadata


class TestLeakageValidation:
    """Unit tests for split leakage detection."""

    def test_clean_split_no_leakage(self):
        train = Manifest(_make_dummy_records(10, split="train", generator="sd14"))
        val = Manifest(_make_dummy_records(5, split="val", generator="sd14"))
        # IDs and paths are distinct because split="val"
        errors = validate_no_leakage(train, val, strict=False)
        assert errors == []

    def test_sample_id_leakage_detected(self):
        train = Manifest(_make_dummy_records(5, split="train", generator="sd14"))
        leaked_record = ManifestRecord(
            sample_id=train[0].sample_id,
            image_path="/path/other.jpg",
            label=0,
            dataset="genimage",
            generator="sd14",
            split="val",
        )
        val = Manifest([leaked_record])

        errors = validate_no_leakage(train, val, strict=False)
        assert len(errors) > 0
        assert any("sample_id leakage" in err for err in errors)

        with pytest.raises(ValueError, match="Leakage detected"):
            validate_no_leakage(train, val, strict=True)

    def test_image_path_leakage_detected(self):
        train = Manifest(_make_dummy_records(5, split="train", generator="sd14"))
        leaked_record = ManifestRecord(
            sample_id="unique_id_999",
            image_path=train[0].image_path,  # Duplicate path!
            label=0,
            dataset="genimage",
            generator="sd14",
            split="val",
        )
        val = Manifest([leaked_record])

        errors = validate_no_leakage(train, val, strict=False)
        assert len(errors) > 0
        assert any("image_path leakage" in err for err in errors)

        with pytest.raises(ValueError, match="Leakage detected"):
            train.validate_no_leakage(val, strict=True)

    def test_cross_generator_leakage_in_validate_no_leakage(self):
        train = Manifest([
            ManifestRecord(sample_id="t1", image_path="/t1.jpg", label=1, dataset="genimage", generator="sd14", split="train"),
        ])
        test = Manifest([
            ManifestRecord(sample_id="o1", image_path="/o1.jpg", label=1, dataset="genimage", generator="sd14", split="cross_generator_ood"),
        ])
        # Generator sd14 in both train and cross_generator_ood!
        errors = validate_no_leakage(train, test, check_generators=True, strict=False)
        assert any("Generator leakage" in err for err in errors)


class TestGeneratorDisjoint:
    """Unit tests for assert_generator_disjoint helper."""

    def test_disjoint_generators_pass(self):
        train = Manifest([
            ManifestRecord(sample_id="t0", image_path="/t0.jpg", label=0, dataset="genimage", generator="nature", split="train"),
            ManifestRecord(sample_id="t1", image_path="/t1.jpg", label=1, dataset="genimage", generator="sd14", split="train"),
        ])
        ood = Manifest([
            ManifestRecord(sample_id="o0", image_path="/o0.jpg", label=0, dataset="genimage", generator="nature", split="cross_generator_ood"),
            ManifestRecord(sample_id="o1", image_path="/o1.jpg", label=1, dataset="genimage", generator="midjourney", split="cross_generator_ood"),
            ManifestRecord(sample_id="o2", image_path="/o2.jpg", label=1, dataset="genimage", generator="adm", split="cross_generator_ood"),
        ])
        # Should not raise
        assert_generator_disjoint(train, ood)

    def test_generator_overlap_raises(self):
        train = Manifest([
            ManifestRecord(sample_id="t1", image_path="/t1.jpg", label=1, dataset="genimage", generator="sd14", split="train"),
        ])
        ood = Manifest([
            ManifestRecord(sample_id="o1", image_path="/o1.jpg", label=1, dataset="genimage", generator="sd14", split="cross_generator_ood"),
        ])
        with pytest.raises(AssertionError, match="Generator leakage detected"):
            assert_generator_disjoint(train, ood)

    def test_multiple_ood_manifests_supported(self):
        train = Manifest([
            ManifestRecord(sample_id="t1", image_path="/t1.jpg", label=1, dataset="genimage", generator="sd14", split="train"),
        ])
        ood_mj = Manifest([
            ManifestRecord(sample_id="m1", image_path="/m1.jpg", label=1, dataset="genimage", generator="midjourney", split="cross_generator_ood"),
        ])
        ood_leak = Manifest([
            ManifestRecord(sample_id="l1", image_path="/l1.jpg", label=1, dataset="genimage", generator="sd14", split="cross_generator_ood"),
        ])
        assert_generator_disjoint(train, [ood_mj])
        assert_generator_disjoint(train, {"midjourney": ood_mj})

        with pytest.raises(AssertionError, match="sd14"):
            assert_generator_disjoint(train, [ood_mj, ood_leak])


class TestDeterministicScaling:
    """Unit tests for subsample_manifest_by_class and scale tiers."""

    def _create_multi_class_manifest(self) -> Manifest:
        records = []
        # Create 3 classes, each with 20 real and 20 fake
        for c_idx in range(3):
            class_id = f"n0144076{c_idx}"
            for i in range(20):
                records.append(
                    ManifestRecord(
                        sample_id=f"real_{class_id}_{i}",
                        image_path=f"/path/{class_id}/real_{i}.jpg",
                        label=0,
                        dataset="genimage",
                        generator="sd14",
                        split="train",
                        class_id=class_id,
                    )
                )
                records.append(
                    ManifestRecord(
                        sample_id=f"fake_{class_id}_{i}",
                        image_path=f"/path/{class_id}/fake_{i}.jpg",
                        label=1,
                        dataset="genimage",
                        generator="sd14",
                        split="train",
                        class_id=class_id,
                    )
                )
        return Manifest(records)

    def test_smoke_manifest_scaling(self):
        manifest = self._create_multi_class_manifest()
        assert len(manifest) == 3 * 40  # 120 items

        smoke = create_scale_manifest(manifest, tier="smoke", seed=42)
        # Smoke: max 1 real + 1 fake per class -> 3 classes * 2 = 6 items
        assert len(smoke) == 6

        # Check each class has exactly 1 real and 1 fake
        for c_idx in range(3):
            cid = f"n0144076{c_idx}"
            class_recs = smoke.filter(class_id=cid)
            assert len(class_recs.filter(label=0)) == 1
            assert len(class_recs.filter(label=1)) == 1

    def test_pilot_manifest_scaling(self):
        manifest = self._create_multi_class_manifest()
        pilot = create_scale_manifest(manifest, tier="pilot", seed=42)
        # Pilot: max 10 real + 10 fake per class -> 3 classes * 20 = 60 items
        assert len(pilot) == 60

        for c_idx in range(3):
            cid = f"n0144076{c_idx}"
            class_recs = pilot.filter(class_id=cid)
            assert len(class_recs.filter(label=0)) == 10
            assert len(class_recs.filter(label=1)) == 10

    def test_main_manifest_unmodified(self):
        manifest = self._create_multi_class_manifest()
        main_m = create_scale_manifest(manifest, tier="main", seed=42)
        assert len(main_m) == len(manifest)
        assert [r.sample_id for r in main_m] == [r.sample_id for r in manifest]

    def test_sampling_determinism(self):
        manifest = self._create_multi_class_manifest()

        sub1 = subsample_manifest_by_class(manifest, max_per_class_real=3, max_per_class_fake=3, seed=123)
        sub2 = subsample_manifest_by_class(manifest, max_per_class_real=3, max_per_class_fake=3, seed=123)

        assert [r.sample_id for r in sub1] == [r.sample_id for r in sub2]

        # Shuffled input order must yield the exact same sample
        import random
        shuffled_records = manifest.records.copy()
        random.Random(999).shuffle(shuffled_records)
        shuffled_manifest = Manifest(shuffled_records)

        sub_shuffled = subsample_manifest_by_class(shuffled_manifest, max_per_class_real=3, max_per_class_fake=3, seed=123)
        assert [r.sample_id for r in sub1] == [r.sample_id for r in sub_shuffled]

    def test_invalid_scale_tier_raises(self):
        manifest = self._create_multi_class_manifest()
        with pytest.raises(ValueError, match="Unknown scale tier"):
            create_scale_manifest(manifest, tier="ultra_large")


class TestProtocolSummary:
    """Unit tests for protocol constants and generator membership summary."""

    def test_protocol_v1_splits_consistency(self):
        assert "train" in PROTOCOL_V1_SPLITS
        assert "val" in PROTOCOL_V1_SPLITS
        assert "near_ood" in PROTOCOL_V1_SPLITS
        assert "cross_generator_ood" in PROTOCOL_V1_SPLITS
        assert "modern_external" in PROTOCOL_V1_SPLITS
        assert "real_world_external" in PROTOCOL_V1_SPLITS

        train_gens = PROTOCOL_V1_SPLITS["train"]["generators"]
        assert train_gens == ["sd14"]

        ood_gens = PROTOCOL_V1_SPLITS["cross_generator_ood"]["generators"]
        expected_ood = ["midjourney", "adm", "glide", "wukong", "vqdm", "biggan"]
        assert ood_gens == expected_ood

        # Guarantee disjointness between train and cross_generator_ood
        assert set(train_gens).isdisjoint(set(ood_gens))

    def test_membership_summary_structure(self):
        summary = get_generator_membership_summary()
        assert summary["protocol_version"] == "v1"
        assert summary["train_generators"] == ["sd14"]
        assert "cross_generator_ood" in summary
        assert set(summary["train_generators"]).isdisjoint(set(summary["cross_generator_ood"]))
        assert summary["scale_tiers"] == list(SCALE_TIERS.keys())
