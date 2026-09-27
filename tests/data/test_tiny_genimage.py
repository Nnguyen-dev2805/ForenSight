"""Unit tests for Tiny-GenImage dataset extraction and manifest generation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forensight.data.split import assert_generator_disjoint, validate_no_leakage
from forensight.data.tiny_genimage import (
    CROSS_GENERATOR_OOD_IDS,
    KAGGLE_TINY_GENIMAGE_GENERATORS,
    TINY_GENIMAGE_GENERATOR_MAP,
    TINY_GENIMAGE_SEVEN_GENERATORS,
    build_all7_manifests,
    build_all_experiment_manifests,
    build_all_in_one_manifests,
    build_all_logo_manifests,
    build_logo_manifests,
    build_protocol_manifests,
    build_single_generator_manifests,
    build_tiny_genimage_manifests,
    detect_image_extension,
    download_tiny_genimage_parquets,
    extract_image_bytes,
    extract_parquet_images,
    scan_kaggle_tiny_genimage,
)


class TestImageByteExtraction:
    def test_extract_from_bytes(self):
        data = b"\x89PNG\r\n\x1a\nfakeimage"
        assert extract_image_bytes(data) == data

    def test_extract_from_bytearray(self):
        data = bytearray(b"bytearray_content")
        assert extract_image_bytes(data) == b"bytearray_content"

    def test_extract_from_dict_bytes(self):
        data = {"bytes": b"dict_bytes", "path": None}
        assert extract_image_bytes(data) == b"dict_bytes"

    def test_extract_from_dict_path(self, tmp_path: Path):
        file = tmp_path / "test.jpg"
        file.write_bytes(b"image_on_disk")
        data = {"bytes": None, "path": str(file)}
        assert extract_image_bytes(data) == b"image_on_disk"

    def test_extract_from_pyarrow_scalar(self):
        arr = pa.array([{"bytes": b"pyarrow_bytes", "path": None}])
        scalar = arr[0]
        assert extract_image_bytes(scalar) == b"pyarrow_bytes"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unable to extract image bytes"):
            extract_image_bytes(12345)


class TestImageExtensionDetection:
    def test_png_detection(self):
        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00"
        assert detect_image_extension(data) == ".png"

    def test_jpeg_detection(self):
        data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert detect_image_extension(data) == ".jpg"

    def test_webp_detection(self):
        data = b"RIFF\x00\x00\x00\x00WEBPVP8 "
        assert detect_image_extension(data) == ".webp"

    def test_fallback_detection(self):
        data = b"unknown_signature_data"
        assert detect_image_extension(data) == ".png"


class TestParquetExtraction:
    def _create_synthetic_parquet(self, path: Path, n_rows: int = 10) -> Path:
        """Create a tiny valid Parquet file with synthetic image data."""
        # 1x1 1-byte dummy PNG
        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        images = [{"bytes": dummy_png, "path": None} for _ in range(n_rows)]
        labels = [0 if i % 2 == 0 else 1 for i in range(n_rows)]
        # generator: 0 for nature (even), 6 for SD1.5 (odd)
        generators = [0 if i % 2 == 0 else 6 for i in range(n_rows)]

        table = pa.Table.from_arrays(
            [pa.array(images), pa.array(labels), pa.array(generators)],
            names=["image", "label", "generator"],
        )
        pq.write_table(table, path)
        return path

    def test_extract_parquet_images(self, tmp_path: Path):
        parquet_file = self._create_synthetic_parquet(tmp_path / "test.parquet", n_rows=6)
        output_dir = tmp_path / "extracted"

        records, next_idx = extract_parquet_images(
            parquet_path=parquet_file,
            output_root=output_dir,
            raw_split="train",
            start_index=0,
        )

        assert len(records) == 6
        assert next_idx == 6

        # Check file creation on disk
        for rec in records:
            img_path = output_dir / rec["image_path"]
            assert img_path.exists()
            assert img_path.stat().st_size > 0
            assert rec["raw_split"] == "train"
            if rec["label"] == 0:
                assert rec["generator"] == "nature"
            else:
                assert rec["generator"] == "sd15"


class TestManifestConstructionAndLeakageInvariants:
    def _build_synthetic_dataset_records(self) -> list[dict]:
        """Build synthetic extracted metadata for the seven-generator Kaggle protocol."""
        records = []
        counter = 0

        # Raw train: 20 real, 10 SD1.5 fake, 10 Midjourney fake
        for i in range(20):
            records.append({
                "sample_id": f"train_real_{i}",
                "image_path": f"nature/train/nature_{i}.png",
                "label": 0,
                "generator": "nature",
                "raw_split": "train",
                "index": counter,
            })
            counter += 1

        for i in range(10):
            records.append({
                "sample_id": f"train_sd15_{i}",
                "image_path": f"sd15/train/sd15_{i}.png",
                "label": 1,
                "generator": "sd15",
                "raw_split": "train",
                "index": counter,
            })
            counter += 1

        for i in range(10):
            records.append({
                "sample_id": f"train_mj_{i}",
                "image_path": f"midjourney/train/mj_{i}.png",
                "label": 1,
                "generator": "midjourney",
                "raw_split": "train",
                "index": counter,
            })
            counter += 1

        # Raw validation: 40 real, SD1.5 + six OOD generators
        for i in range(40):
            records.append({
                "sample_id": f"val_real_{i}",
                "image_path": f"nature/val/nature_{i}.png",
                "label": 0,
                "generator": "nature",
                "raw_split": "validation",
                "index": counter,
            })
            counter += 1

        # SD1.5 validation fakes (6 images: 3 for val, 3 for in_domain_test)
        for i in range(6):
            records.append({
                "sample_id": f"val_sd15_{i}",
                "image_path": f"sd15/val/sd15_{i}.png",
                "label": 1,
                "generator": "sd15",
                "raw_split": "validation",
                "index": counter,
            })
            counter += 1

        # 6 OOD generators (4 images each = 24 images)
        for gen_id in CROSS_GENERATOR_OOD_IDS:
            for i in range(4):
                records.append({
                    "sample_id": f"val_{gen_id}_{i}",
                    "image_path": f"{gen_id}/val/{gen_id}_{i}.png",
                    "label": 1,
                    "generator": gen_id,
                    "raw_split": "validation",
                    "index": counter,
                })
                counter += 1

        return records

    def test_manifest_disjointness_and_invariants(self, tmp_path: Path):
        records = self._build_synthetic_dataset_records()
        manifest_dir = tmp_path / "manifests"

        manifests = build_tiny_genimage_manifests(
            extracted_records=records,
            manifest_dir=manifest_dir,
            val_in_domain_ratio=0.5,
        )

        train_m = manifests["tiny_genimage_train"]
        val_m = manifests["tiny_genimage_val"]
        test_in_domain_m = manifests["tiny_genimage_in_domain_test"]
        cross_ood_m = manifests["tiny_genimage_cross_generator_ood"]

        # 1. Zero generator leakage
        assert_generator_disjoint(train_m, cross_ood_m)
        assert_generator_disjoint(val_m, cross_ood_m)

        # Train contains ONLY sd15 and nature (midjourney from raw train MUST be excluded)
        train_generators = {r.generator for r in train_m}
        assert train_generators == {"nature", "sd15"}

        # 2. Zero sample leakage between val and in_domain_test
        val_ids = {r.sample_id for r in val_m}
        test_in_domain_ids = {r.sample_id for r in test_in_domain_m}
        assert val_ids.isdisjoint(test_in_domain_ids)

        # Zero sample leakage between train and val/test
        train_ids = {r.sample_id for r in train_m}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_in_domain_ids)

        # Zero sample leakage between evaluation splits' real images
        cross_ood_ids = {r.sample_id for r in cross_ood_m}
        assert val_ids.isdisjoint(cross_ood_ids)
        assert test_in_domain_ids.isdisjoint(cross_ood_ids)

        # 3. Label balance (1:1)
        for name, m in manifests.items():
            reals = len(m.filter(label=0))
            fakes = len(m.filter(label=1))
            assert reals == fakes, f"Split {name} is not balanced: {reals} reals vs {fakes} fakes"

        # 4. Check serialized files on disk
        for name in (
            "tiny_genimage_train",
            "tiny_genimage_val",
            "tiny_genimage_in_domain_test",
            "tiny_genimage_cross_generator_ood",
        ):
            jsonl_path = manifest_dir / f"{name}.jsonl"
            assert jsonl_path.exists()
            assert jsonl_path.stat().st_size > 0

    def test_scan_kaggle_tree_recognizes_seven_generators(self, tmp_path: Path):
        for gen in ("imagenet_ai_0424_sdv5", "imagenet_midjourney", "imagenet_ai_0508_adm",
                    "imagenet_glide", "imagenet_ai_0424_wukong", "imagenet_ai_0419_vqdm",
                    "imagenet_ai_0419_biggan"):
            fake = tmp_path / gen / "train" / "ai" / "fake.jpg"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.write_bytes(b"fake")
            val_fake = tmp_path / gen / "val" / "ai" / "fake.jpg"
            val_fake.parent.mkdir(parents=True, exist_ok=True)
            val_fake.write_bytes(b"fake")

        for split in ("train", "val"):
            real = tmp_path / "Nature" / split / "real" / "real.jpg"
            real.parent.mkdir(parents=True, exist_ok=True)
            real.write_bytes(b"real")

        records = scan_kaggle_tiny_genimage(tmp_path)
        assert {r["generator"] for r in records if r["label"] == 1} == {
            "sd15", "midjourney", "adm", "glide", "wukong", "vqdm", "biggan"
        }
        assert {r["raw_split"] for r in records} == {"train", "validation"}


class TestMockHfDownload:
    @patch("huggingface_hub.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_download_parquets_mock(self, mock_download, mock_api_cls, tmp_path: Path):
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.list_repo_files.return_value = [
            "data/train-00000-of-00014.parquet",
            "data/validation-00000-of-00004.parquet",
            "README.md",
        ]

        def fake_download(repo_id, filename, repo_type, token, local_dir):
            p = Path(local_dir) / Path(filename).name
            p.write_bytes(b"dummy_parquet_data")
            return str(p)

        mock_download.side_effect = fake_download

        splits = download_tiny_genimage_parquets(output_dir=tmp_path / "parquets")
        assert "train" in splits
        assert "validation" in splits
        assert len(splits["train"]) == 1
        assert len(splits["validation"]) == 1


class TestMultiGeneratorManifests:
    def _build_multi_gen_records(self) -> list[dict]:
        records = []
        counter = 0
        gens = ["sd15", "midjourney", "wukong", "glide", "vqdm", "biggan", "adm"]

        # Raw train: 140 real, 20 fake per generator (total 140 fake)
        for i in range(140):
            records.append({
                "sample_id": f"train_real_{i}",
                "image_path": f"nature/train/nature_{i}.png",
                "label": 0,
                "generator": "nature",
                "raw_split": "train",
                "index": counter,
            })
            counter += 1

        for gen in gens:
            for i in range(20):
                records.append({
                    "sample_id": f"train_{gen}_{i}",
                    "image_path": f"{gen}/train/{gen}_{i}.png",
                    "label": 1,
                    "generator": gen,
                    "raw_split": "train",
                    "index": counter,
                })
                counter += 1

        # Raw validation: 70 real, 10 fake per generator (total 70 fake)
        for i in range(70):
            records.append({
                "sample_id": f"val_real_{i}",
                "image_path": f"nature/val/nature_{i}.png",
                "label": 0,
                "generator": "nature",
                "raw_split": "validation",
                "index": counter,
            })
            counter += 1

        for gen in gens:
            for i in range(10):
                records.append({
                    "sample_id": f"val_{gen}_{i}",
                    "image_path": f"{gen}/val/{gen}_{i}.png",
                    "label": 1,
                    "generator": gen,
                    "raw_split": "validation",
                    "index": counter,
                })
                counter += 1

        return records

    def test_build_logo_manifests_zero_leakage(self, tmp_path: Path):
        records = self._build_multi_gen_records()
        manifest_dir = tmp_path / "logo_manifests"

        manifests = build_logo_manifests(
            extracted_records=records,
            manifest_dir=manifest_dir,
            leave_out_gen="midjourney",
            n_val_per_gen=2,
        )

        train_m = manifests["train"]
        val_m = manifests["val"]
        test_held_out = manifests["test_midjourney"]
        test_seen = manifests["test_in_domain_seen"]

        # 1. Zero generator leakage: midjourney must NEVER appear in train or val
        assert "midjourney" not in {r.generator for r in train_m if r.label == 1}
        assert "midjourney" not in {r.generator for r in val_m if r.label == 1}
        assert_generator_disjoint(train_m, test_held_out)
        assert_generator_disjoint(val_m, test_held_out)

        # 2. Held-out test set contains midjourney fakes and balanced reals
        assert all(r.generator == "midjourney" for r in test_held_out if r.label == 1)
        reals = len(test_held_out.filter(label=0))
        fakes = len(test_held_out.filter(label=1))
        assert reals == fakes and fakes > 0

        # 3. Held-out fakes come from raw validation only, matching the real side.
        #    Mixing in raw-train fakes would let raw-split source be a shortcut feature.
        raw_split_by_id = {r["sample_id"]: r["raw_split"] for r in records}
        assert all(
            raw_split_by_id[r.sample_id] == "validation" for r in test_held_out if r.label == 1
        )
        assert all(
            raw_split_by_id[r.sample_id] == "validation" for r in test_held_out if r.label == 0
        )

        # 4. Label balance across all splits
        for name, m in manifests.items():
            r_cnt = len(m.filter(label=0))
            f_cnt = len(m.filter(label=1))
            assert r_cnt == f_cnt, f"Split {name} unbalanced: {r_cnt} reals vs {f_cnt} fakes"

        # 5. Check files created
        assert (manifest_dir / "train.jsonl").exists()
        assert (manifest_dir / "test_midjourney.jsonl").exists()

    def test_build_all_in_one_manifests(self, tmp_path: Path):
        records = self._build_multi_gen_records()
        manifest_dir = tmp_path / "all_in_one_manifests"

        manifests = build_all_in_one_manifests(
            extracted_records=records,
            manifest_dir=manifest_dir,
            n_val_per_gen=2,
        )

        train_m = manifests["train"]
        val_m = manifests["val"]
        test_combined = manifests["test_all_combined"]

        # 1. All 7 generators present in train, val, and test
        expected_gens = {"sd15", "midjourney", "wukong", "glide", "vqdm", "biggan", "adm"}
        train_fakes_gens = {r.generator for r in train_m if r.label == 1}
        val_fakes_gens = {r.generator for r in val_m if r.label == 1}
        test_fakes_gens = {r.generator for r in test_combined if r.label == 1}

        assert train_fakes_gens == expected_gens
        assert val_fakes_gens == expected_gens
        assert test_fakes_gens == expected_gens

        # 2. Mutually disjoint sample IDs between train, val, test
        train_ids = {r.sample_id for r in train_m}
        val_ids = {r.sample_id for r in val_m}
        test_ids = {r.sample_id for r in test_combined}

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

        # 3. Label balance (1:1)
        for name, m in manifests.items():
            r_cnt = len(m.filter(label=0))
            f_cnt = len(m.filter(label=1))
            assert r_cnt == f_cnt, f"Split {name} unbalanced: {r_cnt} reals vs {f_cnt} fakes"

    def test_protocol_dispatcher_matches_direct_builders(self, tmp_path: Path):
        """The dispatcher must route to the same builders with the same parameters.

        In-kernel manifest generation is only sound if it reproduces locally prepared
        splits exactly, so this pins dispatcher output against the direct builders.
        """

        def signature(manifests):
            return {name: [r.sample_id for r in m] for name, m in manifests.items()}

        records = self._build_multi_gen_records()

        assert signature(build_protocol_manifests(
            extracted_records=records, manifest_dir=tmp_path / "b", experiment="single", seed=42,
        )) == signature(build_single_generator_manifests(
            extracted_records=records, manifest_dir=tmp_path / "a", train_generator="sd15", seed=42,
        ))

        folds = build_all_logo_manifests(extracted_records=records, manifest_dir=tmp_path / "c", seed=42)
        assert signature(build_protocol_manifests(
            extracted_records=records, manifest_dir=tmp_path / "d",
            experiment="logo", leave_out_gen="midjourney", seed=42,
        )) == signature(folds["midjourney"])

        assert signature(build_protocol_manifests(
            extracted_records=records, manifest_dir=tmp_path / "f", experiment="all7", seed=42,
        )) == signature(build_all7_manifests(
            extracted_records=records, manifest_dir=tmp_path / "e", seed=42,
        ))

        # Files must land in the protocol subdirectories the entrypoints resolve from.
        assert (tmp_path / "b" / "single" / "train.jsonl").is_file()
        assert (tmp_path / "d" / "logo" / "leave_midjourney" / "train.jsonl").is_file()
        assert (tmp_path / "f" / "all7" / "train.jsonl").is_file()

    def test_protocol_dispatcher_rejects_unknown_protocol_or_fold(self, tmp_path: Path):
        records = self._build_multi_gen_records()
        with pytest.raises(ValueError, match="Unsupported experiment protocol"):
            build_protocol_manifests(records, tmp_path, experiment="nope")
        with pytest.raises(ValueError, match="held-out generator"):
            build_protocol_manifests(records, tmp_path, experiment="logo", leave_out_gen="sd14")

    def test_build_single_generator_manifests(self, tmp_path: Path):
        records = self._build_multi_gen_records()
        manifest_dir = tmp_path / "single_manifests"

        manifests = build_single_generator_manifests(
            extracted_records=records,
            manifest_dir=manifest_dir,
            train_generator="sd15",
            val_in_domain_ratio=0.5,
            seed=42,
        )

        train_m = manifests["train"]
        val_m = manifests["val"]
        test_in_domain_m = manifests["in_domain_test"]
        cross_ood_m = manifests["cross_generator_ood"]

        # 1. Generator membership: SD1.5 in train/val/in_domain, remaining 6 in cross OOD
        assert {r.generator for r in train_m if r.label == 1} == {"sd15"}
        assert {r.generator for r in val_m if r.label == 1} == {"sd15"}
        assert {r.generator for r in test_in_domain_m if r.label == 1} == {"sd15"}
        expected_ood = {"adm", "biggan", "glide", "midjourney", "vqdm", "wukong"}
        assert {r.generator for r in cross_ood_m if r.label == 1} == expected_ood

        # 2. Zero generator leakage
        assert_generator_disjoint(train_m, cross_ood_m)
        assert_generator_disjoint(val_m, cross_ood_m)

        # 3. Sample ID disjointness
        train_ids = {r.sample_id for r in train_m}
        val_ids = {r.sample_id for r in val_m}
        test_ids = {r.sample_id for r in test_in_domain_m}
        ood_ids = {r.sample_id for r in cross_ood_m}

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert train_ids.isdisjoint(ood_ids)
        assert val_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(ood_ids)
        assert test_ids.isdisjoint(ood_ids)

        # 4. Label balance (1:1)
        for name, m in manifests.items():
            r_cnt = len(m.filter(label=0))
            f_cnt = len(m.filter(label=1))
            assert r_cnt == f_cnt, f"Split {name} unbalanced: {r_cnt} reals vs {f_cnt} fakes"

        # 5. Check files on disk
        assert (manifest_dir / "train.jsonl").exists()
        assert (manifest_dir / "val.jsonl").exists()
        assert (manifest_dir / "in_domain_test.jsonl").exists()
        assert (manifest_dir / "cross_generator_ood.jsonl").exists()
        for gid in expected_ood:
            assert (manifest_dir / f"test_{gid}.jsonl").exists()

    @pytest.mark.parametrize("leave_out_gen", [
        "sd15", "adm", "biggan", "glide", "midjourney", "vqdm", "wukong"
    ])
    def test_build_logo_manifests_all_seven_generators(self, tmp_path: Path, leave_out_gen: str):
        records = self._build_multi_gen_records()
        manifest_dir = tmp_path / f"logo_{leave_out_gen}"

        manifests = build_logo_manifests(
            extracted_records=records,
            manifest_dir=manifest_dir,
            leave_out_gen=leave_out_gen,
            n_val_per_gen=2,
            seed=42,
        )

        train_m = manifests["train"]
        val_m = manifests["val"]
        test_held_out = manifests[f"test_{leave_out_gen}"]

        # 1. leave_out_gen NEVER in train or val
        train_fakes = {r.generator for r in train_m if r.label == 1}
        val_fakes = {r.generator for r in val_m if r.label == 1}
        assert leave_out_gen not in train_fakes
        assert leave_out_gen not in val_fakes

        # 2. Held-out test set contains exclusively leave_out_gen fakes
        held_out_fakes = {r.generator for r in test_held_out if r.label == 1}
        assert held_out_fakes == {leave_out_gen}

        # 3. Generator disjointness assertion
        assert_generator_disjoint(train_m, test_held_out)
        assert_generator_disjoint(val_m, test_held_out)

        # 4. Sample ID disjointness
        train_ids = {r.sample_id for r in train_m}
        val_ids = {r.sample_id for r in val_m}
        held_out_ids = {r.sample_id for r in test_held_out}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(held_out_ids)
        assert val_ids.isdisjoint(held_out_ids)

        # 5. Label balance (1:1)
        for name, m in manifests.items():
            r_cnt = len(m.filter(label=0))
            f_cnt = len(m.filter(label=1))
            assert r_cnt == f_cnt, f"Split {name} unbalanced: {r_cnt} reals vs {f_cnt} fakes"

        # 6. File on disk
        assert (manifest_dir / f"test_{leave_out_gen}.jsonl").exists()

    def test_build_logo_manifests_invalid_generator_raises(self):
        records = self._build_multi_gen_records()
        with pytest.raises(ValueError, match="Invalid leave_out_gen"):
            build_logo_manifests(records, leave_out_gen="invalid_gen")
        with pytest.raises(ValueError, match="Invalid leave_out_gen"):
            build_logo_manifests(records, leave_out_gen="sd14")

    def test_builders_reject_partial_generator_set(self):
        """A missing generator must fail loudly instead of silently redefining the protocol."""
        records = self._build_multi_gen_records()
        partial = [r for r in records if r.get("generator") != "wukong"]

        with pytest.raises(ValueError, match="canonical generator set"):
            build_logo_manifests(partial, leave_out_gen="midjourney")
        with pytest.raises(ValueError, match="canonical generator set"):
            build_all7_manifests(partial)
        with pytest.raises(ValueError, match="canonical generator set"):
            build_all_in_one_manifests(partial)

    def test_build_all7_manifests(self, tmp_path: Path):
        records = self._build_multi_gen_records()
        manifest_dir = tmp_path / "all7_manifests"

        manifests = build_all7_manifests(
            extracted_records=records,
            manifest_dir=manifest_dir,
            n_val_per_gen=2,
            seed=42,
        )

        train_m = manifests["train"]
        val_m = manifests["val"]
        test_combined = manifests["test_all_combined"]

        expected_gens = {"sd15", "midjourney", "wukong", "glide", "vqdm", "biggan", "adm"}
        assert {r.generator for r in train_m if r.label == 1} == expected_gens
        assert {r.generator for r in val_m if r.label == 1} == expected_gens
        assert {r.generator for r in test_combined if r.label == 1} == expected_gens

        # Sample IDs are mutually disjoint
        train_ids = {r.sample_id for r in train_m}
        val_ids = {r.sample_id for r in val_m}
        test_ids = {r.sample_id for r in test_combined}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

        # Check per-generator test files
        assert (manifest_dir / "test_all_combined.jsonl").exists()
        for gen in expected_gens:
            assert (manifest_dir / f"test_{gen}.jsonl").exists()

    def test_determinism_with_same_seed(self):
        records = self._build_multi_gen_records()
        run1 = build_single_generator_manifests(records, seed=42)
        run2 = build_single_generator_manifests(records, seed=42)

        for split in ("train", "val", "in_domain_test", "cross_generator_ood"):
            ids1 = [r.sample_id for r in run1[split]]
            ids2 = [r.sample_id for r in run2[split]]
            assert ids1 == ids2, f"Split {split} is not deterministic across runs with same seed"

    def test_build_all_experiment_manifests_hierarchy(self, tmp_path: Path):
        records = self._build_multi_gen_records()
        base_dir = tmp_path / "manifests"

        manifests_by_protocol = build_all_experiment_manifests(
            extracted_records=records,
            output_dir=base_dir,
            n_val_per_gen=2,
            seed=42,
        )

        assert "single" in manifests_by_protocol
        assert "logo" in manifests_by_protocol
        assert "all7" in manifests_by_protocol

        # Check directory hierarchy
        assert (base_dir / "single" / "train.jsonl").exists()
        assert (base_dir / "single" / "val.jsonl").exists()
        assert (base_dir / "single" / "in_domain_test.jsonl").exists()
        assert (base_dir / "single" / "cross_generator_ood.jsonl").exists()

        for gen in KAGGLE_TINY_GENIMAGE_GENERATORS:
            assert (base_dir / "logo" / f"leave_{gen}" / "train.jsonl").exists()
            assert (base_dir / "logo" / f"leave_{gen}" / f"test_{gen}.jsonl").exists()

        assert (base_dir / "all7" / "train.jsonl").exists()
        assert (base_dir / "all7" / "val.jsonl").exists()
        assert (base_dir / "all7" / "test_all_combined.jsonl").exists()

    def test_class_id_from_path_extraction(self):
        from forensight.data.tiny_genimage import _class_id_from_path

        assert _class_id_from_path(Path("imagenet_ai_0424_sdv5/train/ai/001_sdv5_00094.png")) == "c0001"
        assert _class_id_from_path(Path("imagenet_ai_0508_adm/val/ai/056_adm_00012.png")) == "c0056"
        assert _class_id_from_path(Path("imagenet_ai_0424_sdv5/train/nature/n01440764_12548.JPEG")) == "n01440764"
        assert _class_id_from_path(Path("some/folder/n02123045/image_123.jpg")) == "n02123045"
        assert _class_id_from_path(Path("imagenet_ai_0424_sdv5/val/nature/ILSVRC2012_val_00000091.JPEG")) is None

    def test_class_id_preserved_in_manifests(self):
        records = self._build_multi_gen_records()
        for idx, r in enumerate(records):
            r["class_id"] = f"c{idx % 5:04d}"

        manifests = build_single_generator_manifests(records)
        for split_name in ("train", "val", "in_domain_test"):
            m = manifests[split_name]
            class_ids = {rec.class_id for rec in m}
            assert None not in class_ids, f"Split {split_name} has None class_id"

    def test_logo_zero_train_reals_leakage(self):
        records = self._build_multi_gen_records()
        # Ensure distinct sample IDs
        train_nature_ids = {r["sample_id"] for r in records if r["raw_split"] == "train" and r["label"] == 0}

        manifests = build_logo_manifests(records, leave_out_gen="midjourney", n_val_per_gen=2)
        test_held_out = manifests["test_midjourney"]
        train_m = manifests["train"]

        # 1. Zero sample leakage between train and test_held_out
        test_held_out_ids = {rec.sample_id for rec in test_held_out}
        train_ids = {rec.sample_id for rec in train_m}
        assert train_ids.isdisjoint(test_held_out_ids)

        # 2. Crucial: NO real image in test_held_out may come from train_nature
        test_held_out_real_ids = {rec.sample_id for rec in test_held_out if rec.label == 0}
        assert test_held_out_real_ids.isdisjoint(train_nature_ids), (
            "Leakage detected: test_held_out contains real images from train_nature!"
        )

    def test_scale_tier_subsampling_with_class_id(self):
        from forensight.data.split import subsample_manifest_by_class

        records = self._build_multi_gen_records()
        for idx, r in enumerate(records):
            r["class_id"] = f"c{idx % 5:04d}"

        manifests = build_single_generator_manifests(records)
        train_m = manifests["train"]

        # Subsample max 1 real, 1 fake per class
        smoke_m = subsample_manifest_by_class(train_m, max_per_class_real=1, max_per_class_fake=1, seed=42)
        assert len(smoke_m) > 2, "Smoke subsampling failed to preserve multiple classes"
        smoke_classes = {r.class_id for r in smoke_m}
        assert len(smoke_classes) > 1, "Expected multiple classes in smoke manifest"

    def test_single_generator_manifests_seed_variation(self):
        records = self._build_multi_gen_records()
        for idx, r in enumerate(records):
            r["class_id"] = f"c{idx % 4:04d}"

        m_42 = build_single_generator_manifests(records, seed=42)
        m_1337 = build_single_generator_manifests(records, seed=1337)

        val_ids_42 = {r.sample_id for r in m_42["val"]}
        val_ids_1337 = {r.sample_id for r in m_1337["val"]}

        # Seeds 42 and 1337 must yield different sample splits
        assert val_ids_42 != val_ids_1337, "Validation sample IDs must vary with seed"

        test_ids_42 = {r.sample_id for r in m_42["in_domain_test"]}
        test_ids_1337 = {r.sample_id for r in m_1337["in_domain_test"]}
        assert test_ids_42 != test_ids_1337, "In-domain test sample IDs must vary with seed"

    def test_split_records_by_class_and_seed_balance(self):
        from forensight.data.tiny_genimage import split_records_by_class_and_seed

        # 4 classes, 10 samples each = 40 samples
        records = [
            {"sample_id": f"s_{i}", "class_id": f"class_{i % 4}", "generator": "sd15"}
            for i in range(40)
        ]

        val_part, test_part = split_records_by_class_and_seed(records, val_ratio=0.5, seed=42)
        assert len(val_part) == 20
        assert len(test_part) == 20

        # Disjoint
        val_ids = {r["sample_id"] for r in val_part}
        test_ids = {r["sample_id"] for r in test_part}
        assert val_ids.isdisjoint(test_ids)

        # Class balanced: each class should have exactly 5 in val and 5 in test
        for c in range(4):
            val_c = [r for r in val_part if r["class_id"] == f"class_{c}"]
            test_c = [r for r in test_part if r["class_id"] == f"class_{c}"]
            assert len(val_c) == 5
            assert len(test_c) == 5

        # Different seed yields different sample IDs
        val_part_other, _ = split_records_by_class_and_seed(records, val_ratio=0.5, seed=1337)
        val_ids_other = {r["sample_id"] for r in val_part_other}
        assert val_ids != val_ids_other

