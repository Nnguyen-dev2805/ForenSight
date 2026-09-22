"""Unit tests for Tiny-GenImage dataset extraction and manifest generation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forensight.data.split import assert_generator_disjoint, validate_no_leakage
from forensight.data.tiny_genimage import (
    CROSS_GENERATOR_OOD_IDS,
    TINY_GENIMAGE_GENERATOR_MAP,
    build_tiny_genimage_manifests,
    detect_image_extension,
    download_tiny_genimage_parquets,
    extract_image_bytes,
    extract_parquet_images,
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
        # generator: 0 for nature (even), 5 for SD1.4 (odd)
        generators = [0 if i % 2 == 0 else 5 for i in range(n_rows)]

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
                assert rec["generator"] == "sd14"


class TestManifestConstructionAndLeakageInvariants:
    def _build_synthetic_dataset_records(self) -> list[dict]:
        """Build synthetic extracted metadata for all 8 generators and real images."""
        records = []
        counter = 0

        # Raw train: 20 real, 10 SD1.4 fake, 10 Midjourney fake
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
                "sample_id": f"train_sd14_{i}",
                "image_path": f"sd14/train/sd14_{i}.png",
                "label": 1,
                "generator": "sd14",
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

        # Raw validation: 40 real, 8 fake per each generator (SD1.4, SD1.5, + 6 OODs)
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

        # SD1.4 validation fakes (6 images: 3 for val, 3 for in_domain_test)
        for i in range(6):
            records.append({
                "sample_id": f"val_sd14_{i}",
                "image_path": f"sd14/val/sd14_{i}.png",
                "label": 1,
                "generator": "sd14",
                "raw_split": "validation",
                "index": counter,
            })
            counter += 1

        # SD1.5 near-OOD validation fakes (4 images)
        for i in range(4):
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
        near_ood_m = manifests["tiny_genimage_near_ood"]
        cross_ood_m = manifests["tiny_genimage_cross_generator_ood"]

        # 1. Zero generator leakage
        assert_generator_disjoint(train_m, cross_ood_m)
        assert_generator_disjoint(train_m, near_ood_m)
        assert_generator_disjoint(val_m, cross_ood_m)
        assert_generator_disjoint(val_m, near_ood_m)

        # Train contains ONLY sd14 and nature (midjourney from raw train MUST be excluded)
        train_generators = {r.generator for r in train_m}
        assert train_generators == {"nature", "sd14"}

        # 2. Zero sample leakage between val and in_domain_test
        val_ids = {r.sample_id for r in val_m}
        test_in_domain_ids = {r.sample_id for r in test_in_domain_m}
        assert val_ids.isdisjoint(test_in_domain_ids)

        # Zero sample leakage between train and val/test
        train_ids = {r.sample_id for r in train_m}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_in_domain_ids)

        # Zero sample leakage between evaluation splits' real images
        near_ood_ids = {r.sample_id for r in near_ood_m}
        cross_ood_ids = {r.sample_id for r in cross_ood_m}
        assert val_ids.isdisjoint(near_ood_ids)
        assert val_ids.isdisjoint(cross_ood_ids)
        assert test_in_domain_ids.isdisjoint(near_ood_ids)
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
            "tiny_genimage_near_ood",
            "tiny_genimage_cross_generator_ood",
        ):
            jsonl_path = manifest_dir / f"{name}.jsonl"
            assert jsonl_path.exists()
            assert jsonl_path.stat().st_size > 0


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
