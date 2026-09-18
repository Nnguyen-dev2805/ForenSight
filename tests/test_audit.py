"""Comprehensive tests for ForenSight dataset audit, leakage checks, and bias quantification."""

import io
import json
from pathlib import Path
import pytest
from PIL import Image

import sys

from forensight.data.audit import (
    AuditReport,
    ImageAttributes,
    audit_manifest_images,
    audit_manifest_leakage,
    calculate_class_balance,
    calculate_distribution_stats,
    calculate_format_stats,
    calculate_jpeg_quality_buckets,
    calculate_resolution_counts,
    calculate_source_balance,
    compute_dhash,
    compute_file_sha256,
    detect_watermark_shortcuts,
    estimate_jpeg_quality,
    extract_image_attributes,
    generate_audit_markdown,
    hamming_distance,
)
from forensight.data.split import Manifest, ManifestRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import scripts.check_dataset as cli


@pytest.fixture
def synthetic_image_dir(tmp_path: Path) -> Path:
    """Fixture creating a controlled set of synthetic images with varied attributes."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real images: varied sizes, JPEG quality 75
    for i in range(5):
        w = 400 + i * 30
        h = 300 + i * 20
        im = Image.new("RGB", (w, h), color=(100 + i * 10, 50 + i * 5, 200 - i * 10))
        im.save(img_dir / f"real_{i}.jpg", format="JPEG", quality=75)

    # 2. Fake images: fixed 512x512, JPEG quality 95 (literature confound)
    for i in range(5):
        im = Image.new("RGB", (512, 512), color=(50 + i * 10, 150 + i * 5, 100 + i * 10))
        im.save(img_dir / f"fake_{i}.jpg", format="JPEG", quality=95)

    # 3. Exactly identical images (for SHA256 duplicate testing)
    dup_im = Image.new("RGB", (256, 256), color=(42, 42, 42))
    dup_im.save(img_dir / "dup_original.jpg", format="JPEG", quality=80)
    dup_im.save(img_dir / "dup_copy.jpg", format="JPEG", quality=80)

    # 4. Near-duplicate images (same content, slight resize)
    near_im1 = Image.new("RGB", (300, 300), color=(123, 123, 123))
    near_im2 = Image.new("RGB", (310, 310), color=(123, 123, 123))
    near_im1.save(img_dir / "near_orig.png", format="PNG")
    near_im2.save(img_dir / "near_scale.png", format="PNG")

    # 5. PNG format images
    png_im = Image.new("RGB", (200, 200), color=(20, 200, 20))
    png_im.save(img_dir / "fake_png.png", format="PNG")

    return img_dir


class TestImageAttributeExtraction:
    """Unit tests for low-level image inspection and attribute extraction helpers."""

    def test_estimate_jpeg_quality(self):
        for target_q in (50, 75, 85, 95):
            buf = io.BytesIO()
            im = Image.new("RGB", (128, 128), (120, 130, 140))
            im.save(buf, format="JPEG", quality=target_q)
            buf.seek(0)
            with Image.open(buf) as loaded:
                est_q = estimate_jpeg_quality(loaded)
                assert est_q is not None
                assert abs(est_q - target_q) <= 3

    def test_estimate_jpeg_quality_non_jpeg_returns_none(self):
        im = Image.new("RGB", (64, 64), (10, 20, 30))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        with Image.open(buf) as loaded:
            assert estimate_jpeg_quality(loaded) is None

    def test_compute_dhash_and_hamming_distance(self):
        im1 = Image.new("RGB", (100, 100), (255, 0, 0))
        im2 = Image.new("RGB", (200, 200), (255, 0, 0))
        h1 = compute_dhash(im1)
        h2 = compute_dhash(im2)
        assert len(h1) == 16
        assert hamming_distance(h1, h2) == 0

        # Different patterns yield non-zero hamming distance
        im3 = Image.new("RGB", (100, 100), (0, 0, 0))
        for x in range(100):
            for y in range(100):
                if (x // 10) % 2 == 0:
                    im3.putpixel((x, y), (255, 255, 255))
        h3 = compute_dhash(im3)
        assert hamming_distance(h1, h3) > 0

    def test_compute_file_sha256(self, tmp_path: Path):
        test_file = tmp_path / "test.bin"
        data = b"ForenSight test content 12345"
        test_file.write_bytes(data)
        import hashlib
        expected = hashlib.sha256(data).hexdigest()
        assert compute_file_sha256(test_file) == expected

    def test_extract_image_attributes_success(self, synthetic_image_dir: Path):
        rec = ManifestRecord(
            sample_id="s1",
            image_path=str(synthetic_image_dir / "real_0.jpg"),
            label=0,
            dataset="genimage",
            generator="nature",
            split="train",
            class_id="n01440764",
        )
        attrs = extract_image_attributes(rec)
        assert attrs.error is None
        assert attrs.width == 400
        assert attrs.height == 300
        assert attrs.format == "JPEG"
        assert attrs.jpeg_quality is not None
        assert abs(attrs.jpeg_quality - 75) <= 3
        assert len(attrs.sha256) == 64
        assert attrs.phash is not None

    def test_extract_image_attributes_missing_file_handles_gracefully(self, tmp_path: Path):
        rec = ManifestRecord(
            sample_id="s_missing",
            image_path=str(tmp_path / "nonexistent.jpg"),
            label=1,
            dataset="genimage",
            generator="sd14",
            split="train",
        )
        attrs = extract_image_attributes(rec)
        assert attrs.error is not None
        assert "File not found" in attrs.error
        assert attrs.width == 0
        assert attrs.format == "UNKNOWN"

    def test_extract_image_attributes_corrupt_file_handles_gracefully(self, tmp_path: Path):
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a real jpeg file")
        rec = ManifestRecord(
            sample_id="s_corrupt",
            image_path=str(corrupt_path),
            label=1,
            dataset="genimage",
            generator="sd14",
            split="train",
        )
        attrs = extract_image_attributes(rec)
        assert attrs.error is not None
        assert "Error reading image" in attrs.error
        assert attrs.format == "CORRUPT"

    def test_detect_watermark_shortcuts_metadata(self):
        im = Image.new("RGB", (100, 100), (128, 128, 128))
        im.info["comment"] = "Copyright Shutterstock 2024"
        indicators = detect_watermark_shortcuts(im)
        assert any("shutterstock" in ind for ind in indicators)

    def test_detect_watermark_shortcuts_letterbox(self):
        # Create image with texture in center and solid black bars on top and bottom
        im = Image.new("RGB", (100, 100), (0, 0, 0))
        # Center rows 20 to 80 textured
        for y in range(20, 80):
            for x in range(100):
                im.putpixel((x, y), ((x * 5) % 255, (y * 5) % 255, 100))
        indicators = detect_watermark_shortcuts(im)
        assert any("letterbox" in ind for ind in indicators)

    def test_detect_watermark_shortcuts_clean(self):
        # Clean uniform or textured image without border
        im = Image.new("RGB", (100, 100), (128, 128, 128))
        indicators = detect_watermark_shortcuts(im)
        assert len(indicators) == 0


class TestDistributionStatistics:
    """Unit tests for summary statistics and binning functions."""

    def test_calculate_distribution_stats_empty(self):
        stats = calculate_distribution_stats([])
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["median"] is None

    def test_calculate_distribution_stats_values(self):
        vals = [10, 20, 30, 40, 50]
        stats = calculate_distribution_stats(vals)
        assert stats["count"] == 5
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["p25"] == 20.0
        assert stats["p75"] == 40.0
        assert stats["iqr"] == 20.0

    def test_calculate_jpeg_quality_buckets(self):
        qualities = [30, 45, 60, 75, 85, 92, 98, 100]
        buckets = calculate_jpeg_quality_buckets(qualities)
        assert buckets["<50"] == 2
        assert buckets["50-70"] == 1
        assert buckets["71-80"] == 1
        assert buckets["81-90"] == 1
        assert buckets["91-95"] == 1
        assert buckets["96-100"] == 2

    def test_calculate_class_balance(self):
        records = [
            ManifestRecord(sample_id="1", image_path="/1.jpg", label=0, dataset="g", generator="n", split="train", class_id="c0"),
            ManifestRecord(sample_id="2", image_path="/2.jpg", label=0, dataset="g", generator="n", split="train", class_id="c0"),
            ManifestRecord(sample_id="3", image_path="/3.jpg", label=1, dataset="g", generator="sd14", split="train", class_id="c0"),
            ManifestRecord(sample_id="4", image_path="/4.jpg", label=0, dataset="g", generator="n", split="train", class_id="c1"),
        ]
        cb = calculate_class_balance(records)
        assert cb["num_classes"] == 2
        assert cb["by_class"]["c0"]["real"] == 2
        assert cb["by_class"]["c0"]["fake"] == 1
        assert cb["by_class"]["c1"]["fake"] == 0
        assert "c1" in cb["classes_missing_fake"]


class TestAuditLeakage:
    """Unit tests for audit_manifest_leakage across splits."""

    def test_clean_split_audit(self):
        train = Manifest([
            ManifestRecord(sample_id="t1", image_path="/p1.jpg", label=0, dataset="g", generator="nature", split="train"),
            ManifestRecord(sample_id="t2", image_path="/p2.jpg", label=1, dataset="g", generator="sd14", split="train"),
        ])
        eval_m = Manifest([
            ManifestRecord(sample_id="e1", image_path="/p3.jpg", label=0, dataset="g", generator="nature", split="val"),
            ManifestRecord(sample_id="e2", image_path="/p4.jpg", label=1, dataset="g", generator="sd14", split="val"),
        ])
        res = audit_manifest_leakage(train, eval_m)
        assert res["has_leakage"] is False
        assert len(res["sample_id_collisions"]) == 0
        assert len(res["image_path_collisions"]) == 0

    def test_sample_and_path_leakage_detected(self):
        train = Manifest([
            ManifestRecord(sample_id="shared_id", image_path="/shared/path.jpg", label=1, dataset="g", generator="sd14", split="train"),
        ])
        eval_m = Manifest([
            ManifestRecord(sample_id="shared_id", image_path="/shared/path.jpg", label=1, dataset="g", generator="sd14", split="test"),
        ])
        res = audit_manifest_leakage(train, {"test_split": eval_m})
        assert res["has_leakage"] is True
        assert len(res["sample_id_collisions"]) == 1
        assert len(res["image_path_collisions"]) == 1

    def test_cross_generator_ood_leakage_detected(self):
        train = Manifest([
            ManifestRecord(sample_id="t1", image_path="/t1.jpg", label=1, dataset="g", generator="sd14", split="train"),
        ])
        ood = Manifest([
            ManifestRecord(sample_id="o1", image_path="/o1.jpg", label=1, dataset="g", generator="sd14", split="cross_generator_ood"),
        ])
        res = audit_manifest_leakage(train, {"cross_gen": ood}, check_generators=True)
        assert res["has_leakage"] is True
        assert len(res["generator_overlaps"]) == 1
        assert res["generator_overlaps"][0]["overlapping_generators"] == ["sd14"]

    def test_near_ood_generator_leakage_detected(self):
        train = Manifest([
            ManifestRecord(sample_id="t1", image_path="/t1.jpg", label=1, dataset="g", generator="sd14", split="train"),
        ])
        near_ood = Manifest([
            ManifestRecord(sample_id="n1", image_path="/n1.jpg", label=1, dataset="g", generator="sd14", split="near_ood"),
        ])
        res = audit_manifest_leakage(train, {"near_ood": near_ood}, check_generators=True)
        assert res["has_leakage"] is True
        assert len(res["generator_overlaps"]) == 1
        assert res["generator_overlaps"][0]["overlapping_generators"] == ["sd14"]

    def test_audit_manifest_leakage_with_base_dir(self, tmp_path: Path):
        train = Manifest([
            ManifestRecord(sample_id="t1", image_path="images/t1.jpg", label=1, dataset="g", generator="sd14", split="train"),
        ])
        eval_m = Manifest([
            ManifestRecord(sample_id="e1", image_path="images/t1.jpg", label=1, dataset="g", generator="sd15", split="val"),
        ])
        # Resolving relative paths with base_dir should flag image path collision
        res = audit_manifest_leakage(train, {"val": eval_m}, base_dir=tmp_path)
        assert res["has_leakage"] is True
        assert len(res["image_path_collisions"]) == 1
        expected_path = str(tmp_path / "images/t1.jpg")
        assert expected_path in res["image_path_collisions"][0]["colliding_paths"]


class TestAuditManifestImages:
    """Integration unit tests for audit_manifest_images and literature confound detection."""

    def test_audit_manifest_images_full(self, synthetic_image_dir: Path):
        records = []
        # Reals
        for i in range(5):
            records.append(
                ManifestRecord(
                    sample_id=f"real_{i}",
                    image_path=str(synthetic_image_dir / f"real_{i}.jpg"),
                    label=0,
                    dataset="genimage",
                    generator="nature",
                    split="train",
                    class_id=f"c_{i % 2}",
                )
            )
        # Fakes (512x512, Q=95)
        for i in range(5):
            records.append(
                ManifestRecord(
                    sample_id=f"fake_{i}",
                    image_path=str(synthetic_image_dir / f"fake_{i}.jpg"),
                    label=1,
                    dataset="genimage",
                    generator="sd14",
                    split="train",
                    class_id=f"c_{i % 2}",
                )
            )
        # Duplicate pair
        records.append(
            ManifestRecord(
                sample_id="dup_1",
                image_path=str(synthetic_image_dir / "dup_original.jpg"),
                label=0,
                dataset="genimage",
                generator="nature",
                split="train",
                class_id="c_dup",
            )
        )
        records.append(
            ManifestRecord(
                sample_id="dup_2",
                image_path=str(synthetic_image_dir / "dup_copy.jpg"),
                label=0,
                dataset="genimage",
                generator="nature",
                split="train",
                class_id="c_dup",
            )
        )

        manifest = Manifest(records)
        report = audit_manifest_images(manifest, max_workers=2, near_duplicate_threshold=2)

        assert report.total_samples == 12
        assert report.num_real == 7
        assert report.num_fake == 5

        # Check exact duplicates
        assert len(report.duplicate_check) == 1
        dup = report.duplicate_check[0]
        assert dup["count"] == 2
        assert "dup_1" in dup["sample_ids"]
        assert "dup_2" in dup["sample_ids"]
        assert dup["cross_label"] is False

        # Check literature confounds flagged in findings:
        findings_map = {f["check"]: f for f in report.summary_findings}

        # Resolution disparity should be flagged (fakes are 512x512 std=0, reals have varied widths)
        assert "resolution_disparity" in findings_map
        assert findings_map["resolution_disparity"]["status"] == "WARNING"

        # JPEG quality disparity should be flagged (fakes Q=95 vs reals Q=75)
        assert "jpeg_quality_disparity" in findings_map
        assert findings_map["jpeg_quality_disparity"]["status"] == "WARNING"

        # Report serializability
        d = report.to_dict()
        assert d["total_samples"] == 12
        json_str = report.to_json()
        assert "resolution_disparity" in json_str

        # Markdown generation
        md = report.generate_markdown()
        assert "# ForenSight Dataset Audit Report" in md
        assert "Literature Confound" in md
        assert "JPEG Compression & Quality Distribution" in md

    def test_critical_cross_label_duplicate_detection(self, synthetic_image_dir: Path):
        # Same image path labeled as real in one record, fake in another
        records = [
            ManifestRecord(sample_id="r1", image_path=str(synthetic_image_dir / "dup_original.jpg"), label=0, dataset="g", generator="nature", split="train"),
            ManifestRecord(sample_id="f1", image_path=str(synthetic_image_dir / "dup_copy.jpg"), label=1, dataset="g", generator="sd14", split="train"),
        ]
        report = audit_manifest_images(records)
        assert report.has_critical_findings()
        assert report.passed() is False
        dup_finding = next(f for f in report.summary_findings if f["check"] == "exact_duplicates")
        assert dup_finding["status"] == "FAIL"
        assert dup_finding["severity"] == "CRITICAL"


class TestCLIExecution:
    """Unit tests for scripts/check_dataset.py CLI."""

    def test_cli_execution_on_manifest(self, synthetic_image_dir: Path, tmp_path: Path):
        manifest_path = tmp_path / "test_manifest.jsonl"
        records = [
            ManifestRecord(sample_id="r1", image_path=str(synthetic_image_dir / "real_0.jpg"), label=0, dataset="g", generator="nature", split="train"),
            ManifestRecord(sample_id="f1", image_path=str(synthetic_image_dir / "fake_0.jpg"), label=1, dataset="g", generator="sd14", split="train"),
        ]
        Manifest(records).to_jsonl(manifest_path)

        out_json = tmp_path / "report.json"
        out_md = tmp_path / "report.md"

        exit_code = cli.main([
            "--manifest", str(manifest_path),
            "--output-json", str(out_json),
            "--output-md", str(out_md),
            "--quiet",
        ])
        assert exit_code == 0
        assert out_json.exists()
        assert out_md.exists()

        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["total_samples"] == 2

    def test_cli_strict_failure_on_leakage(self, synthetic_image_dir: Path, tmp_path: Path):
        train_path = tmp_path / "train.jsonl"
        eval_path = tmp_path / "eval.jsonl"

        train_records = [
            ManifestRecord(sample_id="leaked_id", image_path=str(synthetic_image_dir / "real_0.jpg"), label=0, dataset="g", generator="nature", split="train"),
        ]
        eval_records = [
            ManifestRecord(sample_id="leaked_id", image_path=str(synthetic_image_dir / "real_1.jpg"), label=0, dataset="g", generator="nature", split="test"),
        ]
        Manifest(train_records).to_jsonl(train_path)
        Manifest(eval_records).to_jsonl(eval_path)

        # Strict mode must return non-zero on sample ID leakage
        exit_code = cli.main([
            "--train-manifest", str(train_path),
            "--eval-manifest", str(eval_path),
            "--strict",
            "--quiet",
        ])
        assert exit_code == 1

    def test_cli_missing_manifest_returns_error(self):
        exit_code = cli.main([])
        assert exit_code == 2

    def test_cli_base_dir_and_sample_size(self, synthetic_image_dir: Path, tmp_path: Path):
        manifest_path = tmp_path / "relative_manifest.jsonl"
        records = [
            ManifestRecord(sample_id=f"r_{i}", image_path=f"real_{i}.jpg", label=0, dataset="g", generator="nature", split="train")
            for i in range(5)
        ]
        Manifest(records).to_jsonl(manifest_path)

        out_json = tmp_path / "sampled_report.json"
        exit_code = cli.main([
            "--manifest", str(manifest_path),
            "--base-dir", str(synthetic_image_dir),
            "--sample-size", "3",
            "--output-json", str(out_json),
            "--quiet",
        ])
        assert exit_code == 0
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["total_samples"] == 3


class TestAuditEdgeCasesAndFeatures:
    """Tests for near duplicates, format disparity, empty manifests, and report persistence."""

    def test_near_duplicate_detection(self, synthetic_image_dir: Path):
        # near_orig.png and near_scale.png have distance 0 or <= 2
        records = [
            ManifestRecord(sample_id="near_1", image_path=str(synthetic_image_dir / "near_orig.png"), label=0, dataset="g", generator="nature", split="train"),
            ManifestRecord(sample_id="near_2", image_path=str(synthetic_image_dir / "near_scale.png"), label=0, dataset="g", generator="nature", split="train"),
        ]
        report = audit_manifest_images(records, compute_phash=True, near_duplicate_threshold=2)
        assert len(report.near_duplicate_check) >= 1
        entry = report.near_duplicate_check[0]
        assert "near_1" in entry["sample_ids"]
        assert "near_2" in entry["sample_ids"]
        assert entry["distance"] <= 2

    def test_format_disparity_detection(self, synthetic_image_dir: Path):
        records = [
            ManifestRecord(sample_id="r1", image_path=str(synthetic_image_dir / "real_0.jpg"), label=0, dataset="g", generator="nature", split="train"),
            ManifestRecord(sample_id="f1", image_path=str(synthetic_image_dir / "fake_png.png"), label=1, dataset="g", generator="sd14", split="train"),
        ]
        report = audit_manifest_images(records)
        finding = next(f for f in report.summary_findings if f["check"] == "format_disparity")
        assert finding["status"] == "WARNING"

    def test_empty_manifest_audit(self):
        report = audit_manifest_images([])
        assert report.total_samples == 0
        assert report.num_real == 0
        assert report.num_fake == 0
        assert report.passed() is True
        md = report.generate_markdown()
        assert "# ForenSight Dataset Audit Report" in md

    def test_save_json_and_markdown(self, synthetic_image_dir: Path, tmp_path: Path):
        records = [
            ManifestRecord(sample_id="r1", image_path=str(synthetic_image_dir / "real_0.jpg"), label=0, dataset="g", generator="nature", split="train"),
        ]
        report = audit_manifest_images(records)

        json_file = tmp_path / "sub" / "report.json"
        md_file = tmp_path / "sub" / "report.md"

        report.save_json(json_file)
        report.save_markdown(md_file)

        assert json_file.exists()
        assert md_file.exists()
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["total_samples"] == 1
        assert "ForenSight Dataset Audit Report" in md_file.read_text(encoding="utf-8")

    def test_near_duplicate_skipped_when_exceeding_limit(self, synthetic_image_dir: Path):
        records = [
            ManifestRecord(sample_id=f"r_{i}", image_path=str(synthetic_image_dir / f"real_{i}.jpg"), label=0, dataset="g", generator="nature", split="train")
            for i in range(5)
        ]
        # Set max_near_duplicate_candidates=3 so 5 records exceeds it and triggers skip
        report = audit_manifest_images(records, compute_phash=True, max_near_duplicate_candidates=3)
        assert report.metadata.get("near_duplicate_skipped") is True
        finding = next(f for f in report.summary_findings if f["check"] == "near_duplicates")
        assert finding["status"] == "SKIPPED"
        assert "exceeds pairwise limit" in finding["message"]

        md = report.generate_markdown()
        assert "SKIPPED (sample size exceeds pairwise limit)" in md

    def test_source_balance_and_correlation_finding(self, synthetic_image_dir: Path):
        # Real images come from dataset 'source_a', fake images from 'source_b'
        records = [
            ManifestRecord(sample_id="r1", image_path=str(synthetic_image_dir / "real_0.jpg"), label=0, dataset="source_a", generator="nature", split="train"),
            ManifestRecord(sample_id="f1", image_path=str(synthetic_image_dir / "fake_0.jpg"), label=1, dataset="source_b", generator="sd14", split="train"),
        ]
        report = audit_manifest_images(records)
        sb = report.source_balance
        assert sb["is_strongly_correlated"] is True
        assert sb["num_sources"] == 2
        assert sb["max_source_disparity"] == 1.0

        finding = next(f for f in report.summary_findings if f["check"] == "source_label_correlation")
        assert finding["status"] == "WARNING"
        assert "Strong source-label correlation detected" in finding["message"]

        md = report.generate_markdown()
        assert "## 7. Dataset Source Distribution & Correlation" in md
        assert "Strong Source-Label Correlation:** YES (WARNING)" in md
        assert "- **Distinct Sources:** 2 (source_a, source_b)" in md
        assert "| `source_a` | 1 (100.0%) | 0 (0.0%) | 1 (50.0%) |" in md
        assert "| `source_b` | 0 (0.0%) | 1 (100.0%) | 1 (50.0%) |" in md

    def test_watermark_shortcuts_audit_finding(self, tmp_path: Path):
        # Create one real image with watermark comment and one clean fake image
        wm_img_path = tmp_path / "watermarked_real.png"
        clean_img_path = tmp_path / "clean_fake.png"

        from PIL import PngImagePlugin

        im_wm = Image.new("RGB", (64, 64), (100, 100, 100))
        meta = PngImagePlugin.PngInfo()
        meta.add_text("comment", "stock watermark")
        im_wm.save(wm_img_path, format="PNG", pnginfo=meta)

        im_clean = Image.new("RGB", (64, 64), (200, 200, 200))
        im_clean.save(clean_img_path)

        records = [
            ManifestRecord(sample_id="wm_real", image_path=str(wm_img_path), label=0, dataset="g", generator="nature", split="train"),
            ManifestRecord(sample_id="clean_fake", image_path=str(clean_img_path), label=1, dataset="g", generator="sd14", split="train"),
        ]
        report = audit_manifest_images(records)
        assert report.watermark_stats["total_flagged"] == 1
        assert report.watermark_stats["real_flagged"] == 1
        assert report.watermark_stats["fake_flagged"] == 0

        finding = next(f for f in report.summary_findings if f["check"] == "watermark_shortcuts")
        assert finding["status"] == "WARNING"
        assert "Watermark shortcut alert" in finding["message"]

        md = report.generate_markdown()
        assert "## 8. Watermark & Artifact Shortcut Analysis" in md
        assert "Images with Watermark/Border Indicators:** 1 (Real: 1, Fake: 0)" in md
