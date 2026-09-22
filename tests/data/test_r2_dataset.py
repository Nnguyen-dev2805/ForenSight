from pathlib import Path
import pytest
from PIL import Image

from forensight.data.split import Manifest, ManifestRecord


def test_load_manifest_file_jsonl(tmp_path: Path):
    from forensight.data.r2_dataset import load_manifest_file

    manifest_path = tmp_path / "manifest.jsonl"
    Manifest([
        ManifestRecord(
            sample_id="s1",
            image_path="image.png",
            label=1,
            dataset="genimage",
            generator="sd14",
            split="train",
        )
    ]).to_jsonl(manifest_path)

    loaded = load_manifest_file(manifest_path)
    assert len(loaded) == 1
    assert loaded[0].sample_id == "s1"


def test_load_manifest_file_csv(tmp_path: Path):
    from forensight.data.r2_dataset import load_manifest_file

    manifest_path = tmp_path / "manifest.csv"
    Manifest([
        ManifestRecord(
            sample_id="s2",
            image_path="image2.png",
            label=0,
            dataset="genimage",
            generator="nature",
            split="val",
        )
    ]).to_csv(manifest_path)

    loaded = load_manifest_file(manifest_path)
    assert len(loaded) == 1
    assert loaded[0].sample_id == "s2"


def test_load_manifest_file_unsupported(tmp_path: Path):
    from forensight.data.r2_dataset import load_manifest_file

    bad_path = tmp_path / "manifest.txt"
    bad_path.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported manifest format"):
        load_manifest_file(bad_path)


def test_dataset_loads_image_once_and_preserves_metadata(tmp_path: Path):
    import torch
    from forensight.data.r2_dataset import R2ImageDataset

    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 12), color=(100, 110, 120)).save(image_path)
    manifest = Manifest([
        ManifestRecord(
            sample_id="s1",
            image_path=str(image_path),
            label=1,
            dataset="genimage",
            generator="sd14",
            split="train",
        )
    ])

    def clip_transform(image):
        return torch.ones(3, 8, 8)

    def forensic_transform(image):
        return torch.zeros(3, 8, 8)

    dataset = R2ImageDataset(
        manifest,
        clip_transform=clip_transform,
        forensic_transform=forensic_transform,
    )
    assert len(dataset) == 1
    sample = dataset[0]

    assert sample["clip_image"].shape == (3, 8, 8)
    assert sample["forensic_image"].shape == (3, 8, 8)
    assert sample["label"].dtype == torch.float32
    assert sample["label"].item() == 1.0
    assert sample["sample_id"] == "s1"
    assert sample["split"] == "train"
    assert sample["generator"] == "sd14"
    assert sample["dataset"] == "genimage"


def test_dataset_requires_at_least_one_transform():
    from forensight.data.r2_dataset import R2ImageDataset

    manifest = Manifest([])
    with pytest.raises(ValueError, match="At least one branch transform"):
        R2ImageDataset(manifest, clip_transform=None, forensic_transform=None)


def test_dataset_resolves_relative_path_with_base_dir(tmp_path: Path):
    import torch
    from forensight.data.r2_dataset import R2ImageDataset

    base_dir = tmp_path / "data_root"
    base_dir.mkdir()
    rel_path = "subdir/img.png"
    (base_dir / "subdir").mkdir()
    Image.new("RGB", (10, 10), color=(1, 2, 3)).save(base_dir / rel_path)

    manifest = Manifest([
        ManifestRecord(
            sample_id="s_rel",
            image_path=rel_path,
            label=0,
            dataset="genimage",
            generator="nature",
            split="in_domain_test",
        )
    ])

    dataset = R2ImageDataset(
        manifest,
        base_dir=base_dir,
        clip_transform=lambda img: torch.zeros(3, 4, 4),
    )
    sample = dataset[0]
    assert sample["sample_id"] == "s_rel"
    assert "clip_image" in sample
    assert "forensic_image" not in sample
    assert sample["label"].item() == 0.0


def test_build_forensic_input_transform():
    import torch
    from forensight.data.r2_dataset import build_forensic_input_transform

    transform = build_forensic_input_transform(image_size=32)
    img = Image.new("RGB", (64, 48), color=(50, 60, 70))
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
