"""Tests for dataset inventory schema, validation, serialization, and defaults."""

import json
from pathlib import Path
import pytest

from forensight.data.inventory import (
    DatasetEntry,
    DatasetInventory,
    GeneratorSubset,
    build_default_inventory,
    load_inventory,
    save_inventory,
)


class TestGeneratorSubset:
    """Unit tests for GeneratorSubset dataclass."""

    def test_init_and_to_dict(self):
        subset = GeneratorSubset(
            generator_id="sd14",
            generator_name="Stable Diffusion v1.4",
            role="train_and_in_domain",
            resolution="512x512 for fake; variable for real",
            estimated_count={"train_real": 160000, "train_fake": 160000, "total": 360000},
            local_path="data/raw/genimage/sdv4",
            download_urls=["https://github.com/GenImage-Dataset/GenImage"],
            notes="Primary Stage 1 training subset",
        )
        data = subset.to_dict()
        assert data["generator_id"] == "sd14"
        assert data["generator_name"] == "Stable Diffusion v1.4"
        assert data["role"] == "train_and_in_domain"
        assert data["estimated_count"]["total"] == 360000

        # Roundtrip
        reconstructed = GeneratorSubset.from_dict(data)
        assert reconstructed == subset


class TestDatasetEntry:
    """Unit tests for DatasetEntry dataclass."""

    def test_init_and_roundtrip(self):
        entry = DatasetEntry(
            name="dummy",
            display_name="Dummy Dataset",
            version="1.0",
            source="https://example.com",
            evaluation_role="unit_test",
            real_source="Dummy Photos",
            fake_generators=["Generator A"],
            image_count={"real": 100, "fake": 100, "total": 200},
            resolution_distribution="512x512",
            format_compression="JPEG QF 95",
            label_mapping={"real": 0, "fake": 1},
            license="MIT",
            access_requirements="Public",
            local_path="data/raw/dummy",
            download_instructions=["Download from https://example.com"],
            notes="Test entry",
        )
        data = entry.to_dict()
        assert data["name"] == "dummy"
        reconstructed = DatasetEntry.from_dict(data)
        assert reconstructed == entry


class TestDatasetInventory:
    """Unit tests for DatasetInventory container, queries, and validation."""

    def test_default_inventory_datasets(self):
        inv = build_default_inventory()
        assert len(inv) == 4
        assert "genimage" in inv
        assert "genimage_plus_plus" in inv
        assert "wildrf" in inv
        assert "chameleon" in inv

        names = inv.list_dataset_names()
        assert set(names) == {"genimage", "genimage_plus_plus", "wildrf", "chameleon"}

    def test_genimage_generators(self):
        inv = build_default_inventory()
        genimage = inv["genimage"]
        expected_generators = {
            "sd14",
            "sd15",
            "midjourney",
            "adm",
            "glide",
            "wukong",
            "vqdm",
            "biggan",
        }
        actual_generators = {s.generator_id for s in genimage.generator_subsets}
        assert actual_generators == expected_generators

        # Check SD1.4 role and local path
        sd14 = inv.get_generator_subset("genimage", "sd14")
        assert sd14 is not None
        assert sd14.role == "train_and_in_domain"
        assert sd14.local_path == "data/raw/genimage/sdv4"

        # Check SD1.5 role
        sd15 = inv.get_generator_subset("genimage", "sd15")
        assert sd15 is not None
        assert sd15.role == "near_ood"

        # Check cross-generator OOD subsets
        cross_ood = inv.find_subsets_by_role("cross_generator_ood")
        assert len(cross_ood) == 6
        cross_ids = {s.generator_id for _, s in cross_ood}
        assert cross_ids == {"midjourney", "adm", "glide", "wukong", "vqdm", "biggan"}

    def test_label_mapping_consistency(self):
        inv = build_default_inventory()
        for name, entry in inv.items():
            assert entry.label_mapping == {"real": 0, "fake": 1}, (
                f"Dataset {name} has inconsistent label mapping: {entry.label_mapping}"
            )

    def test_validation_clean_on_default(self):
        inv = build_default_inventory()
        errors = inv.validate()
        assert errors == [], f"Validation failed with errors: {errors}"
        assert inv.is_valid() is True

    def test_validation_catches_invalid_entries(self):
        inv = DatasetInventory()
        invalid_entry = DatasetEntry(
            name="",
            display_name="",
            version="",
            source="",
            evaluation_role="invalid_role",
            real_source="",
            fake_generators=[],
            image_count={},
            resolution_distribution="",
            format_compression="",
            label_mapping={"unknown": 99},
            license="",
            access_requirements="",
        )
        inv.add_entry(invalid_entry)
        errors = inv.validate()
        assert len(errors) > 0

        with pytest.raises(ValueError, match="Inventory validation failed"):
            inv.validate(strict=True)

    def test_validation_evaluation_role_and_image_count(self):
        inv = build_default_inventory()
        inv["genimage"].evaluation_role = ""
        inv["genimage"].image_count = {}
        errors = inv.validate()
        assert any("empty evaluation_role" in err for err in errors)
        assert any("empty image_count" in err for err in errors)

    def test_validation_label_mapping_exact_keys(self):
        inv = build_default_inventory()
        inv["genimage"].label_mapping = {"real": 0, "fake": 1, "extra": 2}
        errors = inv.validate()
        assert any("exactly keys" in err for err in errors)


    def test_json_roundtrip(self, tmp_path):
        inv = build_default_inventory()
        json_path = tmp_path / "test_inventory.json"
        save_inventory(inv, json_path)

        loaded = load_inventory(json_path)
        assert len(loaded) == len(inv)
        assert loaded.list_dataset_names() == inv.list_dataset_names()
        assert loaded.validate() == []
        assert loaded["genimage"].name == inv["genimage"].name

    def test_committed_inventory_json_file(self):
        """Verifies the actual data/dataset_inventory.json file in the repo."""
        repo_root = Path(__file__).resolve().parent.parent
        json_file = repo_root / "data" / "dataset_inventory.json"
        assert json_file.exists(), f"File {json_file} does not exist"

        loaded = load_inventory(json_file)
        assert len(loaded) == 4
        assert loaded.is_valid()
        assert "genimage" in loaded
        assert "genimage_plus_plus" in loaded
        assert "wildrf" in loaded
        assert "chameleon" in loaded
