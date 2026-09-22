"""Tests for selective Hugging Face dataset acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forensight.data import hf_download


def test_genimage_hub_spec_sd14():
    spec = hf_download.get_genimage_hub_spec(" SD14 ")
    assert spec.remote_prefix == "stable_diffusion_v_1_4"
    assert spec.local_name == "sdv4"


def test_unknown_generator_raises():
    with pytest.raises(ValueError, match="Unknown GenImage generator"):
        hf_download.get_genimage_hub_spec("unknown")


def test_download_pins_revision_and_filters_one_generator(tmp_path: Path, monkeypatch):
    calls: dict[str, object] = {}

    class FakeHfApi:
        def dataset_info(self, *, repo_id: str, revision: str):
            calls["dataset_info"] = {"repo_id": repo_id, "revision": revision}
            return SimpleNamespace(sha="abc123deadbeef")

    def fake_snapshot_download(**kwargs):
        calls["snapshot_download"] = kwargs
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "stable_diffusion_v_1_4").mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(
        hf_download,
        "_load_hf_backend",
        lambda: (FakeHfApi, fake_snapshot_download),
    )

    subset_root, record = hf_download.download_genimage_subset(
        "sd14",
        repo_id="example/GenImage",
        revision="main",
        output_root=tmp_path,
        max_workers=3,
    )

    assert subset_root == tmp_path / "sd14"
    assert record.resolved_revision == "abc123deadbeef"
    assert calls["dataset_info"] == {
        "repo_id": "example/GenImage",
        "revision": "main",
    }

    download_call = calls["snapshot_download"]
    assert isinstance(download_call, dict)
    assert download_call["repo_type"] == "dataset"
    assert download_call["revision"] == "abc123deadbeef"
    assert download_call["allow_patterns"] == ["stable_diffusion_v_1_4/**"]
    assert download_call["max_workers"] == 3

    source = json.loads((subset_root / "source.json").read_text(encoding="utf-8"))
    assert source["repo_id"] == "example/GenImage"
    assert source["requested_revision"] == "main"
    assert source["resolved_revision"] == "abc123deadbeef"
    assert source["generator_id"] == "sd14"


def test_download_rejects_invalid_worker_count(tmp_path: Path):
    with pytest.raises(ValueError, match="max_workers"):
        hf_download.download_genimage_subset("sd14", output_root=tmp_path, max_workers=0)
