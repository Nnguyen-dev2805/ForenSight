"""Selective Hugging Face downloads for ForenSight datasets.

The research pipeline consumes immutable local files and versioned manifests. This
module therefore uses Hugging Face only as the transport layer: it downloads one
GenImage generator archive at a time, pins the requested Hub revision to a commit
SHA, and writes source metadata next to the downloaded archive files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable


DEFAULT_GENIMAGE_REPO_ID = "ENSTA-U2IS/GenImage"


@dataclass(frozen=True)
class GenImageHubSpec:
    """Mapping between ForenSight generator IDs and the Hugging Face mirror."""

    generator_id: str
    remote_prefix: str
    local_name: str


GENIMAGE_HUB_SPECS: dict[str, GenImageHubSpec] = {
    "sd14": GenImageHubSpec("sd14", "stable_diffusion_v_1_4", "sdv4"),
    "sd15": GenImageHubSpec("sd15", "stable_diffusion_v_1_5", "sdv5"),
    "midjourney": GenImageHubSpec("midjourney", "Midjourney", "midjourney"),
    "adm": GenImageHubSpec("adm", "ADM", "adm"),
    "glide": GenImageHubSpec("glide", "glide", "glide"),
    "wukong": GenImageHubSpec("wukong", "wukong", "wukong"),
    "vqdm": GenImageHubSpec("vqdm", "VQDM", "vqdm"),
    "biggan": GenImageHubSpec("biggan", "BigGAN", "biggan"),
}


@dataclass(frozen=True)
class HubDownloadRecord:
    """Provenance written after a successful selective Hub download."""

    repo_id: str
    requested_revision: str
    resolved_revision: str
    generator_id: str
    remote_prefix: str
    local_name: str
    allow_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_hf_backend() -> tuple[type[Any], Callable[..., Any]]:
    """Import huggingface_hub lazily so the rest of ForenSight stays lightweight."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in CLI environments
        raise RuntimeError(
            "huggingface-hub is required for dataset downloads. "
            "Install project dependencies first (for example: pip install -e .)."
        ) from exc
    return HfApi, snapshot_download


def get_genimage_hub_spec(generator_id: str) -> GenImageHubSpec:
    """Return the Hub mapping for a canonical ForenSight generator ID."""
    key = generator_id.lower().strip()
    try:
        return GENIMAGE_HUB_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GenImage generator '{generator_id}'. "
            f"Expected one of: {sorted(GENIMAGE_HUB_SPECS)}"
        ) from exc


def download_genimage_subset(
    generator_id: str = "sd14",
    *,
    repo_id: str = DEFAULT_GENIMAGE_REPO_ID,
    revision: str = "main",
    output_root: str | Path = "data/raw/genimage/_downloads",
    max_workers: int = 8,
    force_download: bool = False,
) -> tuple[Path, HubDownloadRecord]:
    """Download exactly one GenImage generator archive from Hugging Face.

    The mirror stores each generator as split ZIP archive parts. Files are kept in
    ``data/raw/genimage/_downloads``; extraction into the canonical
    ``data/raw/genimage/<generator>`` directory is a separate step so the immutable
    raw archive and its provenance remain intact.

    The requested branch/tag is first resolved to a commit SHA. The actual download
    is then pinned to that SHA so a later Hub update cannot silently change an
    experiment's input bytes.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    spec = get_genimage_hub_spec(generator_id)
    HfApi, snapshot_download = _load_hf_backend()

    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, revision=revision)
    resolved_revision = str(getattr(info, "sha", "") or "").strip()
    if not resolved_revision:
        raise RuntimeError(
            f"Hugging Face did not return a commit SHA for {repo_id}@{revision}."
        )

    subset_root = Path(output_root) / spec.generator_id
    subset_root.mkdir(parents=True, exist_ok=True)
    allow_patterns = [f"{spec.remote_prefix}/**"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        allow_patterns=allow_patterns,
        local_dir=str(subset_root),
        max_workers=max_workers,
        force_download=force_download,
    )

    record = HubDownloadRecord(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        generator_id=spec.generator_id,
        remote_prefix=spec.remote_prefix,
        local_name=spec.local_name,
        allow_patterns=allow_patterns,
    )
    (subset_root / "source.json").write_text(
        json.dumps(record.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return subset_root, record


__all__ = [
    "DEFAULT_GENIMAGE_REPO_ID",
    "GENIMAGE_HUB_SPECS",
    "GenImageHubSpec",
    "HubDownloadRecord",
    "download_genimage_subset",
    "get_genimage_hub_spec",
]
