"""Shared training utilities for ForenSight R2 experiments.

Provides unified execution across Semantic-only, Forensic-only, and Fusion
detector variants: seed initialization, config loading, forward dispatch,
epoch training/validation, atomic checkpointing, and R0-compatible prediction export.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from forensight.data.r2_dataset import build_forensic_input_transform
from forensight.evaluation.runner import PredictionRecord, PredictionSet
from forensight.models.forensic import ForensicOnlyDetector, build_resnet18_forensic
from forensight.models.fusion import FusionDetector
from forensight.models.semantic import SemanticOnlyDetector, load_open_clip_semantic


VALID_VARIANTS = {"semantic", "forensic", "fusion", "semantic_only", "forensic_only", "concat_fusion"}
REQUIRED_CONFIG_SECTIONS = {"model", "training", "evaluation"}


def select_device() -> torch.device:
    """Select MPS on Apple Silicon, CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_r2_model(
    config: dict[str, Any],
) -> tuple[nn.Module, Callable[[Any], torch.Tensor] | None, Callable[[Any], torch.Tensor] | None]:
    """Assemble an R2 model and branch transforms according to experiment config."""
    variant = config["variant"]
    model_cfg = config["model"]
    clip_transform = None
    forensic_transform = None

    if variant in {"semantic", "fusion"}:
        semantic_encoder, clip_transform = load_open_clip_semantic(
            model_name=model_cfg["clip_model"],
            pretrained=model_cfg["clip_pretrained"],
            projection_dim=model_cfg["projection_dim"],
        )

    if variant in {"forensic", "fusion"}:
        forensic_encoder = build_resnet18_forensic(
            projection_dim=model_cfg["projection_dim"],
            npr_scale_factor=model_cfg["npr_scale_factor"],
            npr_mode=model_cfg["npr_mode"],
            pretrained=model_cfg.get("forensic_pretrained", False),
        )
        forensic_transform = build_forensic_input_transform(model_cfg.get("image_size", 224))

    if variant == "semantic":
        model = SemanticOnlyDetector(
            semantic_encoder,
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    elif variant == "forensic":
        model = ForensicOnlyDetector(
            forensic_encoder,
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    elif variant == "fusion":
        model = FusionDetector(
            semantic_encoder,
            forensic_encoder,
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unsupported R2 variant: {variant}")

    return model, clip_transform, forensic_transform


def set_seed(seed: int) -> None:
    """Set random seed across python stdlib, numpy, and pytorch, and pin cuDNN.

    The cuDNN flags matter as much as the RNG seeds: with `benchmark=True` cuDNN
    picks convolution algorithms by timing heuristics, so two runs of the same
    seed can still diverge. `torch.use_deterministic_algorithms(True)` is
    deliberately NOT enabled -- it rejects some CUDA ops this model needs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_r2_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an R2 JSON experiment configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    variant = config.get("variant")
    if variant in {"semantic", "semantic_only"}:
        config["variant"] = "semantic"
    elif variant in {"forensic", "forensic_only"}:
        config["variant"] = "forensic"
    elif variant in {"fusion", "concat_fusion"}:
        config["variant"] = "fusion"
    elif variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unsupported R2 variant: {variant!r}. Must be one of {sorted(VALID_VARIANTS)}"
        )

    for section in REQUIRED_CONFIG_SECTIONS:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing required configuration section: '{section}'")

    if config["variant"] in {"forensic", "fusion"}:
        normalization = config["model"].setdefault("npr_normalization", "spatial_std")
        if normalization != "spatial_std":
            raise ValueError("npr_normalization must be 'spatial_std' for the R2 baseline")

    return config


def forward_variant(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    variant: str,
    device: torch.device,
) -> torch.Tensor:
    """Dispatch batch inputs to model based on the active variant."""
    if variant == "semantic":
        return model(batch["clip_image"].to(device))
    if variant == "forensic":
        return model(batch["forensic_image"].to(device))
    if variant == "fusion":
        return model(
            batch["clip_image"].to(device),
            batch["forensic_image"].to(device),
        )
    raise ValueError(f"Unsupported R2 variant: {variant}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    variant: str,
) -> float:
    """Run one epoch of training and return average BCEWithLogitsLoss."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        labels = batch["label"].to(device).float().view(-1, 1)
        logits = forward_variant(model, batch, variant=variant, device=device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: str,
    return_metrics: bool = False,
) -> float | tuple[float, float | None]:
    """Run validation pass and return average BCEWithLogitsLoss (and optionally val_auroc)."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_batches = 0
    all_labels: list[int] = []
    all_scores: list[float] = []

    for batch in loader:
        labels = batch["label"].to(device).float().view(-1, 1)
        logits = forward_variant(model, batch, variant=variant, device=device)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        num_batches += 1

        if return_metrics:
            scores = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            lbls = labels.squeeze(1).cpu().int().tolist()
            if isinstance(lbls, int):
                lbls = [lbls]
            all_scores.extend(scores)
            all_labels.extend(lbls)

    avg_loss = total_loss / max(1, num_batches)
    if not return_metrics:
        return avg_loss

    from forensight.evaluation.metrics import calculate_auroc
    val_auroc: float | None = None
    if len(all_labels) > 0 and len(set(all_labels)) >= 2:
        val_auroc = float(calculate_auroc(all_labels, all_scores))

    return avg_loss, val_auroc


STATE_DICT_SCOPE_TRAINABLE = "trainable_subtrees"


def _frozen_subtree_names(model: nn.Module) -> list[str]:
    """Names of submodules whose whole subtree is frozen (holds no trainable parameter)."""
    frozen: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        parameters = list(module.parameters(recurse=True))
        if parameters and not any(p.requires_grad for p in parameters):
            frozen.append(name)
    return frozen


def trainable_state_dict(model: nn.Module) -> dict[str, Any]:
    """Return the model state restricted to trainable subtrees.

    Frozen backbones (e.g. the CLIP vision tower) are fully determined by the
    experiment config, so serializing them is redundant. Everything that actually
    changes during training is kept, including buffers such as BatchNorm running
    statistics inside the trainable forensic backbone.
    """
    frozen = _frozen_subtree_names(model)
    return {
        key: value
        for key, value in model.state_dict().items()
        if not any(key == prefix or key.startswith(prefix + ".") for prefix in frozen)
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    config: dict[str, Any],
) -> None:
    """Save trainable model state and optimizer state alongside epoch and config.

    Only trainable subtrees are written; frozen backbones are rebuilt from `config`
    at load time, which keeps run artifacts small.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": trainable_state_dict(model),
        "state_dict_scope": STATE_DICT_SCOPE_TRAINABLE,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model and optional optimizer state from a checkpoint file.

    Checkpoints written by `save_checkpoint` contain only trainable subtrees and are
    verified against the trainable keys of the freshly built `model`; the frozen parts
    come from that model. Legacy full state dictionaries are still accepted.
    """
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint["model_state_dict"]

    if checkpoint.get("state_dict_scope") == STATE_DICT_SCOPE_TRAINABLE:
        expected = set(trainable_state_dict(model))
        if set(state) != expected:
            missing = sorted(expected - set(state))[:5]
            unexpected = sorted(set(state) - expected)[:5]
            raise ValueError(
                "Checkpoint does not match the trainable parameters of the model built "
                f"from the supplied config (missing={missing}, unexpected={unexpected}). "
                "Check that the checkpoint variant matches the config variant."
            )
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


@torch.no_grad()
def predict_to_prediction_set(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: str,
) -> PredictionSet:
    """Run inference over a loader and return a ForenSight PredictionSet."""
    model.eval()
    records: list[PredictionRecord] = []

    for batch in loader:
        logits = forward_variant(model, batch, variant=variant, device=device)
        scores = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        batch_size = len(scores)

        for index in range(batch_size):
            img_path = None
            if "path" in batch:
                img_path = str(batch["path"][index])
            records.append(
                PredictionRecord(
                    sample_id=str(batch["sample_id"][index]),
                    label=int(batch["label"][index].item()),
                    score=float(scores[index]),
                    split=str(batch["split"][index]),
                    generator=str(batch["generator"][index]),
                    dataset=str(batch["dataset"][index]),
                    path=img_path,
                    metadata={
                        # Absent for hand-built batches that bypass R2ImageDataset.
                        "evaluation_generator": str(
                            batch["evaluation_generator"][index]
                        ) if "evaluation_generator" in batch else "",
                    },
                )
            )

    return PredictionSet(records)
