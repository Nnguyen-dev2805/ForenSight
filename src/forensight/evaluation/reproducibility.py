"""Experiment reproducibility recording, environment capture, and provenance tracking.

This module implements Task 0.8 of ForenSight Milestone R0:
- ReproducibilityRecord: Standardized dataclass enforcing docs/evaluation.md:
  - run_id: unique run identifier
  - experiment_name: experiment or baseline family
  - timestamp: UTC ISO 8601 string
  - git_commit: git commit SHA (HEAD) if version-controlled
  - seed: random seed
  - split_version: dataset split version/tag
  - config: hyperparameter/configuration dictionary
  - threshold_source: provenance of decision threshold
  - threshold_value: numerical decision threshold
  - metrics: evaluation metrics dictionary
  - environment: runtime platform, python version, dependencies, hardware
  - notes: free-text experiment annotations
  - validate() -> list[str]: validation against schema invariants
  - serialization: to_dict(), to_json(), save_json(), from_dict(), from_json(), load_json()
- create_reproducibility_record(): helper capturing runtime environment, git commit, and metrics.
- get_git_commit(), get_environment_info(): introspection utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence
import numpy as np


_SENTINEL: Any = object()

DEFAULT_KEY_PACKAGES: list[str] = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pillow",
    "pytest",
]


def get_git_commit(cwd: str | Path | None = None) -> str | None:
    """Retrieve the current Git commit SHA (HEAD), or None if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            return commit if commit else None
    except Exception:
        pass
    return None


def get_environment_info(packages: Sequence[str] | None = None) -> dict[str, Any]:
    """Capture runtime environment metadata including Python, OS platform, and package versions."""
    target_pkgs = list(packages) if packages is not None else DEFAULT_KEY_PACKAGES

    pkg_versions: dict[str, str | None] = {}
    for pkg in target_pkgs:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = None

    # Detect hardware/acceleration capabilities
    hardware: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
    }

    # Check PyTorch device availability if torch is loaded
    if "torch" in sys.modules:
        try:
            import torch  # type: ignore

            hardware["cuda_available"] = bool(torch.cuda.is_available())
            if hardware["cuda_available"]:
                hardware["cuda_device_count"] = torch.cuda.device_count()
                hardware["cuda_device_name"] = torch.cuda.get_device_name(0)
            hardware["mps_available"] = bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
        except Exception:
            pass

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": pkg_versions,
        "hardware": hardware,
    }


@dataclass
class ReproducibilityRecord:
    """Standardized record of experiment provenance, environment, and evaluation results.

    Enforces docs/evaluation.md.
    Every run must preserve:
    - config
    - seed
    - split_version
    - threshold_source and threshold_value
    - metrics
    - timestamp (ISO 8601 UTC)
    - git_commit (if available)
    - environment (python version, platform, dependencies, hardware)

    Attributes:
        run_id: Unique identifier for the individual run.
        experiment_name: Identifier for the experiment or baseline group.
        timestamp: ISO 8601 UTC timestamp of execution.
        git_commit: Git commit SHA (HEAD) if running in a version-controlled repo.
        seed: Random seed used for model initialization, sampling, or evaluation.
        split_version: Identifier/version of the dataset split or manifest used.
        config: Full configuration dictionary capturing all relevant hyperparameters.
        threshold_source: Description or strategy of how the decision threshold was chosen.
        threshold_value: Numerical decision threshold applied to evaluation scores.
        metrics: Complete metric dictionary (e.g. from EvaluationReport.to_dict()).
        environment: Runtime environment metadata (Python version, OS, packages, hardware).
        notes: Free-text annotations, rationale, or operational notes.
    """

    run_id: str
    experiment_name: str
    timestamp: str
    git_commit: str | None
    seed: int | str | None
    split_version: str
    config: dict[str, Any]
    threshold_source: str
    threshold_value: float
    metrics: dict[str, Any]
    environment: dict[str, Any]
    notes: str = ""
    dataset: str = "Tiny-GenImage"
    dataset_revision: str | None = None

    def validate(self, require_provenance: bool = False) -> list[str]:
        """Validate presence, types, and invariants of all required fields.

        Args:
            require_provenance: If True, enforce that git_commit and dataset_revision
                are non-empty strings for strict reproducibility tracking.

        Returns:
            List of validation error strings. Returns an empty list if all invariants pass.
        """
        errors: list[str] = []

        # 1. run_id: non-empty string
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            errors.append("run_id must be a non-empty string.")

        # 2. experiment_name: non-empty string
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            errors.append("experiment_name must be a non-empty string.")

        # 3. timestamp: non-empty valid ISO 8601 string
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            errors.append("timestamp must be a valid ISO 8601 formatted string.")
        else:
            try:
                datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                errors.append(
                    f"timestamp is not a valid ISO 8601 formatted string: {self.timestamp}"
                )

        # 4. split_version: non-empty string
        if not isinstance(self.split_version, str) or not self.split_version.strip():
            errors.append("split_version must be a non-empty string.")

        # 5. threshold_source: non-empty string
        if not isinstance(self.threshold_source, str) or not self.threshold_source.strip():
            errors.append("threshold_source must be a non-empty string.")

        # 6. threshold_value: finite numerical float
        if (
            not (
                isinstance(self.threshold_value, (int, float))
                and not isinstance(self.threshold_value, bool)
            )
            or np.isnan(self.threshold_value)
            or np.isinf(self.threshold_value)
        ):
            errors.append(
                f"threshold_value must be a finite numerical float, got: {self.threshold_value}"
            )

        # 7. config: dict
        if not isinstance(self.config, dict):
            errors.append(f"config must be a dictionary, got: {type(self.config).__name__}")

        # 8. metrics: dict
        if not isinstance(self.metrics, dict):
            errors.append(f"metrics must be a dictionary, got: {type(self.metrics).__name__}")

        # 9. environment: dict
        if not isinstance(self.environment, dict):
            errors.append(
                f"environment must be a dictionary, got: {type(self.environment).__name__}"
            )

        # 10. seed: int, str, or None
        if self.seed is not None:
            if not (
                isinstance(self.seed, (int, str))
                and not isinstance(self.seed, bool)
            ):
                errors.append(
                    f"seed must be an int, str, or None, got: {type(self.seed).__name__}"
                )
            elif isinstance(self.seed, str) and not self.seed.strip():
                errors.append("seed cannot be an empty string if specified.")

        # 11. git_commit: str or None (mandatory if require_provenance=True)
        if require_provenance and (self.git_commit is None or not str(self.git_commit).strip()):
            errors.append("git_commit must be a non-empty string for strict reproducibility provenance.")
        elif self.git_commit is not None:
            if not isinstance(self.git_commit, str):
                errors.append(
                    f"git_commit must be a string or None, got: {type(self.git_commit).__name__}"
                )
            elif not self.git_commit.strip():
                errors.append("git_commit cannot be an empty string if specified.")

        # 12. dataset: non-empty string
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            errors.append("dataset must be a non-empty string.")

        # 13. dataset_revision: str or None (mandatory if require_provenance=True)
        if require_provenance and (self.dataset_revision is None or not str(self.dataset_revision).strip()):
            errors.append("dataset_revision must be a non-empty string for strict reproducibility provenance.")
        elif self.dataset_revision is not None:
            if not isinstance(self.dataset_revision, str):
                errors.append(
                    f"dataset_revision must be a string or None, got: {type(self.dataset_revision).__name__}"
                )
            elif not self.dataset_revision.strip():
                errors.append("dataset_revision cannot be an empty string if specified.")

        # 14. notes: str
        if not isinstance(self.notes, str):
            errors.append(f"notes must be a string, got: {type(self.notes).__name__}")

        return errors

    @property
    def is_valid(self) -> bool:
        """Return True if record satisfies all validation rules."""
        return len(self.validate()) == 0

    def assert_valid(self, require_provenance: bool = False) -> None:
        """Raise ValueError if the record violates any validation rules."""
        errs = self.validate(require_provenance=require_provenance)
        if errs:
            raise ValueError(
                f"ReproducibilityRecord validation failed with {len(errs)} error(s):\n"
                + "\n".join(f"- {e}" for e in errs)
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "seed": self.seed,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "split_version": self.split_version,
            "config": dict(self.config),
            "threshold_source": self.threshold_source,
            "threshold_value": float(self.threshold_value),
            "metrics": dict(self.metrics),
            "environment": dict(self.environment),
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize record to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        """Save record to JSON file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproducibilityRecord:
        """Construct ReproducibilityRecord from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        raw_thresh = data.get("threshold_value", 0.0)
        try:
            thresh_val = float(raw_thresh)
        except (ValueError, TypeError):
            thresh_val = float("nan")

        return cls(
            run_id=str(data.get("run_id") or ""),
            experiment_name=str(data.get("experiment_name") or ""),
            timestamp=str(data.get("timestamp") or ""),
            git_commit=data.get("git_commit"),
            seed=data.get("seed"),
            dataset=str(data.get("dataset") or "Tiny-GenImage"),
            dataset_revision=data.get("dataset_revision"),
            split_version=str(data.get("split_version") or ""),
            config=dict(data.get("config") or {}),
            threshold_source=str(data.get("threshold_source") or ""),
            threshold_value=thresh_val,
            metrics=dict(data.get("metrics") or {}),
            environment=dict(data.get("environment") or {}),
            notes=str(data.get("notes") or ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ReproducibilityRecord:
        """Construct ReproducibilityRecord from JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON string: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load_json(cls, path: str | Path) -> ReproducibilityRecord:
        """Load ReproducibilityRecord from JSON file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        with open(target, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def generate_markdown(self) -> str:
        """Generate human-readable Markdown summary of reproducibility record."""
        commit_str = (
            f"`{self.git_commit[:10]}`" if self.git_commit else "N/A (unversioned)"
        )
        seed_str = str(self.seed) if self.seed is not None else "None"
        py_ver = self.environment.get("python_version", "unknown")
        plat = self.environment.get("platform", "unknown")

        overall = (
            self.metrics.get("overall", {}) if isinstance(self.metrics, dict) else {}
        )
        auroc_val = overall.get("auroc")
        acc_val = overall.get("accuracy")
        f1_val = overall.get("f1")

        auroc_str = f"{auroc_val:.4f}" if isinstance(auroc_val, (int, float)) else "N/A"
        acc_str = f"{acc_val:.4f}" if isinstance(acc_val, (int, float)) else "N/A"
        f1_str = f"{f1_val:.4f}" if isinstance(f1_val, (int, float)) else "N/A"

        lines = [
            f"# ForenSight Reproducibility Record: `{self.run_id}`",
            "",
            "## 1. Experiment & Provenance Metadata",
            f"- **Experiment Name:** `{self.experiment_name}`",
            f"- **Timestamp (UTC):** `{self.timestamp}`",
            f"- **Git Commit:** {commit_str}",
            f"- **Random Seed:** `{seed_str}`",
            f"- **Split Version:** `{self.split_version}`",
            f"- **Threshold Source:** `{self.threshold_source}`",
            f"- **Decision Threshold:** `{self.threshold_value:.4f}`",
            "",
            "## 2. Environment",
            f"- **Python Version:** `{py_ver}`",
            f"- **Platform:** `{plat}`",
        ]

        pkgs = self.environment.get("packages", {})
        if pkgs and isinstance(pkgs, dict):
            lines.append("- **Key Dependencies:**")
            for pkg, ver in sorted(pkgs.items()):
                if ver is not None:
                    lines.append(f"  - `{pkg}`: `{ver}`")

        if self.notes:
            lines.extend([
                "",
                "## 3. Notes",
                self.notes,
            ])

        lines.extend([
            "",
            "## 4. Key Metrics Summary",
            f"- **Overall AUROC:** {auroc_str}",
            f"- **Overall Accuracy:** {acc_str}",
            f"- **Overall F1 Score:** {f1_str}",
        ])
        return "\n".join(lines)


def create_reproducibility_record(
    *,
    run_id: str | None = None,
    experiment_name: str = "evaluation",
    split_version: str = "r0-default",
    config: dict[str, Any] | None = None,
    threshold_source: str | None = None,
    threshold_value: float | None = None,
    metrics: dict[str, Any] | Any = None,
    seed: int | str | None = None,
    dataset: str = "Tiny-GenImage",
    dataset_revision: str | None = None,
    git_commit: Any = _SENTINEL,
    environment: dict[str, Any] | None = None,
    notes: str = "",
    report: Any = None,
    cwd: str | Path | None = None,
) -> ReproducibilityRecord:
    """Helper capturing current git commit, environment info, timestamp, threshold, and metrics.

    Args:
        run_id: Unique run ID (auto-generated if None).
        experiment_name: Name of experiment or baseline family.
        split_version: Dataset split/manifest version tag.
        config: Run configuration dictionary.
        threshold_source: Provenance of decision threshold.
        threshold_value: Numerical decision threshold.
        metrics: Metric dictionary or EvaluationReport instance.
        seed: Random seed.
        git_commit: Git commit SHA (auto-detected if sentinel).
        environment: Environment dict (auto-detected if None).
        notes: Experiment notes.
        report: Optional EvaluationReport instance to extract metrics, threshold, seed, and run name.
        cwd: Optional directory for git detection.

    Returns:
        Instantiated and validated ReproducibilityRecord.
    """
    # If metrics is an EvaluationReport, treat it as report
    from forensight.evaluation.runner import EvaluationReport

    if isinstance(metrics, EvaluationReport):
        report = metrics
        metrics = None

    if report is not None:
        if metrics is None:
            metrics = report.to_dict()
        if threshold_source is None:
            threshold_source = report.threshold_metadata.get(
                "threshold_source", report.overall.threshold_source
            )
        if threshold_value is None:
            threshold_value = float(
                report.threshold_metadata.get("threshold", report.overall.threshold)
            )
        if seed is None:
            seed = report.seed
        if run_id is None and report.run_metadata.get("run_name"):
            run_id = str(report.run_metadata["run_name"])

    # Normalization & defaults
    if metrics is None:
        metrics = {}
    elif hasattr(metrics, "to_dict"):
        metrics = metrics.to_dict()
    elif not isinstance(metrics, dict):
        raise TypeError(
            f"metrics must be a dict or EvaluationReport, got: {type(metrics).__name__}"
        )

    if threshold_source is None:
        threshold_source = "default"
    if threshold_value is None:
        threshold_value = 0.5
    if config is None:
        config = {}

    timestamp = datetime.now(timezone.utc).isoformat()
    if run_id is None:
        safe_time = timestamp.replace(":", "").replace("-", "").replace(".", "_")
        run_id = f"{experiment_name}_{safe_time}"

    if git_commit is _SENTINEL:
        git_commit = get_git_commit(cwd=cwd)

    if environment is None:
        environment = get_environment_info()

    return ReproducibilityRecord(
        run_id=str(run_id),
        experiment_name=str(experiment_name),
        timestamp=timestamp,
        git_commit=git_commit,
        seed=seed,
        split_version=str(split_version),
        config=dict(config),
        threshold_source=str(threshold_source),
        threshold_value=float(threshold_value),
        metrics=dict(metrics),
        environment=dict(environment),
        notes=str(notes),
        dataset=str(dataset),
        dataset_revision=dataset_revision,
    )
