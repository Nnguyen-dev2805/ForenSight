#!/usr/bin/env python3
"""Local orchestrator CLI to manage ForenSight Kaggle GPU kernels.

Usage:
    python scripts/run_on_kaggle.py --push
    python scripts/run_on_kaggle.py --status
    python scripts/run_on_kaggle.py --logs
    python scripts/run_on_kaggle.py --download
    python scripts/run_on_kaggle.py --watch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_on_kaggle")

DEFAULT_KERNEL_DIR = "deploy/kaggle"
DEFAULT_KERNEL_SLUG = "truongnhatnguyen2805/forensight-r2-fusion-training"
DEFAULT_OUTPUT_DIR = "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage ForenSight Kaggle GPU execution.")
    parser.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR, help="Path to kernel directory containing main.py and metadata.")
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG, help="Kaggle kernel identifier (username/slug).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save downloaded outputs.")
    parser.add_argument("--push", action="store_true", help="Push kernel and trigger execution on Kaggle GPU.")
    parser.add_argument("--status", action="store_true", help="Check current execution status.")
    parser.add_argument("--logs", action="store_true", help="Print recent execution logs.")
    parser.add_argument("--download", action="store_true", help="Download output files (checkpoint, report).")
    parser.add_argument("--watch", action="store_true", help="Watch kernel status until complete and auto-download results.")
    parser.add_argument("--experiment", choices=["canonical", "single", "logo", "all_in_one", "all7", "all"], default=None, help="Experiment paradigm to execute")
    parser.add_argument("--variant", choices=["fusion", "semantic_only", "forensic_only", "all"], default=None, help="Model variant to run")
    parser.add_argument("--leave-out", default=None, help="Held-out generator for LOGO")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of random seeds (e.g. 42 1337 2024)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override")
    parser.add_argument("--dataset-ref", default=None, help="Pinned Kaggle dataset handle, e.g. yangsangtai/tiny-genimage/versions/1")
    parser.add_argument(
        "--prepare-manifests",
        action="store_true",
        help="Build sealed manifests inside the kernel from the pinned dataset instead of shipping local manifests.",
    )
    return parser.parse_args()


def get_kaggle_cmd() -> list[str]:
    # Look for virtualenv kaggle first
    venv_kaggle = Path(".venv/bin/kaggle")
    if venv_kaggle.exists():
        return [str(venv_kaggle)]
    return ["kaggle"]


# Canonical code shipped to every variant kernel. Kept explicit so the code
# digest cannot silently start covering a run-specific file.
CODE_PAYLOAD_FILES = (
    "src",
    "configs",
    "scripts/check_dataset.py",
    "scripts/train_r2.py",
    "scripts/evaluate_r2.py",
    "scripts/aggregate_runs.py",
)


def _code_payload_files(bundle_dir: Path) -> list[Path]:
    """Files under bundle_dir that make up the canonical code payload."""
    files: list[Path] = []
    for entry in CODE_PAYLOAD_FILES:
        target = bundle_dir / entry
        if target.is_dir():
            files.extend(p for p in target.rglob("*") if p.is_file())
        elif target.is_file():
            files.append(target)
    return files


def prepare_official_bundle(
    repo_root: Path, kernel_dir: Path, bundle_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Stage canonical code and sealed manifests without changing the checkout."""
    dataset_ref = str(config.get("dataset_ref") or "")
    if "/versions/" not in dataset_ref or not dataset_ref.rsplit("/versions/", 1)[-1].isdigit():
        raise ValueError("Official Kaggle runs require a version-pinned --dataset-ref")

    experiment = config.get("experiment")
    prepare_manifests = bool(config.get("prepare_manifests"))
    manifest_root = repo_root / "data" / "manifests"
    if experiment == "single":
        required = [manifest_root / "single" / f"{name}.jsonl" for name in
                    ("train", "val", "in_domain_test", "cross_generator_ood")]
    elif experiment == "logo":
        leave_out = config.get("leave_out")
        if not leave_out:
            raise ValueError("--leave-out is required for LOGO")
        fold = manifest_root / "logo" / f"leave_{leave_out}"
        required = [fold / f"{name}.jsonl" for name in
                    ("train", "val", "test_in_domain_seen", f"test_{leave_out}")]
    elif experiment == "all7":
        required = [manifest_root / "all7" / f"{name}.jsonl" for name in
                    ("train", "val", "test_all_combined")]
    else:
        raise ValueError("Official Kaggle experiment must be single, logo, or all7")
    if prepare_manifests:
        # The kernel rebuilds the splits from the pinned dataset; there is nothing to ship.
        required = []
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"Sealed manifest missing: {path}. Generate it with "
                "scripts/download_tiny_genimage.py, or pass --prepare-manifests to build it "
                "inside the kernel from the pinned dataset."
            )
    if not prepare_manifests:
        provenance_path = manifest_root / "manifest_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Sealed manifest provenance missing: {provenance_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("dataset_ref") != dataset_ref or provenance.get("skip_download"):
            raise ValueError("Staged manifest dataset_ref is unverified or differs from --dataset-ref")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(kernel_dir / "kernel-metadata.json", bundle_dir / "kernel-metadata.json")
    official_src = kernel_dir / "official.py"
    if not official_src.exists():
        official_src = repo_root / "deploy" / "kaggle" / "official.py"
    shutil.copy2(official_src, bundle_dir / "main.py")
    folders = ["src", "scripts", "configs"]
    if not prepare_manifests:
        folders.append("data/manifests")
    for folder in folders:
        shutil.copytree(
            repo_root / folder, bundle_dir / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    # Full payload digest: everything staged, including kernel metadata and the
    # sealed manifests. Identifies one exact run bundle.
    digest = hashlib.sha256()
    for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        digest.update(path.relative_to(bundle_dir).as_posix().encode())
        digest.update(path.read_bytes())

    # Code-only digest: canonical source shared by every variant kernel, so
    # results from different variants are comparable only when this matches.
    # Excludes kernel metadata (differs per variant kernel by design) and the
    # sealed manifests (run inputs, not code).
    code_digest = hashlib.sha256()
    for path in sorted(p for p in _code_payload_files(bundle_dir)):
        code_digest.update(path.relative_to(bundle_dir).as_posix().encode())
        code_digest.update(path.read_bytes())

    prepared = dict(config)
    prepared["prepare_manifests"] = prepare_manifests
    prepared["source_payload_sha256"] = digest.hexdigest()
    prepared["code_payload_sha256"] = code_digest.hexdigest()
    (bundle_dir / "run_config.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    return prepared


def push_kernel(kernel_dir: str | Path, config: dict | None = None) -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    kernel_path = (repo_root / kernel_dir).resolve()
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True)
    dirty = git_status.returncode != 0 or bool(git_status.stdout.strip())
    run_config = dict(config or {})
    run_config.update({
        "git_commit": git_head.stdout.strip() if git_head.returncode == 0 and not dirty else None,
        "base_commit": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "git_dirty": dirty,
    })

    with tempfile.TemporaryDirectory(prefix="forensight_kaggle_") as temporary:
        bundle_dir = Path(temporary)
        prepared = prepare_official_bundle(repo_root, kernel_path, bundle_dir, run_config)
        logger.info("Pushing sealed %s bundle %s", prepared["experiment"], prepared["source_payload_sha256"])
        # Inject self-contained bootstrap bundle and config into main.py for single-file Kaggle sandbox
        import base64
        import io
        import tarfile

        with io.BytesIO() as buf:
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                def _tar_filter(info):
                    if info.name.startswith("._") or "/._" in info.name or "__pycache__" in info.name or ".pyc" in info.name:
                        return None
                    return info
                for f_name in ["src", "configs"]:
                    tar.add(repo_root / f_name, arcname=f_name, filter=_tar_filter)
                for sc_name in ["check_dataset.py", "train_r2.py", "evaluate_r2.py", "aggregate_runs.py"]:
                    tar.add(repo_root / "scripts" / sc_name, arcname=f"scripts/{sc_name}", filter=_tar_filter)
                if not prepared.get("prepare_manifests") and (repo_root / "data" / "manifests").exists():
                    tar.add(repo_root / "data" / "manifests", arcname="data/manifests", filter=_tar_filter)
            bundle_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        main_py = bundle_dir / "main.py"
        main_content = main_py.read_text(encoding="utf-8")
        main_content = main_content.replace(
            "INJECTED_CONFIG: dict = {}",
            f"INJECTED_CONFIG: dict = json.loads({repr(json.dumps(prepared))})",
        )
        main_content = main_content.replace(
            'INJECTED_BUNDLE_B64: str = ""',
            f'INJECTED_BUNDLE_B64: str = """{bundle_b64}"""',
        )
        main_py.write_text(main_content, encoding="utf-8")

        result = subprocess.run(get_kaggle_cmd() + ["kernels", "push", "-p", str(bundle_dir)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0


def get_status(kernel_slug: str) -> str:
    cmd = get_kaggle_cmd() + ["kernels", "status", kernel_slug]
    res = subprocess.run(cmd, capture_output=True, text=True)
    status_output = res.stdout.strip()
    return status_output


def print_logs(kernel_slug: str) -> None:
    cmd = get_kaggle_cmd() + ["kernels", "logs", kernel_slug]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)


def download_outputs(kernel_slug: str, output_dir: str | Path) -> bool:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading outputs to %s...", out_dir)

    try:
        from kaggle import api
        import requests
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

        api.authenticate()
        owner_slug, kernel_name = kernel_slug.split("/")
        with api.build_kaggle_client() as kclient:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner_slug
            req.kernel_slug = kernel_name
            resp = kclient.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in resp.files or []:
                target_file = out_dir / item.file_name
                logger.info("Streaming download: %s", item.file_name)
                r = requests.get(item.url, stream=True, timeout=120)
                r.raise_for_status()
                with open(target_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                logger.info("Saved %s (%.2f MB)", item.file_name, target_file.stat().st_size / (1024 * 1024))
        unpack_downloaded_zips(out_dir)
        return True
    except Exception as e:
        logger.warning("Python streaming download failed (%s), falling back to CLI...", e)
        cmd = get_kaggle_cmd() + ["kernels", "output", kernel_slug, "-p", str(out_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        if res.returncode == 0:
            unpack_downloaded_zips(out_dir)
        return res.returncode == 0


def unpack_downloaded_zips(out_dir: Path) -> None:
    import zipfile
    for zip_file in list(out_dir.glob("*.zip")):
        logger.info("Unpacking archive %s into %s ...", zip_file.name, out_dir)
        with zipfile.ZipFile(zip_file, "r") as zf:
            names = zf.namelist()
            target_dir = out_dir
            if any(n.startswith("r2/") for n in names) and out_dir.resolve().name == "r2":
                target_dir = out_dir.parent
            zf.extractall(target_dir)



def watch_and_download(kernel_slug: str, output_dir: str | Path, poll_interval: int = 30) -> None:
    logger.info("Watching kernel '%s' until completion (poll interval: %ds)...", kernel_slug, poll_interval)
    while True:
        status_line = get_status(kernel_slug)
        logger.info("Status: %s", status_line)
        status_lower = status_line.lower()

        if "complete" in status_lower:
            logger.info("Kernel execution completed successfully! Downloading outputs...")
            download_outputs(kernel_slug, output_dir)
            break
        elif "error" in status_lower or "failed" in status_lower or "cancel" in status_lower:
            logger.error("Kernel ended with error: %s", status_line)
            logger.info("Fetching execution logs:")
            print_logs(kernel_slug)
            break

        time.sleep(poll_interval)


def main() -> int:
    args = parse_args()

    # If no action specified, default to status
    if not (args.push or args.status or args.logs or args.download or args.watch):
        args.status = True

    if args.push:
        has_seed = (args.seed is not None) or (args.seeds is not None)
        if not all((args.experiment in {"single", "logo", "all7"}, args.variant, has_seed, args.dataset_ref)):
            raise ValueError("--push requires --experiment {single,logo,all7}, --variant, --seed (or --seeds), and --dataset-ref")
        run_cfg: dict[str, Any] = {}
        if args.experiment:
            run_cfg["experiment"] = args.experiment
        if args.variant:
            run_cfg["variant"] = args.variant
        if args.leave_out:
            run_cfg["leave_out"] = args.leave_out
        if args.seeds is not None:
            run_cfg["seeds"] = args.seeds
        elif args.seed is not None:
            run_cfg["seed"] = args.seed
        if args.epochs is not None:
            run_cfg["epochs"] = args.epochs
        if args.batch_size is not None:
            run_cfg["batch_size"] = args.batch_size
        if args.learning_rate is not None:
            run_cfg["learning_rate"] = args.learning_rate
        run_cfg["dataset_ref"] = args.dataset_ref
        if args.variant == "all":
            raise ValueError(
                "--push runs exactly one variant; push fusion, semantic_only, and "
                "forensic_only separately so each run keeps its own provenance."
            )
        if args.prepare_manifests:
            run_cfg["prepare_manifests"] = True

        success = push_kernel(args.kernel_dir, config=run_cfg if run_cfg else None)
        if not success:
            return 1
        time.sleep(3)
        print(get_status(args.kernel_slug))

    if args.status and not args.push:
        print(get_status(args.kernel_slug))

    if args.logs:
        print_logs(args.kernel_slug)

    if args.watch:
        watch_and_download(args.kernel_slug, args.output_dir)

    if args.download and not args.watch:
        download_outputs(args.kernel_slug, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
