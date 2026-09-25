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
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_on_kaggle")

DEFAULT_KERNEL_DIR = "deploy/kaggle"
DEFAULT_KERNEL_SLUG = "truongnhatnguyen2805/forensight-r2-fusion-training"
DEFAULT_OUTPUT_DIR = "results/r2_kaggle"


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
    return parser.parse_args()


def get_kaggle_cmd() -> list[str]:
    # Look for virtualenv kaggle first
    venv_kaggle = Path(".venv/bin/kaggle")
    if venv_kaggle.exists():
        return [str(venv_kaggle)]
    return ["kaggle"]


def push_kernel(kernel_dir: str | Path) -> bool:
    cmd = get_kaggle_cmd() + ["kernels", "push", "-p", str(kernel_dir)]
    logger.info("Pushing kernel to Kaggle: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    return res.returncode == 0


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
        return True
    except Exception as e:
        logger.warning("Python streaming download failed (%s), falling back to CLI...", e)
        cmd = get_kaggle_cmd() + ["kernels", "output", kernel_slug, "-p", str(out_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        return res.returncode == 0



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
        success = push_kernel(args.kernel_dir)
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
