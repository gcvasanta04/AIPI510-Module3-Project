"""Step 3: move the cleaned dataset to/from S3.

Usage:
    python -m src.data.storage upload     # local clean CSV -> S3
    python -m src.data.storage download   # S3 -> local (what training calls)

Credentials come from the standard AWS chain (env vars, ~/.aws, or an IAM role).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import boto3

from src.config import load_config, resolve

log = logging.getLogger(__name__)


def _client(cfg: dict):
    return boto3.client("s3", region_name=cfg["data"]["s3"]["region"])


def upload(cfg: dict) -> None:
    s3cfg = cfg["data"]["s3"]
    local = resolve(cfg["data"]["clean_path"])
    log.info("Uploading %s -> s3://%s/%s", local, s3cfg["bucket"], s3cfg["key"])
    _client(cfg).upload_file(str(local), s3cfg["bucket"], s3cfg["key"])


def download(cfg: dict) -> Path:
    s3cfg = cfg["data"]["s3"]
    local = resolve(cfg["data"]["clean_path"])
    local.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading s3://%s/%s -> %s", s3cfg["bucket"], s3cfg["key"], local)
    _client(cfg).download_file(s3cfg["bucket"], s3cfg["key"], str(local))
    return local


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upload", "download"])
    args = parser.parse_args()
    cfg = load_config()
    upload(cfg) if args.action == "upload" else download(cfg)
