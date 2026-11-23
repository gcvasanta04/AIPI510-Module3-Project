"""Step 1: download the raw Inside Airbnb listings file to data/raw/.

Usage:
    python -m src.data.download_raw
"""
from __future__ import annotations

import logging
import urllib.request

from src.config import load_config, resolve

log = logging.getLogger(__name__)


def download_raw(cfg: dict) -> None:
    url = cfg["data"]["source_url"]
    if "YYYY-MM-DD" in url:
        raise ValueError(
            "Set data.source_url in config.yaml to the current Washington, D.C. "
            "listings.csv.gz link from https://insideairbnb.com/get-the-data/"
        )
    out = resolve(cfg["data"]["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, out)
    urllib.request.urlretrieve(url, out)
    log.info("Saved %.1f MB", out.stat().st_size / 1e6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_raw(load_config())
