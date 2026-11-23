"""Step 2: clean the raw Inside Airbnb file into a small, model-ready CSV.

Only deterministic, row-level cleaning happens here (parsing, type fixes,
filtering). Anything learned from the data — imputation values, encodings —
lives inside the sklearn pipeline in src/features.py so it is fit on the
training split only (no leakage).

Usage:
    python -m src.data.clean
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.config import load_config, resolve

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "id",
    "price",
    "room_type",
    "neighbourhood_cleansed",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathrooms_text",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "host_is_superhost",
    "instant_bookable",
]


def parse_price(s: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0; blanks -> NaN."""
    return pd.to_numeric(
        s.astype("string").str.replace(r"[$,\s]", "", regex=True), errors="coerce"
    )


def parse_bathrooms_text(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'1.5 shared baths' -> (1.5, True); 'Half-bath' -> (0.5, False)."""
    text = s.astype("string").str.lower()

    def to_num(t):
        if t is pd.NA or t is None:
            return np.nan
        m = re.search(r"\d+(\.\d+)?", t)
        if m:
            return float(m.group())
        if "half" in t:
            return 0.5
        return np.nan

    count = text.map(to_num).astype(float)
    shared = text.str.contains("shared", na=False).astype(bool)
    return count, shared


def parse_bool(s: pd.Series) -> pd.Series:
    """Inside Airbnb uses 't'/'f'. Unknown -> False."""
    return s.astype("string").str.lower().eq("t").fillna(False).astype(bool)


def clean(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["cleaning"]
    missing = [col for col in RAW_COLUMNS if col not in raw.columns]
    if missing:
        # Older scrapes lack the numeric `bathrooms` column; we can rebuild it.
        if missing == ["bathrooms"]:
            raw = raw.assign(bathrooms=np.nan)
        else:
            raise KeyError(f"Raw file is missing expected columns: {missing}")

    df = raw[RAW_COLUMNS].copy()
    n0 = len(df)

    df = df.drop_duplicates(subset="id")
    df["price"] = parse_price(df["price"])

    bath_count, shared = parse_bathrooms_text(df["bathrooms_text"])
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(bath_count)
    df["shared_bath"] = shared

    df["host_is_superhost"] = parse_bool(df["host_is_superhost"])
    df["instant_bookable"] = parse_bool(df["instant_bookable"])

    for col in ["accommodates", "bedrooms", "beds", "minimum_nights",
                "number_of_reviews", "review_scores_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"neighbourhood_cleansed": "neighbourhood"})
    df["neighbourhood"] = df["neighbourhood"].astype("string").str.strip()
    df["room_type"] = df["room_type"].astype("string").str.strip()

    # --- filtering ---
    before = len(df)
    df = df[df["price"].notna()]
    log.info("Dropped %d rows with no price", before - len(df))

    before = len(df)
    df = df[df["price"].between(c["min_price"], c["max_price"])]
    log.info("Dropped %d price outliers outside [%s, %s]", before - len(df),
             c["min_price"], c["max_price"])

    before = len(df)
    df = df[df["minimum_nights"].le(c["max_minimum_nights"])]
    df = df[df["accommodates"].gt(0) & df["room_type"].notna() & df["neighbourhood"].notna()]
    log.info("Dropped %d rows with invalid nights/guests/category", before - len(df))

    # Collapse rare neighbourhoods so the API/front end only offers well-supported values.
    counts = df["neighbourhood"].value_counts()
    rare = counts[counts < c["min_neighbourhood_count"]].index
    df.loc[df["neighbourhood"].isin(rare), "neighbourhood"] = "Other"

    df = df.drop(columns=["bathrooms_text"]).reset_index(drop=True)
    log.info("Cleaned: %d -> %d rows", n0, len(df))
    return df


def main() -> None:
    cfg = load_config()
    raw_path = resolve(cfg["data"]["raw_path"])
    out = resolve(cfg["data"]["clean_path"])
    log.info("Reading %s", raw_path)
    raw = pd.read_csv(raw_path, low_memory=False)
    df = clean(raw, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
