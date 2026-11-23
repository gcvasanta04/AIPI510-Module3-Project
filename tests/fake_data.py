"""Generate a fake raw file with the Inside Airbnb schema (and its quirks).

Used by the tests and for running the pipeline offline:
    python -m tests.fake_data data/raw/listings.csv.gz
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

NEIGHBOURHOODS = [
    "Capitol Hill, Lincoln Park", "Dupont Circle, Connecticut Avenue/K Street",
    "Union Station, Stanton Park, Kingman Park", "Columbia Heights, Mt. Pleasant, Pleasant Plains, Park View",
    "Shaw, Logan Circle", "Georgetown, Burleith/Hillandale", "Brightwood Park, Crestwood, Petworth",
    "Navy Yard, Fort McNair, Buzzard Point", "Tiny Rare Neighbourhood",
]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]


def make_fake_raw(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hood = rng.choice(NEIGHBOURHOODS[:-1], n)
    hood[:3] = NEIGHBOURHOODS[-1]  # a rare category for the "Other" logic
    room = rng.choice(ROOM_TYPES, n, p=[0.7, 0.25, 0.03, 0.02])
    acc = rng.integers(1, 9, n)
    bedrooms = np.clip(np.round(acc / 2 + rng.normal(0, 0.5, n)), 0, None)
    baths = rng.choice([1, 1.5, 2, 2.5], n)
    shared = rng.random(n) < 0.1
    hood_effect = {h: i * 12 for i, h in enumerate(NEIGHBOURHOODS)}
    price = (60 + 25 * acc + 20 * bedrooms + np.vectorize(hood_effect.get)(hood)
             - 50 * (room != "Entire home/apt") + rng.normal(0, 30, n)).clip(15)

    bathrooms_text = [
        f"{b:g} {'shared ' if s else ''}bath{'s' if b != 1 else ''}" for b, s in zip(baths, shared)
    ]
    bathrooms_text[5] = "Half-bath"
    df = pd.DataFrame({
        "id": np.arange(n),
        "price": [f"${p:,.2f}" for p in price],
        "room_type": room,
        "neighbourhood_cleansed": hood,
        "accommodates": acc,
        "bedrooms": bedrooms,
        "beds": acc,
        "bathrooms": np.nan,  # often empty in real scrapes -> rebuilt from text
        "bathrooms_text": bathrooms_text,
        "minimum_nights": rng.choice([1, 2, 3, 30, 400], n, p=[0.3, 0.3, 0.2, 0.18, 0.02]),
        "number_of_reviews": rng.poisson(30, n),
        "review_scores_rating": np.round(rng.uniform(3.5, 5, n), 2),
        "host_is_superhost": rng.choice(["t", "f", None], n),
        "instant_bookable": rng.choice(["t", "f"], n),
        "description": "filler column the cleaner should drop",
    })
    # Real-world mess: missing prices, outliers, missing ratings, duplicates.
    df.loc[df.sample(frac=0.15, random_state=seed).index, "price"] = None
    df.loc[10, "price"] = "$25,000.00"
    df.loc[df.sample(frac=0.2, random_state=seed + 1).index, "review_scores_rating"] = np.nan
    return pd.concat([df, df.head(5)], ignore_index=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/listings.csv.gz"
    make_fake_raw().to_csv(out, index=False)
    print(f"wrote {out}")
