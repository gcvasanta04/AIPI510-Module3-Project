import pandas as pd

from src.config import load_config
from src.data.clean import clean, parse_bathrooms_text, parse_price
from tests.fake_data import make_fake_raw


def test_parse_price():
    out = parse_price(pd.Series(["$1,234.00", "$85.00", None, ""]))
    assert out.iloc[0] == 1234.0 and out.iloc[1] == 85.0
    assert out.iloc[2:].isna().all()


def test_parse_bathrooms_text():
    count, shared = parse_bathrooms_text(pd.Series(["1.5 shared baths", "Half-bath", "2 baths", None]))
    assert count.tolist()[:3] == [1.5, 0.5, 2.0]
    assert shared.tolist() == [True, False, False, False]


def test_clean_end_to_end():
    cfg = load_config()
    df = clean(make_fake_raw(), cfg)
    c = cfg["cleaning"]
    assert df["id"].is_unique
    assert df["price"].between(c["min_price"], c["max_price"]).all()
    assert df["minimum_nights"].le(c["max_minimum_nights"]).all()
    assert "Tiny Rare Neighbourhood" not in set(df["neighbourhood"])
    assert df["bathrooms"].notna().all()
    expected = set(cfg["features"]["numeric"] + cfg["features"]["boolean"]
                   + cfg["features"]["categorical"] + [cfg["features"]["target"]])
    assert expected <= set(df.columns)
