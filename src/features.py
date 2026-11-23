"""Feature preprocessing + model construction, driven by config.yaml."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def to_float(x):
    """Module-level (not a lambda) so the fitted pipeline can be pickled."""
    return x.astype(float)


def feature_columns(cfg: dict) -> list[str]:
    f = cfg["features"]
    return f["numeric"] + f["boolean"] + f["categorical"]


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    f = cfg["features"]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    boolean = FunctionTransformer(to_float, feature_names_out="one-to-one")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, f["numeric"]),
        ("bool", boolean, f["boolean"]),
        ("cat", categorical, f["categorical"]),
    ])


def build_model(cfg: dict):
    kind = cfg["model"]["type"]
    params = dict(cfg["model"]["params"].get(kind, {}))
    seed = cfg["project"]["random_state"]
    if kind == "hist_gb":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    if kind == "random_forest":
        return RandomForestRegressor(random_state=seed, **params)
    if kind == "ridge":
        return Ridge(**params)
    raise ValueError(f"Unknown model.type: {kind}")


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(cfg)), ("model", build_model(cfg))])
