"""Step 4: ingest -> split -> train -> evaluate -> log to MLflow -> save model.

Usage:
    python -m src.train                    # uses config.yaml
    python -m src.train --model ridge      # quick override for comparison runs
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import flatten, load_config, resolve  # noqa: E402
from src.data import storage  # noqa: E402
from src.features import build_pipeline, feature_columns  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- data
def ingest(cfg: dict) -> pd.DataFrame:
    if cfg["ingest"]["source"] == "s3":
        path = storage.download(cfg)
    else:
        path = resolve(cfg["data"]["clean_path"])
    log.info("Loading %s", path)
    return pd.read_csv(path)


def split(df: pd.DataFrame, cfg: dict):
    s, seed = cfg["split"], cfg["project"]["random_state"]
    train_val, test = train_test_split(df, test_size=s["test_size"], random_state=seed)
    val_frac = s["val_size"] / (1 - s["test_size"])
    train, val = train_test_split(train_val, test_size=val_frac, random_state=seed)
    return train, val, test


# ---------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


def predict_dollars(pipe, X, log_target: bool) -> np.ndarray:
    p = pipe.predict(X)
    return np.expm1(p) if log_target else p


# ---------------------------------------------------------------- plots
def save_plots(y_true, y_pred, out_dir: Path) -> list[Path]:
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    lim = [0, max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="Actual price ($)", ylabel="Predicted price ($)", title="Validation: predicted vs actual")
    fig.tight_layout()
    p = out_dir / "pred_vs_actual.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_pred - y_true, bins=50)
    ax.set(xlabel="Prediction error ($)", ylabel="Listings", title="Validation residuals")
    fig.tight_layout()
    p = out_dir / "residuals.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- main
def train(cfg: dict) -> dict:
    target = cfg["features"]["target"]
    log_target = cfg["features"]["log_target"]
    cols = feature_columns(cfg)

    df = ingest(cfg)
    train_df, val_df, test_df = split(df, cfg)
    log.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    y_train = train_df[target].to_numpy()
    y_fit = np.log1p(y_train) if log_target else y_train

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{cfg['model']['type']}") as run:
        mlflow.log_params({k: v for k, v in flatten(cfg).items()
                           if not k.startswith(("data.", "mlflow."))})
        mlflow.log_params({"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)})

        pipe = build_pipeline(cfg)
        pipe.fit(train_df[cols], y_fit)

        # Baseline: always predict the training median. The model should beat this.
        median = float(np.median(y_train))
        metrics = regression_metrics(val_df[target], np.full(len(val_df), median), "baseline_val")

        val_pred = predict_dollars(pipe, val_df[cols], log_target)
        test_pred = predict_dollars(pipe, test_df[cols], log_target)
        metrics |= regression_metrics(val_df[target], val_pred, "val")
        metrics |= regression_metrics(test_df[target], test_pred, "test")
        mlflow.log_metrics(metrics)
        for k, v in metrics.items():
            log.info("%-18s %.3f", k, v)

        # Empirical prediction interval from validation residuals (in log space so it
        # scales with price). Reported by the API as a "typical range".
        cov = cfg["output"]["interval_coverage"]
        if log_target:
            resid = np.log1p(val_df[target].to_numpy()) - np.log1p(val_pred)
        else:
            resid = val_df[target].to_numpy() - val_pred
        lo_q, hi_q = np.quantile(resid, [(1 - cov) / 2, 1 - (1 - cov) / 2])

        out_dir = resolve(cfg["output"]["model_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / cfg["output"]["model_file"]
        joblib.dump(pipe, model_path)

        metadata = {
            "model_type": cfg["model"]["type"],
            "mlflow_run_id": run.info.run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_target": log_target,
            "feature_columns": cols,
            "interval": {"coverage": cov, "low_resid": float(lo_q), "high_resid": float(hi_q)},
            "metrics": metrics,
            "n_train": len(train_df),
            # Allowed values / ranges -> the front end builds its form from these.
            "categories": {c: sorted(df[c].dropna().astype(str).unique().tolist())
                           for c in cfg["features"]["categorical"]},
            "numeric_ranges": {
                c: {"min": float(df[c].min()), "max": float(df[c].max()),
                    "median": float(df[c].median())}
                for c in cfg["features"]["numeric"]
            },
        }
        meta_path = out_dir / cfg["output"]["metadata_file"]
        meta_path.write_text(json.dumps(metadata, indent=2))

        with tempfile.TemporaryDirectory() as tmp:
            for p in save_plots(val_df[target].to_numpy(), val_pred, Path(tmp)):
                mlflow.log_artifact(str(p), artifact_path="plots")
        mlflow.log_artifact(str(meta_path))
        mlflow.log_artifact(str(resolve("config.yaml")))
        # cloudpickle: we trust our own model, and skops' allow-list differs per model type.
        mlflow.sklearn.log_model(pipe, name="model", input_example=train_df[cols].head(3),
                                 serialization_format="cloudpickle")

        log.info("Saved model to %s (MLflow run %s)", model_path, run.info.run_id)
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hist_gb", "random_forest", "ridge"])
    parser.add_argument("--source", choices=["s3", "local"])
    args = parser.parse_args()
    cfg = load_config()
    if args.model:
        cfg["model"]["type"] = args.model
    if args.source:
        cfg["ingest"]["source"] = args.source
    train(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
