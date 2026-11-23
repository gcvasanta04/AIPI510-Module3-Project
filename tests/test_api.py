"""Trains a small model on fake data into a temp dir, then exercises the API."""
import importlib

import pytest
from fastapi.testclient import TestClient

from src.config import load_config
from src.data.clean import clean
from tests.fake_data import make_fake_raw


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    tmp = tmp_path_factory.mktemp("run")
    cfg = load_config()
    clean_path = tmp / "clean.csv"
    clean(make_fake_raw(), cfg).to_csv(clean_path, index=False)
    cfg["data"]["clean_path"] = str(clean_path)
    cfg["ingest"]["source"] = "local"
    cfg["output"]["model_dir"] = str(tmp / "models")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp / 'mlflow.db'}"
    cfg["model"]["params"]["hist_gb"]["max_iter"] = 50

    from src import train
    train.train(cfg)

    monkeypatch_module.setenv("MODEL_DIR", str(tmp / "models"))
    import src.serve.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_metadata(client):
    meta = client.get("/metadata").json()
    assert "neighbourhood" in meta["categories"]
    assert "val_rmse" in meta["metrics"]


def test_predict(client):
    hood = client.get("/metadata").json()["categories"]["neighbourhood"][0]
    r = client.post("/predict", json={
        "neighbourhood": hood, "room_type": "Entire home/apt", "accommodates": 4,
        "bedrooms": 2, "bathrooms": 1,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["range_low"] <= body["predicted_price"] <= body["range_high"]


def test_predict_rejects_bad_input(client):
    r = client.post("/predict", json={"neighbourhood": "Atlantis", "room_type": "Castle", "accommodates": 0})
    assert r.status_code == 422
