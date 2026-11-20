"""Tests on the real ULB data and the committed model."""
import data
import model
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def df():
    return data.load()


@pytest.fixture(scope="module")
def scorer():
    return model.Scorer.load(model.Path(__file__).resolve().parent / "model")


def test_split_is_by_time(df):
    assert len(df) == 284_807 and df.Class.sum() == 492
    t = df.groupby("split").Time.agg(["min", "max"])
    assert t.loc["train", "max"] <= t.loc["valid", "min"] and t.loc["valid", "max"] <= t.loc["test", "min"]


def test_threshold_for_precision():
    y = np.array([0, 0, 1, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    assert model.threshold_for_precision(y, s, 1.0) == pytest.approx(0.8)
    assert model.threshold_for_precision(y, s, 0.75) == pytest.approx(0.3)


def test_shap_values_add_up_to_the_score(df, scorer):
    rows = df[df.split == "test"].head(50)
    p = scorer.score(rows)
    it = (0, scorer.booster.best_iteration + 1)
    c = scorer.booster.predict(model.xgb.DMatrix(rows[data.FEATURES], feature_names=data.FEATURES),
                               pred_contribs=True, iteration_range=it)
    assert np.allclose(1 / (1 + np.exp(-c.sum(axis=1))), p, atol=1e-5)
    reasons = scorer.explain(rows.head(1))[0]
    assert len(reasons) == 3 and abs(reasons[0]["contribution"]) >= abs(reasons[1]["contribution"])


def test_api_matches_batch_scoring(df, scorer):
    import replay
    import service
    client = TestClient(service.app)
    rows = df[(df.split == "test")]
    rows = model.pd.concat([rows[rows.Class == 1].head(5), rows[rows.Class == 0].head(5)])
    batch = scorer.score(rows)
    for row, p in zip(rows.itertuples(), batch):
        r = client.post("/score", json=replay.payload(row)).json()
        assert r["fraud_probability"] == pytest.approx(p, abs=1e-5)
        assert r["alert"] == (p >= scorer.threshold) and len(r["reasons"]) == 3


def test_api_rejects_bad_input():
    import service
    client = TestClient(service.app)
    assert client.post("/score", json={"time_s": 1, "amount": -5, "pca": [0] * 28}).status_code == 422
    assert client.post("/score", json={"time_s": 1, "amount": 5, "pca": [0] * 27}).status_code == 422
