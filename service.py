"""Scoring API: one transaction in, a fraud probability, an alert decision and the top reasons out.

    uvicorn service:app --port 8000
    curl -X POST localhost:8000/score -H 'content-type: application/json' -d @example.json

Endpoints
  POST /score              one transaction
  POST /score/batch        up to 1,000 transactions
  GET  /health             liveness + which model version is loaded
  GET  /model              the model card (version, training data hash, metrics, threshold)
  GET  /monitoring/drift   PSI of recent traffic vs the training data, per feature and for the score
  GET  /metrics            Prometheus metrics (latency, throughput, alert rate, scores, drift, cost)

Cost: COST_PER_HOUR_USD (the price of the machine the API runs on) is exposed as a metric, so the
dashboard can show cost per 1,000 predictions from the live request rate.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from pydantic import BaseModel, Field

from model import Scorer
from monitoring import DriftMonitor

MODEL_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", Path(__file__).resolve().parent / "model"))
DRIFT_EVERY = int(os.environ.get("DRIFT_EVERY", "500"))          # recompute PSI every N transactions
COST_PER_HOUR = float(os.environ.get("COST_PER_HOUR_USD", "0"))

scorer = Scorer.load(MODEL_DIR)
scorer.booster.set_param({"nthread": 1})
CARD = json.loads((MODEL_DIR / "card.json").read_text()) if (MODEL_DIR / "card.json").exists() else {}
drift = DriftMonitor.load(MODEL_DIR / "reference.json", window=int(os.environ.get("DRIFT_WINDOW", "5000"))) \
    if (MODEL_DIR / "reference.json").exists() else None

REQ_LATENCY = Histogram("fraud_request_seconds", "End-to-end request time inside the API", ["endpoint"],
                        buckets=(.001, .0025, .005, .0075, .01, .015, .02, .03, .05, .1, .25, .5, 1))
SCORE_LATENCY = Histogram("fraud_scoring_seconds", "Model scoring + SHAP time per transaction",
                          buckets=(.0005, .001, .002, .003, .005, .0075, .01, .02, .05))
PREDICTIONS = Counter("fraud_predictions_total", "Scored transactions", ["alert"])
SCORES = Histogram("fraud_score", "Fraud probability of scored transactions",
                   buckets=(.001, .01, .05, .1, .2, .3, .5, .7, .9, .99, 1))
AMOUNT = Counter("fraud_alert_amount_total", "Transaction amount of alerted transactions")
ERRORS = Counter("fraud_request_errors_total", "Requests rejected or failed", ["status"])
PSI = Gauge("fraud_feature_psi", "PSI of the last window vs training data", ["feature"])
COST = Gauge("fraud_cost_per_hour_usd", "Price of the machine serving the API")
MODEL = Info("fraud_model", "Model being served")
COST.set(COST_PER_HOUR)
MODEL.info({"name": str(CARD.get("name", "fraud-xgb")), "version": str(CARD.get("version", "local")),
            "candidate": str(CARD.get("candidate", "")), "threshold": f"{scorer.threshold:.6f}"})

_seen = 0
app = FastAPI(title="Card fraud scoring", version=str(CARD.get("version", "local")))


class Transaction(BaseModel):
    transaction_id: str | None = None
    time_s: float = Field(description="seconds since the start of the data period")
    amount: float = Field(ge=0)
    pca: list[float] = Field(min_length=28, max_length=28, description="V1..V28")


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


def features(ts: list[Transaction]) -> pd.DataFrame:
    rows = []
    for t in ts:
        row = {f"V{i}": v for i, v in enumerate(t.pca, 1)}
        row["log_amount"] = math.log1p(t.amount)
        row["hour"] = (t.time_s // 3600) % 24
        rows.append(row)
    return pd.DataFrame(rows)


def _score(ts: list[Transaction]) -> list[dict]:
    t0 = time.perf_counter()
    x = features(ts)
    p = scorer.score(x)
    reasons = scorer.explain(x)
    per = (time.perf_counter() - t0) / len(ts)
    out = []
    for t, prob, r, row in zip(ts, p, reasons, x.to_dict("records")):
        prob = float(prob)
        alert = prob >= scorer.threshold
        SCORE_LATENCY.observe(per)
        PREDICTIONS.labels(alert=str(alert).lower()).inc()
        SCORES.observe(prob)
        if alert:
            AMOUNT.inc(t.amount)
        if drift is not None:
            drift.add(row, prob)
        out.append({"transaction_id": t.transaction_id, "fraud_probability": round(prob, 6), "alert": alert,
                    "threshold": scorer.threshold, "reasons": r})
    global _seen
    before, _seen = _seen, _seen + len(ts)
    if drift is not None and _seen // DRIFT_EVERY > before // DRIFT_EVERY:
        for f, v in drift.report()["psi"].items():
            PSI.labels(feature=f).set(v)
    return out


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    if request.url.path.startswith("/score"):
        REQ_LATENCY.labels(endpoint=request.url.path).observe(time.perf_counter() - t0)
        if resp.status_code >= 400:
            ERRORS.labels(status=str(resp.status_code)).inc()
    return resp


@app.post("/score")
def score(t: Transaction) -> dict:
    t0 = time.perf_counter()
    r = _score([t])[0]
    return r | {"model_version": CARD.get("version"), "server_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/score/batch")
def score_batch(b: Batch) -> dict:
    t0 = time.perf_counter()
    res = _score(b.transactions)
    return {"results": res, "model_version": CARD.get("version"),
            "server_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_version": CARD.get("version"), "threshold": scorer.threshold}


@app.get("/model")
def model_card() -> dict:
    return CARD


@app.get("/monitoring/drift")
def drift_report() -> dict:
    return drift.report() if drift is not None else {"n": 0, "psi": {}, "note": "no reference.json"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
