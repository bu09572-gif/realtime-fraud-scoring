"""Scoring API: one transaction in, a fraud probability, an alert decision and the top reasons out.

    uvicorn service:app --port 8000
    curl -X POST localhost:8000/score -H 'content-type: application/json' -d @example.json
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import Scorer

MODEL_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", Path(__file__).resolve().parent / "model"))
scorer = Scorer.load(MODEL_DIR)
app = FastAPI(title="Card fraud scoring")


class Transaction(BaseModel):
    transaction_id: str | None = None
    time_s: float = Field(description="seconds since the start of the data period")
    amount: float = Field(ge=0)
    pca: list[float] = Field(min_length=28, max_length=28, description="V1..V28")


def features(t: Transaction) -> pd.DataFrame:
    row = {f"V{i}": v for i, v in enumerate(t.pca, 1)}
    row["log_amount"] = math.log1p(t.amount)
    row["hour"] = (t.time_s // 3600) % 24
    return pd.DataFrame([row])


@app.post("/score")
def score(t: Transaction) -> dict:
    t0 = time.perf_counter()
    x = features(t)
    p = float(scorer.score(x)[0])
    reasons = scorer.explain(x)[0]
    return {"transaction_id": t.transaction_id, "fraud_probability": round(p, 6),
            "alert": p >= scorer.threshold, "threshold": scorer.threshold, "reasons": reasons,
            "server_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "threshold": scorer.threshold}
