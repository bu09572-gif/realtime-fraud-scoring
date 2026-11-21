"""ULB credit card fraud data (Dal Pozzolo et al. 2015, Machine Learning Group, ULB; Open Database License):
284,807 card transactions by European cardholders over two days in September 2013, 492 of them fraud.
V1-V28 are PCA components (the bank could not release raw features); Time and Amount are raw.

The copy used is the one TensorFlow's imbalanced-data tutorial downloads; its SHA-256 is pinned below.

    python data.py     # -> data/raw/ulb_fraud/transactions.parquet
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[0]
RAW = Path(os.environ.get("ULB_DIR", ROOT / "data" / "raw" / "ulb_fraud"))
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
PCA = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA + ["log_amount", "hour"]


def load() -> pd.DataFrame:
    out = RAW / "transactions.parquet"
    if out.exists():
        return pd.read_parquet(out)
    RAW.mkdir(parents=True, exist_ok=True)
    raw = requests.get(URL, timeout=600).content
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise RuntimeError("creditcard.csv changed upstream; check it before using it")
    df = pd.read_csv(io.BytesIO(raw)).sort_values("Time", kind="stable").reset_index(drop=True)
    df["log_amount"] = np.log1p(df.Amount)
    df["hour"] = (df.Time // 3600) % 24
    # split by time, never at random: train on the past, tune on what follows, test on the latest 20%
    n = len(df)
    df["split"] = np.where(df.index < 0.6 * n, "train", np.where(df.index < 0.8 * n, "valid", "test"))
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    d = load()
    print(d.groupby("split", sort=False).agg(rows=("Class", "size"), fraud=("Class", "sum"),
                                              hours_from=("Time", lambda t: round(t.min() / 3600, 1)),
                                              hours_to=("Time", lambda t: round(t.max() / 3600, 1))))
