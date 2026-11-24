"""Drift monitoring: compare live traffic with the training data, feature by feature.

PSI (population stability index) per feature, on 10 bins cut at the training deciles:
    PSI = sum over bins of (live% - train%) * ln(live% / train%)
Common reading: < 0.1 stable, 0.1 to 0.25 some shift, > 0.25 large shift (look at it, maybe retrain).

The reference (bin edges and training shares, plus the training score distribution) is saved with
the model by the training pipeline, so the API can compute drift on its own traffic without the
training data.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4
BINS = 10


def reference(df: pd.DataFrame, features: list[str], scores: np.ndarray | None = None) -> dict:
    ref = {}
    cols = {f: df[f].to_numpy(dtype=float) for f in features}
    if scores is not None:
        cols["score"] = np.asarray(scores, dtype=float)
    for f, x in cols.items():
        edges = np.unique(np.quantile(x, np.linspace(0, 1, BINS + 1)[1:-1]))
        share = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1) / len(x)
        ref[f] = {"edges": edges.round(6).tolist(), "share": share.round(6).tolist()}
    return ref


def psi(ref: dict, x: np.ndarray) -> float:
    edges, exp = np.asarray(ref["edges"]), np.asarray(ref["share"])
    act = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(exp)) / max(len(x), 1)
    exp, act = np.clip(exp, EPS, None), np.clip(act, EPS, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi_table(ref: dict, df: pd.DataFrame) -> dict[str, float]:
    return {f: round(psi(r, df[f].to_numpy(dtype=float)), 4) for f, r in ref.items() if f in df}


class DriftMonitor:
    """Keeps the last `window` scored transactions and reports PSI against the training reference."""

    def __init__(self, ref: dict, window: int = 5000):
        self.ref = ref
        self.rows: deque = deque(maxlen=window)

    @classmethod
    def load(cls, path: Path, window: int = 5000) -> DriftMonitor:
        return cls(json.loads(Path(path).read_text()), window)

    def add(self, features: dict, score: float) -> None:
        self.rows.append({**features, "score": score})

    def report(self) -> dict:
        if not self.rows:
            return {"n": 0, "psi": {}}
        df = pd.DataFrame(list(self.rows))
        p = psi_table(self.ref, df)
        return {"n": len(df), "psi": p, "max_feature": max(p, key=p.get),
                "features_over_0.25": sorted(k for k, v in p.items() if v > 0.25)}
