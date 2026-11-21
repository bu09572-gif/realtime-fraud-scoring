"""Fraud model, alert threshold and per-transaction reasons.

XGBoost scores each transaction. The alert threshold is picked on the validation period (the
lowest score that still gives the target precision there), then frozen for the test period.
Reasons are exact TreeSHAP contributions from XGBoost itself (pred_contribs), in log-odds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve

from data import FEATURES

PARAMS = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "max_depth": 4,
          "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
          "seed": 0, "nthread": 4}
TARGET_PRECISION = 0.80


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> xgb.Booster:
    dtr = xgb.DMatrix(train_df[FEATURES], label=train_df.Class, feature_names=FEATURES)
    dva = xgb.DMatrix(valid_df[FEATURES], label=valid_df.Class, feature_names=FEATURES)
    return xgb.train(PARAMS, dtr, num_boost_round=2000, evals=[(dva, "valid")], early_stopping_rounds=100,
                     verbose_eval=False)


def threshold_for_precision(y: np.ndarray, score: np.ndarray, target: float = TARGET_PRECISION) -> float:
    """Lowest threshold whose precision on (y, score) is at least target."""
    p, _, t = precision_recall_curve(y, score)
    ok = np.where(p[:-1] >= target)[0]
    return float(t[ok[0]]) if len(ok) else float(t[-1])


@dataclass
class Scorer:
    booster: xgb.Booster
    threshold: float

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        it = (0, self.booster.best_iteration + 1)
        return self.booster.predict(xgb.DMatrix(rows[FEATURES], feature_names=FEATURES), iteration_range=it)

    def explain(self, rows: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Top features pushing each score up or down, as SHAP values in log-odds."""
        it = (0, self.booster.best_iteration + 1)
        c = self.booster.predict(xgb.DMatrix(rows[FEATURES], feature_names=FEATURES), pred_contribs=True,
                                 iteration_range=it)[:, :-1]          # last column is the bias
        out = []
        for row, vals in zip(c, rows[FEATURES].to_numpy()):
            idx = np.argsort(-np.abs(row))[:top]
            out.append([{"feature": FEATURES[i], "value": round(float(vals[i]), 4),
                         "contribution": round(float(row[i]), 4)} for i in idx])
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path / "fraud_xgb.json")
        (path / "scorer.json").write_text(json.dumps({"threshold": self.threshold,
                                                      "best_iteration": self.booster.best_iteration,
                                                      "features": FEATURES}, indent=2))

    @classmethod
    def load(cls, path: Path) -> Scorer:
        meta = json.loads((path / "scorer.json").read_text())
        b = xgb.Booster()
        b.load_model(path / "fraud_xgb.json")      # the saved model keeps best_iteration
        assert b.best_iteration == meta["best_iteration"]
        return cls(b, meta["threshold"])
