"""Train on the first 60% of time, pick the alert threshold on the next 20%, test on the last 20%.

    python run.py      # -> model/, results/metrics.json, results/pr_curve.png
"""
from __future__ import annotations

import json
from pathlib import Path

import data
import model
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def at_threshold(df: pd.DataFrame, score: np.ndarray, thr: float) -> dict:
    alert = score >= thr
    y = df.Class.to_numpy().astype(bool)
    tp = int((alert & y).sum())
    return {"threshold": round(thr, 6), "alerts": int(alert.sum()), "fraud_caught": tp, "fraud_total": int(y.sum()),
            "precision": round(tp / max(alert.sum(), 1), 3), "recall": round(tp / y.sum(), 3),
            "alerts_per_10k_transactions": round(alert.mean() * 1e4, 2),
            "fraud_amount_caught_share": round(df.Amount[alert & y].sum() / df.Amount[y].sum(), 3)}


def bootstrap_ap(y: np.ndarray, s: np.ndarray, n: int = 1000) -> list[float]:
    """95% interval for average precision, resampling test transactions."""
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            vals.append(average_precision_score(y[i], s[i]))
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


def evaluate(name: str, valid: pd.DataFrame, test: pd.DataFrame, sv: np.ndarray, st: np.ndarray) -> dict:
    thr = model.threshold_for_precision(valid.Class.to_numpy(), sv)
    y = test.Class.to_numpy()
    return {"model": name,
            "test_average_precision": round(average_precision_score(y, st), 4),
            "test_average_precision_95ci": bootstrap_ap(y, st),
            "test_roc_auc": round(roc_auc_score(y, st), 4),
            "valid_average_precision": round(average_precision_score(valid.Class, sv), 4),
            f"alert_rule_precision_{model.TARGET_PRECISION:.0%}_on_valid": {
                "valid": at_threshold(valid, sv, thr), "test": at_threshold(test, st, thr)}}


def chart(test: pd.DataFrame, scores: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(test.Class, s)
        ax.plot(r, p, lw=2, label=f"{name} (AP {average_precision_score(test.Class, s):.3f})")
    ax.axhline(test.Class.mean(), color="#999", ls=":", label=f"random ({test.Class.mean():.4f})")
    ax.set_xlabel("recall (share of fraud caught)")
    ax.set_ylabel("precision (share of alerts that are fraud)")
    ax.set_title("Test period: last 20% of transactions")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "pr_curve.png", dpi=120)
    plt.close(fig)


def main():
    df = data.load()
    tr, va, te = (df[df.split == s] for s in ["train", "valid", "test"])
    OUT.mkdir(exist_ok=True)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    lr.fit(tr[data.FEATURES], tr.Class)
    lr_v, lr_t = lr.predict_proba(va[data.FEATURES])[:, 1], lr.predict_proba(te[data.FEATURES])[:, 1]

    booster = model.train(tr, va)
    s = model.Scorer(booster, 0.0)
    xv, xt = s.score(va), s.score(te)
    s.threshold = model.threshold_for_precision(va.Class.to_numpy(), xv)
    s.save(HERE / "model")

    # which features drive alerts on the test period: mean |SHAP| over alerted transactions
    alerted = te[xt >= s.threshold]
    reasons = pd.Series([r["feature"] for rs in s.explain(alerted, top=1) for r in rs]).value_counts()

    res = {"data": {split: {"transactions": len(d), "fraud": int(d.Class.sum()),
                            "hours": [round(d.Time.min() / 3600, 1), round(d.Time.max() / 3600, 1)]}
                    for split, d in [("train", tr), ("valid", va), ("test", te)]},
           "xgboost": evaluate("xgboost", va, te, xv, xt) | {"trees": booster.best_iteration + 1},
           "logistic_regression": evaluate("logistic regression", va, te, lr_v, lr_t),
           "top_reason_on_test_alerts": reasons.to_dict()}
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    chart(te, {"XGBoost": xt, "logistic regression": lr_t})
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
