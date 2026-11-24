"""Training pipeline: ingest -> validate -> train candidates -> evaluate -> promotion gate -> export.

    python pipeline.py                 # full run; MLflow tracking + model registry in ./mlflow.db
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse runs, metrics and model versions

Every candidate is logged to MLflow (parameters, metrics, the model, its PR curve). The gate decides
on the VALIDATION period only. The test period is scored afterwards and reported, never used to pick.

Candidates:
  xgb_baseline         the original model (XGBoost, depth 4)
  xgb_amount_weighted  same, but fraud rows are weighted by log(amount), so large frauds count more.
                       This targets the known weakness: the baseline catches 73% of frauds but only
                       53% of the fraud amount

Gate (all must pass for a candidate to replace the champion):
  - valid average precision no more than 0.01 below the champion's
  - valid share of fraud amount caught at the alert threshold at least the champion's
  - alert precision on valid at least 0.75 (the alert rule targets 0.80)
  - p95 single-transaction scoring + explanation time under 20 ms

The winner is registered as a new version of "fraud-xgb" with the alias "champion" and exported to
model/ (booster, threshold, model card, drift reference), which is what the API container serves.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import data
import model
from monitoring import reference
from run import at_threshold

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
OUT = HERE / "results"
REGISTERED = "fraud-xgb"
GATE = {"max_ap_drop": 0.01, "min_valid_precision": 0.75, "max_p95_ms": 20.0}


def ingest() -> tuple[pd.DataFrame, dict]:
    df = data.load()
    sha = data.SHA256
    return df, {"rows": len(df), "source_sha256": sha}


def validate(df: pd.DataFrame) -> dict:
    """Data checks before training. Any failure stops the pipeline."""
    checks = {
        "columns_present": all(c in df for c in data.FEATURES + ["Class", "Amount", "Time", "split"]),
        "no_missing_values": bool(df[data.FEATURES + ["Class"]].notna().all().all()),
        "label_is_binary": set(df.Class.unique()) <= {0, 1},
        "amount_not_negative": bool((df.Amount >= 0).all()),
        "time_sorted": bool(df.Time.is_monotonic_increasing),
        "fraud_rate_in_expected_range": 0.0005 <= df.Class.mean() <= 0.01,
        "splits_in_time_order": bool(df.groupby("split").Time.max()["train"] <= df.groupby("split").Time.min()["valid"]
                                     <= df.groupby("split").Time.max()["valid"] <= df.groupby("split").Time.min()["test"]),
        "every_split_has_fraud": bool((df.groupby("split").Class.sum() > 0).all()),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit(f"data validation failed: {failed}")
    return checks


def train_candidate(name: str, tr: pd.DataFrame, va: pd.DataFrame) -> xgb.Booster:
    w = None
    if name == "xgb_amount_weighted":
        w = np.where(tr.Class == 1, 1 + np.log1p(tr.Amount), 1.0)
    dtr = xgb.DMatrix(tr[data.FEATURES], label=tr.Class, weight=w, feature_names=data.FEATURES)
    dva = xgb.DMatrix(va[data.FEATURES], label=va.Class, feature_names=data.FEATURES)
    return xgb.train(model.PARAMS, dtr, num_boost_round=2000, evals=[(dva, "valid")], early_stopping_rounds=100,
                     verbose_eval=False)


def latency_ms(s: model.Scorer, rows: pd.DataFrame, n: int = 500) -> dict:
    """Single-transaction time, as the API sees it (one thread: threads cost more than they save on one row)."""
    s.booster.set_param({"nthread": 1})
    t = []
    for i in range(n):
        x = rows.iloc[[i]]
        t0 = time.perf_counter()
        s.score(x)
        s.explain(x)
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2)}


def evaluate(s: model.Scorer, va: pd.DataFrame, te: pd.DataFrame) -> dict:
    sv, st = s.score(va), s.score(te)
    s.threshold = model.threshold_for_precision(va.Class.to_numpy(), sv)
    return {"valid_ap": round(average_precision_score(va.Class, sv), 4),
            "test_ap": round(average_precision_score(te.Class, st), 4),
            "valid": at_threshold(va, sv, s.threshold), "test": at_threshold(te, st, s.threshold),
            "threshold": round(s.threshold, 6), "trees": s.booster.best_iteration + 1, **latency_ms(s, te)}


def gate(cand: dict, champ: dict | None) -> tuple[bool, list[str]]:
    why = []
    if cand["valid"]["precision"] < GATE["min_valid_precision"]:
        why.append(f"valid precision {cand['valid']['precision']} < {GATE['min_valid_precision']}")
    if cand["p95_ms"] > GATE["max_p95_ms"]:
        why.append(f"p95 latency {cand['p95_ms']} ms > {GATE['max_p95_ms']}")
    if champ:
        if cand["valid_ap"] < champ["valid_ap"] - GATE["max_ap_drop"]:
            why.append(f"valid AP {cand['valid_ap']} vs champion {champ['valid_ap']}")
        if cand["valid"]["fraud_amount_caught_share"] < champ["valid"]["fraud_amount_caught_share"]:
            why.append(f"valid fraud amount caught {cand['valid']['fraud_amount_caught_share']} "
                       f"< champion {champ['valid']['fraud_amount_caught_share']}")
    return not why, why


def main() -> dict:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{HERE / 'mlflow.db'}"))
    mlflow.set_experiment("card-fraud")
    client = MlflowClient()

    t0 = time.time()
    df, ing = ingest()
    checks = validate(df)
    tr, va, te = (df[df.split == s] for s in ["train", "valid", "test"])
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df[data.FEATURES + ["Class"]]).values.tobytes()).hexdigest()[:16]

    results, scorers = {}, {}
    for name in ["xgb_baseline", "xgb_amount_weighted"]:
        with mlflow.start_run(run_name=name) as run:
            b = train_candidate(name, tr, va)
            s = model.Scorer(b, 0.0)
            m = evaluate(s, va, te)
            mlflow.log_params({**model.PARAMS, "candidate": name, "data_sha256": ing["source_sha256"][:16],
                               "features": len(data.FEATURES)})
            mlflow.log_metrics({"valid_ap": m["valid_ap"], "test_ap": m["test_ap"], "threshold": m["threshold"],
                                "valid_precision": m["valid"]["precision"], "valid_recall": m["valid"]["recall"],
                                "valid_amount_caught": m["valid"]["fraud_amount_caught_share"],
                                "test_precision": m["test"]["precision"], "test_recall": m["test"]["recall"],
                                "test_amount_caught": m["test"]["fraud_amount_caught_share"],
                                "p95_ms": m["p95_ms"], "trees": m["trees"]})
            info = mlflow.xgboost.log_model(b, name="model")
            results[name] = m | {"run_id": run.info.run_id, "model_uri": info.model_uri}
            scorers[name] = s

    # the current champion: the model registered with the alias, if any (first run: none)
    champ_name, champ = None, None
    try:
        v = client.get_model_version_by_alias(REGISTERED, "champion")
        champ_name = client.get_run(v.run_id).data.params.get("candidate")
        champ = results.get(champ_name)
    except Exception:  # noqa: BLE001 - no registered model yet
        pass
    baseline = results["xgb_baseline"]
    reference_for_gate = champ or baseline
    decisions = {}
    for name, m in results.items():
        ok, why = gate(m, None if name == "xgb_baseline" and champ is None else reference_for_gate)
        decisions[name] = {"passed": ok, "reasons": why}
    # promote the challenger if it passes against the champion, else keep / install the baseline
    winner = "xgb_amount_weighted" if decisions["xgb_amount_weighted"]["passed"] else (champ_name or "xgb_baseline")
    if champ is None and not decisions[winner]["passed"]:
        raise SystemExit(f"no candidate passed the gate: {decisions}")
    mv = mlflow.register_model(results[winner]["model_uri"], REGISTERED)
    client.set_registered_model_alias(REGISTERED, "champion", mv.version)

    s = scorers[winner]
    s.threshold = results[winner]["threshold"]
    s.save(MODEL_DIR)
    ref = reference(tr, data.FEATURES, s.score(tr))
    (MODEL_DIR / "reference.json").write_text(json.dumps(ref))
    card = {"name": REGISTERED, "version": int(mv.version), "candidate": winner, "run_id": results[winner]["run_id"],
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data_hash": data_hash,
            "source_sha256": ing["source_sha256"], "threshold": s.threshold, "features": data.FEATURES,
            "metrics": {k: results[winner][k] for k in ("valid_ap", "test_ap", "valid", "test", "p50_ms", "p95_ms")}}
    (MODEL_DIR / "card.json").write_text(json.dumps(card, indent=2, default=float))

    report = {"ingest": ing, "validation": checks, "candidates": results, "gate": GATE, "decisions": decisions,
              "champion_before": champ_name, "champion_after": winner, "registered_version": int(mv.version),
              "pipeline_seconds": round(time.time() - t0, 1)}
    OUT.mkdir(exist_ok=True)
    (OUT / "pipeline_run.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in ("decisions", "champion_before", "champion_after", "registered_version",
                                              "pipeline_seconds")}, indent=2))
    return report


if __name__ == "__main__":
    main()
