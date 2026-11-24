"""Drift report: replay the 48 hours of real transactions in time order, one-hour windows, and compute
PSI of every feature and of the model score against the training reference saved with the model.

    python drift_report.py     # -> results/drift.json, results/drift.png

This is what the API's /monitoring/drift does on live traffic, run offline on the whole timeline so
the pattern is visible: which features move, when, and whether the score moves with them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import data
from model import Scorer
from monitoring import psi_table, reference

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main() -> dict:
    df = data.load()
    s = Scorer.load(HERE / "model")
    df["score"] = s.score(df)
    ref = json.loads((HERE / "model" / "reference.json").read_text())
    tr = df[df.split == "train"]
    feats = [f for f in data.FEATURES if f != "hour"]
    # hour-matched reference: training rows from the same hour of the day (night traffic vs night traffic)
    by_hour = {h: reference(g, feats, g.score.to_numpy()) for h, g in tr.groupby("hour")}
    df["window"] = (df.Time // 3600).astype(int)
    rows = []
    for w, g in df.groupby("window"):
        naive = psi_table({k: v for k, v in ref.items() if k != "hour"}, g)
        matched = psi_table(by_hour[int(g.hour.iloc[0])], g)
        rows.append({"window_hour": int(w), "hour_of_day": int(g.hour.iloc[0]), "split": g.split.mode()[0],
                     "n": len(g), "fraud": int(g.Class.sum()),
                     "alert_rate_per_10k": round(float((g.score >= s.threshold).mean() * 1e4), 1),
                     "naive_score_psi": naive["score"], "naive_features_over_0.25": sum(naive[f] > 0.25 for f in feats),
                     "naive_worst_feature_psi": max(naive[f] for f in feats),
                     "matched_score_psi": matched["score"], "matched_features_over_0.25": sum(matched[f] > 0.25 for f in feats),
                     "matched_worst_feature_psi": max(matched[f] for f in feats)})
    t = pd.DataFrame(rows)
    later = t[t.split != "train"]
    summary = {
        "windows": len(t), "later_windows (valid + test)": len(later),
        "naive_reference": {
            "windows_with_3plus_features_over_0.25": int((later["naive_features_over_0.25"] >= 3).sum()),
            "windows_with_score_psi_over_0.25": int((later.naive_score_psi > 0.25).sum()),
            "median_worst_feature_psi": round(float(later.naive_worst_feature_psi.median()), 3)},
        "hour_matched_reference": {
            "windows_with_3plus_features_over_0.25": int((later["matched_features_over_0.25"] >= 3).sum()),
            "windows_with_score_psi_over_0.25": int((later.matched_score_psi > 0.25).sum()),
            "median_worst_feature_psi": round(float(later.matched_worst_feature_psi.median()), 3)},
        "by_window": t.round(4).to_dict("records"),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "drift.json").write_text(json.dumps(summary, indent=2, default=float))
    chart(t)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_window"}, indent=2, default=float))
    print(t[["window_hour", "hour_of_day", "split", "n", "naive_features_over_0.25", "matched_features_over_0.25",
             "naive_score_psi", "matched_score_psi"]].to_string())
    return summary


def chart(t: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t.window_hour, t["naive_features_over_0.25"], lw=2, color="#f97316",
               label="vs all training data")
    ax[0].plot(t.window_hour, t["matched_features_over_0.25"], lw=2, color="#2563eb",
               label="vs training data from the same hour of day")
    ax[0].axhline(3, color="#999", ls=":", label="alert rule: 3 or more features")
    ax[0].set_ylabel("features with PSI > 0.25")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].bar(t.window_hour, t.n, color="#94a3b8")
    ax[1].set_ylabel("transactions per hour")
    ax[1].set_xlabel("hours since start of data (train 0-33, valid 33-40, test 40-48)")
    for a in ax:
        for x in (33.4, 40.3):
            a.axvline(x, color="#555", lw=0.8)
        a.grid(alpha=0.3)
    ax[0].set_title("Hourly drift checks on 48 hours of real card traffic")
    fig.tight_layout()
    fig.savefig(OUT / "drift.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
