"""Replay the test period through the running API, one transaction at a time in time order,
and measure end-to-end latency (client side, over HTTP).

    uvicorn service:app --port 8000 &
    python replay.py --n 5000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import data
import httpx
import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def payload(row) -> dict:
    return {"transaction_id": str(row.Index), "time_s": float(row.Time), "amount": float(row.Amount),
            "pca": [float(getattr(row, f"V{i}")) for i in range(1, 29)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=5000, help="test transactions to replay (0 = all)")
    a = ap.parse_args(argv)
    test = data.load().query("split == 'test'")
    if a.n:
        # all fraud cases plus the first legitimate ones, so the alert check covers every fraud
        test = test[(test.Class == 1) | (test.index.isin(test[test.Class == 0].index[: a.n]))]
    lat, server, alerts = [], [], []
    with httpx.Client(base_url=a.url, timeout=10) as c:
        c.get("/health").raise_for_status()
        for row in test.itertuples():
            t0 = time.perf_counter()
            r = c.post("/score", json=payload(row))
            lat.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            d = r.json()
            server.append(d["server_ms"])
            alerts.append((d["alert"], row.Class))
    lat, server = np.array(lat), np.array(server)
    al = np.array(alerts)
    res = {"transactions": len(test), "fraud": int(test.Class.sum()),
           "client_ms": {f"p{q}": round(float(np.percentile(lat, q)), 2) for q in (50, 95, 99)},
           "server_ms": {f"p{q}": round(float(np.percentile(server, q)), 2) for q in (50, 95, 99)},
           "throughput_per_s_single_client": round(len(lat) / (lat.sum() / 1000), 1),
           "alerts": int(al[:, 0].sum()), "fraud_alerted": int((al[:, 0] & al[:, 1].astype(bool)).sum())}
    OUT.mkdir(exist_ok=True)
    (OUT / "replay.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
