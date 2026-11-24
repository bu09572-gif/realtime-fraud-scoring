"""Load test: real test-period transactions sent to the API by N concurrent clients.

    python loadtest.py --url http://127.0.0.1:8000 --concurrency 1 4 16 --seconds 20 --cost-per-hour 0.0416

For each concurrency level (and each --batch size on /score/batch): throughput in transactions per
second, request latency percentiles, errors, and cost per million predictions
if the machine ran at that throughput all hour: cost_per_hour / (throughput * 3600) * 1e6.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path

import httpx
import numpy as np

import data
from replay import payload

OUT = Path(__file__).resolve().parent / "results"


async def level(url: str, rows: list[dict], conc: int, seconds: float, batch: int = 1) -> dict:
    lat, errors, i = [], 0, 0
    stop = time.perf_counter() + seconds

    async def worker(c: httpx.AsyncClient):
        nonlocal errors, i
        while time.perf_counter() < stop:
            if batch > 1:
                path, body = "/score/batch", {"transactions": [rows[(i + j) % len(rows)] for j in range(batch)]}
            else:
                path, body = "/score", rows[i % len(rows)]
            i += batch
            t0 = time.perf_counter()
            try:
                r = await c.post(path, json=body)
                ok = r.status_code == 200
            except httpx.HTTPError:
                ok = False
            lat.append((time.perf_counter() - t0) * 1000)
            errors += not ok

    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    async with httpx.AsyncClient(base_url=url, timeout=10, limits=limits) as c:
        await c.get("/health")
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(c) for _ in range(conc)))
        wall = time.perf_counter() - t0
    lat = np.array(lat)
    return {"concurrency": conc, "batch": batch, "requests": len(lat), "errors": errors,
            "throughput_per_s": round(len(lat) * batch / wall, 1),
            **{f"p{q}_ms": round(float(np.percentile(lat, q)), 2) for q in (50, 95, 99)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--batch", type=int, nargs="+", default=[1], help="transactions per request (1 = /score)")
    ap.add_argument("--cost-per-hour", type=float, default=float(os.environ.get("COST_PER_HOUR_USD", "0.0416")))
    ap.add_argument("--label", default=os.environ.get("LOADTEST_LABEL", platform.node()))
    a = ap.parse_args(argv)
    test = data.load().query("split == 'test'")
    rows = [payload(r) for r in test.head(20000).itertuples()]
    res = []
    for conc, batch in [(c, 1) for c in a.concurrency] + [(1, b) for b in a.batch if b > 1]:
        r = asyncio.run(level(a.url, rows, conc, a.seconds, batch))
        r["usd_per_million_predictions"] = round(a.cost_per_hour / max(r["throughput_per_s"], 1e-9) / 3600 * 1e6, 3)
        res.append(r)
        print(json.dumps(r))
    out = {"where": a.label, "cpus": os.cpu_count(), "cost_per_hour_usd": a.cost_per_hour, "levels": res}
    OUT.mkdir(exist_ok=True)
    (OUT / "loadtest.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
