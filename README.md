# Real-Time Card Fraud Scoring with SHAP Explanations

[![tests](https://github.com/bu09572-gif/realtime-fraud-scoring/actions/workflows/tests.yml/badge.svg)](https://github.com/bu09572-gif/realtime-fraud-scoring/actions/workflows/tests.yml)

A fraud model behind an HTTP API. Each card transaction gets a fraud probability, an alert decision and the three features that pushed its score the most. It is trained and tested on real card transactions, split by time the way it would run in production: train on the past, set the alert rule on the next period, then test on the latest period with nothing changed.

## Data

The ULB credit card fraud dataset (Dal Pozzolo et al., 2015): 284,807 transactions by European cardholders over two days in September 2013, 492 of them fraud (0.17%). To protect cardholders, the bank released 28 PCA components (V1 to V28) instead of the raw fields. Only the time and the amount are raw. `data.py` downloads the CSV, checks its SHA-256, and adds log(amount) and hour of day.

| Period | Hours | Transactions | Fraud |
|---|---|---|---|
| train | 0 to 33.4 | 170,885 | 360 |
| valid | 33.4 to 40.3 | 56,961 | 57 |
| test | 40.3 to 48.0 | 56,961 | 75 |

## How it works

- **Model:** XGBoost (depth 4, 285 trees, early stopping on validation average precision). Logistic regression is the baseline.
- **Alert rule:** on the validation period, pick the lowest score where at least 80% of alerts are fraud. That threshold (0.299) is then frozen for the test period.
- **Reasons:** exact TreeSHAP values from XGBoost (`pred_contribs`). The tests check that they add up to the model's score.
- **API** (`service.py`, FastAPI): `POST /score` takes the amount, time and the 28 components. It returns the probability, `alert`, and the top 3 reasons with their values and contributions. Bad input (a negative amount, a wrong number of components) gets a 422.

Response for the first fraud in the test period (transaction 229712):

```json
{"transaction_id": "229712", "fraud_probability": 0.994446, "alert": true, "threshold": 0.299,
 "reasons": [{"feature": "V10", "value": -4.0631, "contribution": 5.3526},
             {"feature": "V14", "value": -9.151, "contribution": 2.0524},
             {"feature": "V4", "value": 6.0094, "contribution": 1.093}]}
```

Contributions are in log-odds: positive values push toward fraud. The real response also has `server_ms`.

## Results (test period, last 7.7 hours)

From `results/metrics.json` and `results/replay.json`.

| | XGBoost | Logistic regression |
|---|---|---|
| Average precision (PR-AUC) | **0.768** (95% CI 0.665 to 0.858) | 0.688 (0.571 to 0.790) |
| ROC-AUC | 0.977 | 0.976 |
| Alerts | 69 (12 per 10,000 transactions) | 42 |
| Precision of alerts (target 80%) | **79.7%** | 81.0% |
| Fraud caught | **55 of 75 (73.3%)** | 34 of 75 (45.3%) |
| Share of fraud amount caught | **52.5%** | 5.4% |

![Precision-recall curve on the test period](results/pr_curve.png)

- **The alert rule held up.** It was set for 80% precision on the validation period and gave 79.7% on later data it had never seen. Recall was 77% on validation and 73% on test.
- **XGBoost catches 21 more frauds than logistic regression at the same precision.** ROC-AUC barely separates the two (0.977 vs 0.976). With 0.13% fraud, ROC-AUC is the wrong number to look at, and precision-recall is the one that matters.
- **The missed frauds carry more money.** The model catches 73% of fraud cases but only 52.5% of the fraud amount, so it is weaker on large-amount fraud. A next step would be to weight training by amount, or to add an amount-based rule.
- **The intervals are wide.** There are only 75 frauds in the test period, so the PR-AUC interval spans 0.67 to 0.86.

**Serving.** I replayed 5,075 test transactions (all 75 frauds plus the first 5,000 legitimate ones) one at a time over HTTP to the API running on a 4-core cloud VM:

| | p50 | p95 | p99 |
|---|---|---|---|
| End-to-end latency (client) | 9.0 ms | 12.7 ms | 15.5 ms |
| Scoring + SHAP inside the API | 6.5 ms | 9.7 ms | 11.9 ms |

That is about 105 transactions per second from a single client. The API alerted on the same 55 frauds as offline scoring.

## Limits

- **The reasons can't be read in business terms.** On test alerts, the top reason was V10 (52 times) or V14 (17 times). These are PCA components, so a reason like "V14 = -9.15" means nothing to a fraud analyst. On a real feed the same code would show features like merchant category or distance from home.
- **Only two days of data.** That is too little to see drift, so I can't say how often the threshold would need resetting.

## Running it

```bash
pip install xgboost scikit-learn pandas pyarrow fastapi uvicorn httpx matplotlib
python run.py                       # train, pick threshold, evaluate -> model/, results/
uvicorn service:app --port 8000 &
python replay.py --n 5000           # latency over HTTP
pytest -q test_fraud.py
```

---

Part of my [AI/ML and data engineering portfolio](https://github.com/bu09572-gif/ai-ml-portfolio), where every project uses real public data and free, open-source tools.
