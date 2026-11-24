# Serving image: the API and the exported champion model only (no training data, no MLflow).
FROM python:3.12-slim AS base
LABEL org.opencontainers.image.source="https://github.com/bu09572-gif/realtime-fraud-scoring" \
      org.opencontainers.image.description="Card fraud scoring API (XGBoost + SHAP) with Prometheus metrics and drift monitoring"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY model.py monitoring.py service.py data.py ./
COPY model/ model/
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
