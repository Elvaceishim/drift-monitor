# Drift Monitor — ML Model Serving API with Input Drift Detection

A production-grade REST API that serves sentiment predictions from a fine-tuned DistilBERT model and monitors incoming requests for statistical input drift using the Kolmogorov-Smirnov test.

Built as part of a deliberate MLOps skill-building track targeting production AI engineering roles.

---

## What it does

- Serves real-time sentiment predictions via a REST API
- Logs every incoming request with extracted features (text length, word count, confidence score)
- Runs statistical drift detection comparing live traffic distributions against a baseline
- Flags drift per feature with p-values and KS statistics
- Fully containerized with Docker for portable deployment

---

## Why it matters

Most ML portfolios stop at model training. This project covers the part that actually breaks in production — **what happens after the model is deployed**. Input drift is one of the leading causes of silent model degradation in production systems. This API detects it automatically, giving teams an early signal before prediction quality degrades.

---

## Stack

| Layer            | Technology                                                   |
| ---------------- | ------------------------------------------------------------ |
| API Framework    | FastAPI                                                      |
| Model            | DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) |
| Drift Detection  | Kolmogorov-Smirnov Test (`scipy.stats`)                      |
| Containerization | Docker                                                       |
| Logging          | JSONL flat-file request logger                               |
| Runtime          | Python 3.12                                                  |

---

## Endpoints

| Method | Endpoint   | Description                                    |
| ------ | ---------- | ---------------------------------------------- |
| GET    | `/`        | API info and available endpoints               |
| GET    | `/health`  | Health check                                   |
| POST   | `/predict` | Run sentiment prediction on input text         |
| GET    | `/drift`   | Run drift report against baseline distribution |

---

## Running locally

**Requirements:** Python 3.12+, pip

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

**With Docker:**

```bash
docker build -t drift-monitor .
docker run -p 8000:8000 drift-monitor
```

Visit `http://127.0.0.1:8000/docs`

---

## How drift detection works

On startup, the API loads a baseline distribution from `data/baseline.json` — a snapshot of expected input characteristics (text length, word count, confidence score) derived from representative training traffic.

Every request is logged to `logs/requests.jsonl`. When `/drift` is called, the API runs a two-sample **Kolmogorov-Smirnov test** comparing the last 50 live requests against the baseline for each feature. A p-value below `0.05` on any feature indicates the incoming traffic distribution has shifted significantly — a signal worth investigating before silent model degradation occurs.

At least 10 logged requests are required before drift detection runs.

---

## Example: Predict

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product exceeded all my expectations."}'
```

**Response:**

```json
{
  "label": "POSITIVE",
  "score": 0.9998,
  "text": "This product exceeded all my expectations."
}
```

---

## Example: Drift Report

**Request:**

```bash
curl http://127.0.0.1:8000/drift
```

**Response:**

```json
{
  "status": "drift_detected",
  "request_count": 50,
  "features": {
    "text_length": {
      "statistic": 0.46,
      "p_value": 0.0454,
      "drift_detected": true
    },
    "word_count": {
      "statistic": 0.58,
      "p_value": 0.0044,
      "drift_detected": true
    },
    "confidence_score": {
      "statistic": 1.0,
      "p_value": 0.0,
      "drift_detected": true
    }
  }
}
```

---

## Project structure

```
drift-monitor/
├── app/
│   ├── main.py        # FastAPI app and route definitions
│   ├── model.py       # DistilBERT model loading and inference
│   ├── drift.py       # KS-test drift detection logic
│   └── logger.py      # JSONL request logger
├── data/
│   └── baseline.json  # Baseline feature distributions
├── logs/
│   └── requests.jsonl # Auto-generated at runtime
├── Dockerfile
└── requirements.txt
```

---

## Author

**Elvis Anselm** — AI Engineer & Content Strategist, Lagos Nigeria

[Portfolio](https://gleeful-starburst-ddb1d6.netlify.app) · [LinkedIn](https://www.linkedin.com/in/elvisanselm/) · [GitHub](https://github.com/Elvaceishim) · [Repository](https://github.com/Elvaceishim/drift-monitor)
