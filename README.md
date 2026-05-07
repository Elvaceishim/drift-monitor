# ML Production Stack

A two-part MLOps project covering model serving with observability and automated CI/CD with quality gating. Built as part of a deliberate MLOps skill-building track targeting production AI engineering roles.

**Part 1** — Production REST API serving DistilBERT sentiment predictions with statistical input drift detection.  
**Part 2** — GitHub Actions CI/CD pipeline that automatically tests and gates the model on every push.

---

## Part 1 — Model Serving API with Drift Monitoring

### What it does

- Serves real-time sentiment predictions via REST API
- Logs every incoming request with extracted features (text length, word count, confidence score)
- Runs statistical drift detection comparing live traffic against a baseline distribution
- Flags drift per feature with p-values and KS statistics
- Fully containerized with Docker

### Why it matters

Most ML portfolios stop at model training. This covers what actually breaks in production — input drift is one of the leading causes of silent model degradation. This API detects it automatically, giving teams an early signal before prediction quality degrades.

### Stack

| Layer            | Technology                                                     |
| ---------------- | -------------------------------------------------------------- |
| API Framework    | FastAPI                                                        |
| Model            | DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) |
| Drift Detection  | Kolmogorov-Smirnov Test (`scipy.stats`)                        |
| Containerization | Docker                                                         |
| Logging          | JSONL flat-file request logger                                 |
| Runtime          | Python 3.12                                                    |

### Endpoints

| Method | Endpoint   | Description                                    |
| ------ | ---------- | ---------------------------------------------- |
| GET    | `/`        | API info and available endpoints               |
| GET    | `/health`  | Health check                                   |
| POST   | `/predict` | Run sentiment prediction on input text         |
| GET    | `/drift`   | Run drift report against baseline distribution |

### Running locally

**Without Docker:**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

**With Docker:**

```bash
docker build -t ml-production-stack .
docker run -p 8000:8000 ml-production-stack
```

### How drift detection works

On startup, the API loads a baseline distribution from `data/baseline.json` — a snapshot of expected input characteristics (text length, word count, confidence score) derived from representative training traffic.

Every request is logged to `logs/requests.jsonl`. When `/drift` is called, the API runs a two-sample **Kolmogorov-Smirnov test** comparing the last 50 live requests against the baseline for each feature. A p-value below `0.05` on any feature indicates the incoming traffic distribution has shifted significantly — a signal worth investigating before silent model degradation occurs.

At least 10 logged requests are required before drift detection runs.

### Example: Predict

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

### Example: Drift Report

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

## Part 2 — CI/CD Pipeline with Model Evaluation Gate

### What it does

- Triggers automatically on every push to `main`
- Runs a 6-test API test suite via pytest
- Evaluates the model against 20 labelled samples
- Blocks deployment if accuracy falls below 85%
- Uploads evaluation results as a downloadable artifact
- Tested locally using [Act](https://nektosact.com)

### Why it matters

Most engineers push code and manually check if things work. This pipeline means the code checks itself — the model only proceeds to deployment if it clears both the test suite and the evaluation gate. That's the difference between a developer and an engineer who thinks about production systems.

### Pipeline steps

```
Push to main
↓
GitHub spins up Ubuntu VM
↓
Step 1: Check out code
↓
Step 2: Set up Python 3.12
↓
Step 3: Cache pip dependencies
↓
Step 4: Install dependencies
↓
Step 5: Run API tests (pytest)  ← pipeline stops here if any test fails
↓
Step 6: Run model evaluation    ← pipeline stops here if accuracy < 85%
↓
Step 7: Upload eval results as artifact
```

### Evaluation results (local Act run)

```
Correct:   20/20
Accuracy:  100.00%
Threshold: 85.00%
✅ PASSED — Model meets the quality threshold. Safe to deploy.
```

---

## Project structure

```
ml-production-stack/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml    # CI/CD workflow
├── app/
│   ├── main.py                # FastAPI app and routes
│   ├── model.py               # Model loading and inference
│   ├── drift.py               # KS-test drift detection
│   └── logger.py              # JSONL request logger
├── data/
│   └── baseline.json          # Baseline feature distributions
├── logs/
│   └── requests.jsonl         # Auto-generated at runtime
├── scripts/
│   └── evaluate.py            # Model evaluation + threshold gate
├── tests/
│   └── test_api.py            # API endpoint tests
├── conftest.py                # Pytest path configuration
├── Dockerfile
└── requirements.txt
```

---

## Author

**Elvis Anselm** — AI Engineer & Content Strategist, Lagos Nigeria

[Portfolio](https://elvace.netlify.app) · [LinkedIn](https://www.linkedin.com/in/elvisanselm/) · [GitHub](https://github.com/Elvaceishim) · [Repository](https://github.com/Elvaceishim/drift-monitor)
