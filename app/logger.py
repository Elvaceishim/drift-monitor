import json
import os
from datetime import datetime, timezone

LOG_PATH = "logs/requests.jsonl"

def log_request(text: str, prediction: dict):
    os.makedirs("logs", exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "label": prediction["label"],
        "score": prediction["score"],
        "text_length": len(text),
        "word_count": len(text.split())
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_logs() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]