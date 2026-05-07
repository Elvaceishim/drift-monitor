import json
import numpy as np
from scipy.stats import ks_2samp
from app.logger import load_logs

BASELINE_PATH = "data/baseline.json"
DRIFT_THRESHOLD = 0.05  # p-value threshold — below this means drift detected

def load_baseline() -> dict:
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)

def check_drift() -> dict:
    baseline = load_baseline()
    logs = load_logs()

    if len(logs) < 10:
        return {
            "status": "insufficient_data",
            "message": "Need at least 10 requests to run drift detection.",
            "request_count": len(logs)
        }

    # Extract features from recent requests
    recent_lengths = [entry["text_length"] for entry in logs[-50:]]
    recent_word_counts = [entry["word_count"] for entry in logs[-50:]]
    recent_scores = [entry["score"] for entry in logs[-50:]]

    # Run Kolmogorov-Smirnov test against baseline distributions
    ks_length = ks_2samp(baseline["text_lengths"], recent_lengths)
    ks_words = ks_2samp(baseline["word_counts"], recent_word_counts)
    ks_scores = ks_2samp(baseline["confidence_scores"], recent_scores)

    results = {
        "text_length": {
            "statistic": round(float(ks_length.statistic), 4),
            "p_value": round(float(ks_length.pvalue), 4),
            "drift_detected": bool(ks_length.pvalue < DRIFT_THRESHOLD)
        },
        "word_count": {
            "statistic": round(float(ks_words.statistic), 4),
            "p_value": round(float(ks_words.pvalue), 4),
            "drift_detected": bool(ks_words.pvalue < DRIFT_THRESHOLD)
        },
        "confidence_score": {
            "statistic": round(float(ks_scores.statistic), 4),
            "p_value": round(float(ks_scores.pvalue), 4),
            "drift_detected": bool(ks_scores.pvalue < DRIFT_THRESHOLD)
        }
    }

    any_drift = any(v["drift_detected"] for v in results.values())

    return {
        "status": "drift_detected" if any_drift else "stable",
        "request_count": len(logs),
        "features": results
    }