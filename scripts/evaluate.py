import json
import sys
from transformers import pipeline

# Threshold — model must beat this accuracy to pass the gate ──
ACCURACY_THRESHOLD = 0.85

# Evaluation dataset — labelled samples the model must classify correctly ──
EVAL_DATA = [
    {"text": "This product is absolutely amazing!", "expected": "POSITIVE"},
    {"text": "Best purchase I have ever made.", "expected": "POSITIVE"},
    {"text": "Highly recommend to everyone.", "expected": "POSITIVE"},
    {"text": "Exceeded all my expectations.", "expected": "POSITIVE"},
    {"text": "Fantastic quality, very happy.", "expected": "POSITIVE"},
    {"text": "Terrible product, complete waste of money.", "expected": "NEGATIVE"},
    {"text": "Worst experience I have ever had.", "expected": "NEGATIVE"},
    {"text": "Deeply disappointed, do not buy.", "expected": "NEGATIVE"},
    {"text": "Broke after one day, awful quality.", "expected": "NEGATIVE"},
    {"text": "Total scam, avoid at all costs.", "expected": "NEGATIVE"},
    {"text": "Good value for the price.", "expected": "POSITIVE"},
    {"text": "Not worth the money at all.", "expected": "NEGATIVE"},
    {"text": "Really impressed with the results.", "expected": "POSITIVE"},
    {"text": "Complete disaster from start to finish.", "expected": "NEGATIVE"},
    {"text": "Works exactly as described.", "expected": "POSITIVE"},
    {"text": "Very poor customer service.", "expected": "NEGATIVE"},
    {"text": "Outstanding performance, love it.", "expected": "POSITIVE"},
    {"text": "Arrived broken and unusable.", "expected": "NEGATIVE"},
    {"text": "Solid product, does the job well.", "expected": "POSITIVE"},
    {"text": "Regret this purchase entirely.", "expected": "NEGATIVE"},
]

def evaluate():
    print("Loading model...")
    model = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    correct = 0
    total = len(EVAL_DATA)
    results = []

    print(f"Running evaluation on {total} samples...\n")

    for item in EVAL_DATA:
        prediction = model(item["text"])[0]
        predicted_label = prediction["label"]
        passed = predicted_label == item["expected"]
        if passed:
            correct += 1
        results.append({
            "text": item["text"],
            "expected": item["expected"],
            "predicted": predicted_label,
            "score": round(prediction["score"], 4),
            "passed": passed
        })
        status = "✓" if passed else "✗"
        print(f"{status} [{predicted_label}] {item['text'][:50]}")

    accuracy = correct / total
    print(f"\n─── Evaluation Results ───")
    print(f"Correct:  {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Threshold: {ACCURACY_THRESHOLD:.2%}")

    # Save results to file for the workflow to read
    with open("eval_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "threshold": ACCURACY_THRESHOLD,
            "passed": accuracy >= ACCURACY_THRESHOLD,
            "results": results
        }, f, indent=2)

    if accuracy >= ACCURACY_THRESHOLD:
        print(f"\n✅ PASSED — Model meets the quality threshold. Safe to deploy.")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED — Model below threshold. Deployment blocked.")
        sys.exit(1)  # Non-zero exit code fails the GitHub Actions step

if __name__ == "__main__":
    evaluate()