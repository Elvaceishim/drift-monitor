from transformers import pipeline

# Load the model once at startup — not on every request
_model = None

def load_model():
    global _model
    if _model is None:
        _model = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _model

def predict(text: str) -> dict:
    model = load_model()
    result = model(text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4)
    }