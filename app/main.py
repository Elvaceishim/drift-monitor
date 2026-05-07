from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict
from app.logger import log_request
from app.drift import check_drift

app = FastAPI(
    title="Sentiment API with Drift Monitoring",
    description="Serves DistilBERT sentiment predictions and monitors input drift.",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float
    text: str

@app.get("/")
def root():
    return {
        "status": "online",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "endpoints": ["/predict", "/drift", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    prediction = predict(request.text)
    log_request(request.text, prediction)
    
    return {
        "label": prediction["label"],
        "score": prediction["score"],
        "text": request.text
    }

@app.get("/drift")
def drift_report():
    return check_drift()