import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "online"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_positive():
    response = client.post("/predict", json={"text": "This is absolutely wonderful!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["score"] > 0.5
    assert data["text"] == "This is absolutely wonderful!"

def test_predict_negative():
    response = client.post("/predict", json={"text": "This is absolutely terrible!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert data["score"] > 0.5

def test_predict_empty_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400

def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422