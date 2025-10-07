# tests/test_app.py
import json
from app import app

def test_health():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json().get("status") == "ok"

def test_predict_minimal():
    client = app.test_client()
    payload = {
      "Age": 45,
      "Polyuria": 1,
      "Polydipsia": 1,
      "sudden_weight_loss": 0,
      "weakness": 0,
      "Polyphagia": 0,
      "Genital_thrush": 0,
      "visual_blurring": 0,
      "Itching": 0,
      "Irritability": 0,
      "delayed_healing": 0,
      "partial_paresis": 0,
      "muscle_stiffness": 0,
      "Alopecia": 0,
      "Obesity": 0
    }
    r = client.post("/predict", data=json.dumps(payload), content_type='application/json')
    assert r.status_code == 200
    js = r.get_json()
    assert "probability" in js
    assert "explanation_text" in js