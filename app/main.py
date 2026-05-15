import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import time

app = FastAPI(title="ML Monitoring System API")

# Prometheus Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["endpoint", "method"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request Latency", ["endpoint"])
PREDICTION_COUNTER = Counter("model_predictions_total", "Total Model Predictions", ["predicted_class"])
DATA_DRIFT_SCORE = Gauge("data_drift_score", "Average Z-Score of incoming features compared to baseline")
FEATURE_GAUGE = Gauge("feature_value", "Value of incoming features", ["feature"])

# Add prometheus asgi middleware to route /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Load Model and Baseline Stats
MODEL_PATH = "model.pkl"
STATS_PATH = "baseline_stats.json"

clf = None
baseline_stats = None

@app.on_event("startup")
def load_assets():
    global clf, baseline_stats
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        clf = joblib.load(MODEL_PATH)
        print("Model loaded.")
    else:
        print(f"Warning: {MODEL_PATH} not found or is empty. Please train the model first.")
        
    if os.path.exists(STATS_PATH) and os.path.getsize(STATS_PATH) > 0:
        with open(STATS_PATH, "r") as f:
            baseline_stats = json.load(f)
        print("Baseline statistics loaded.")
    else:
        print(f"Warning: {STATS_PATH} not found or is empty.")

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
async def predict(features: WineFeatures):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    
    if clf is None or baseline_stats is None:
        raise HTTPException(status_code=500, detail="Model or baseline stats not loaded")
        
    feature_dict = features.dict()
    feature_names = list(feature_dict.keys())
    
    # Format input for prediction
    # Scikit-learn expects 2D array and we must ensure column order matches training.
    # We used default df ordering in train.py
    # Since feature_dict keys match dataframe columns exactly, we can just use values
    # But to be safe and match names with train data:
    # Train columns: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
    # free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
    # Pydantic uses underscores. Let's map underscores back to spaces.
    
    mapped_features = {k.replace("_", " "): v for k, v in feature_dict.items()}
    df = pd.DataFrame([mapped_features])
    
    # 1. Calculate Data Drift Score
    # Simple Z-score logic based on training stats
    drift_sum = 0
    for col in df.columns:
        val = df[col].iloc[0]
        mean = baseline_stats["mean"].get(col, 0)
        std = baseline_stats["std"].get(col, 1)
        z_score = abs(val - mean) / (std + 1e-9)
        drift_sum += z_score
        
        # Expose feature values to Prometheus
        FEATURE_GAUGE.labels(feature=col).set(val)
        
    avg_drift = drift_sum / len(df.columns)
    DATA_DRIFT_SCORE.set(avg_drift)
    
    # 2. Prediction
    pred = clf.predict(df)[0]
    pred_class = "High Quality" if pred == 1 else "Low Quality"
    PREDICTION_COUNTER.labels(predicted_class=pred_class).inc()
    
    # 3. Latency
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
    
    return {
        "prediction": pred_class,
        "drift_score": avg_drift
    }

@app.get("/health")
def health_check():
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return {"status": "healthy"}
