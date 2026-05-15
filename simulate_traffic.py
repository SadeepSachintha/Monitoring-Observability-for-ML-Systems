import requests
import json
import time
import random

API_URL = "http://host.docker.internal:8000/predict"

# Baseline normal features based on Wine Quality dataset
normal_features = {
    "fixed_acidity": 8.3,
    "volatile_acidity": 0.5,
    "citric_acid": 0.3,
    "residual_sugar": 2.5,
    "chlorides": 0.08,
    "free_sulfur_dioxide": 15.0,
    "total_sulfur_dioxide": 46.0,
    "density": 0.996,
    "pH": 3.3,
    "sulphates": 0.65,
    "alcohol": 10.4
}

def send_request(features):
    try:
        response = requests.post(API_URL, json=features)
        print(f"Status: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

print("Simulating NORMAL traffic...")
for _ in range(20):
    # Add small random noise to normal features
    payload = {k: v + random.uniform(-0.1 * v, 0.1 * v) for k, v in normal_features.items()}
    send_request(payload)
    time.sleep(0.5)

print("\nSimulating DATA DRIFT...")
for _ in range(15):
    # Add large random noise to simulate drift
    payload = {k: v + random.uniform(2.0 * v, 5.0 * v) for k, v in normal_features.items()}
    send_request(payload)
    time.sleep(0.5)

print("\nSimulation complete.")
