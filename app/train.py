import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import json

# Set up MLflow tracking URI to point to the local server
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Wine_Quality_Model")

def main():
    print("Loading Wine Quality dataset...")
    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=";")
    
    # Binarize the target variable
    data["quality"] = np.where(data["quality"] > 5, 1, 0)
    
    X = data.drop(["quality"], axis=1)
    y = data["quality"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate baseline statistics for Drift Detection (Mean and Std for each feature)
    baseline_stats = {
        "mean": X_train.mean().to_dict(),
        "std": X_train.std().to_dict()
    }
    
    with open("baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f)
        
    print("Training Random Forest model...")
    with mlflow.start_run():
        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {acc}")
        mlflow.log_metric("accuracy", acc)
        
        # Save model locally for easy access by FastAPI without complex MLflow auth in docker if needed
        # Or just use joblib locally
        joblib.dump(clf, "model.pkl")
        mlflow.sklearn.log_model(clf, "model")
        
        # Log baseline stats as artifact
        mlflow.log_artifact("baseline_stats.json")
        
        print("Training completed. Model saved to model.pkl and logged to MLflow.")

if __name__ == "__main__":
    main()
