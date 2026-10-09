import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "lead_time",
    "previous_cancellations",
    "booking_changes",
    "total_of_special_requests",
    "required_car_parking_spaces"
]
TARGET = "is_canceled"

def find_dataset():
    candidates = ["Hotel Bookings.csv", "hotel_bookings.csv", "Hotel_Bookings.csv"]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Dataset not found! Checked: {candidates}. Current dir contains: {os.listdir('.')}"
    )

def train_model():
    dataset_path = find_dataset()
    print(f"Loading dataset from: {dataset_path}...")
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully. Total records:", len(data))

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy:", round(accuracy, 4))

    # Save model and metrics
    joblib.dump(model, "hotel_booking_model.pkl")
    print("Saved hotel_booking_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics.json")

    return accuracy

if __name__ == "__main__":
    train_model()
