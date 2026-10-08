import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "Hotel Bookings.csv"


def train_model():
    print("Loading Hotel Bookings dataset...")
    data = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))

    features = [
        "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
        "stays_in_weekend_nights", "stays_in_week_nights", "adults",
        "children", "babies", "meal", "market_segment", "distribution_channel",
        "is_repeated_guest", "previous_cancellations", "booking_changes",
        "deposit_type", "customer_type", "adr", "required_car_parking_spaces",
        "total_of_special_requests"
    ]
    target = "is_canceled"

    X = data[features]
    y = data[target]

    categorical_features = [
        "hotel", "arrival_date_month", "meal", "market_segment",
        "distribution_channel", "deposit_type", "customer_type"
    ]

    numerical_features = [
        "lead_time", "arrival_date_year", "stays_in_weekend_nights",
        "stays_in_week_nights", "adults", "children", "babies",
        "is_repeated_guest", "previous_cancellations", "booking_changes",
        "adr", "required_car_parking_spaces", "total_of_special_requests"
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "hotel_booking_model.pkl")
    print("\nModel saved as hotel_booking_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")
    return accuracy


if __name__ == "__main__":
    train_model()
