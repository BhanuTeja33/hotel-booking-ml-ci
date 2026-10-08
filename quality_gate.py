import json
import sys

# Define minimum acceptable operational accuracy threshold
MINIMUM_ACCURACY = 0.90

print("Reading model evaluation metrics...")

try:
    with open("metrics.json", "r") as file:
        metrics = json.load(file)
except FileNotFoundError:
    print("ERROR: metrics.json not found! Ensure model training completed successfully.")
    sys.exit(1)

accuracy = metrics.get("accuracy", 0.0)

print(f"Model Accuracy   : {accuracy:.4f}")
print(f"Required Accuracy: {MINIMUM_ACCURACY:.4f}")

if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)  # Non-zero exit code stops GitHub Actions immediately

print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)  # Exit code 0 allows the pipeline to proceed
