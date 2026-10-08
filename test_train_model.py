import os
import json
import unittest
import pandas as pd
from train_model import train_model, DATA_FILE


class TestHotelBookingML(unittest.TestCase):

    def test_dataset_exists_and_valid(self):
        """Test 1: Verify the dataset exists and contains expected columns."""
        self.assertTrue(os.path.exists(DATA_FILE), f"Dataset file '{DATA_FILE}' not found!")
        df = pd.read_csv(DATA_FILE, nrows=50)
        self.assertGreater(len(df), 0, "Dataset is empty!")
        self.assertIn("is_canceled", df.columns, "Target column 'is_canceled' missing from dataset!")

    def test_model_training_and_accuracy(self):
        """Test 2: Verify model trains and achieves acceptable benchmark accuracy (> 75%)."""
        accuracy = train_model()
        self.assertGreaterEqual(
            accuracy,
            0.75,
            f"Model accuracy {accuracy:.4f} is below acceptable threshold of 0.75"
        )

    def test_metrics_and_model_artifacts_saved(self):
        """Test 3: Verify the trained model artifact and metrics.json are generated."""
        self.assertTrue(os.path.exists("hotel_booking_model.pkl"), "hotel_booking_model.pkl not saved!")
        self.assertTrue(os.path.exists("metrics.json"), "metrics.json not saved!")

        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
