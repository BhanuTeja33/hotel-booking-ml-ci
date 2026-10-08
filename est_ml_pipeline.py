import json
import os
import unittest
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        """1. Verify that the Hotel Bookings dataset exists and is accessible."""
        self.assertTrue(os.path.exists("Hotel Bookings.csv"), "Hotel Bookings.csv not found!")

    def test_model_created(self):
        """2. Verify that the trained model artifact (.pkl) was generated."""
        self.assertTrue(
            os.path.exists("hotel_booking_model.pkl"),
            "hotel_booking_model.pkl artifact was not created!"
        )

    def test_metrics_created(self):
        """3. Verify that the evaluation metrics file was generated."""
        self.assertTrue(os.path.exists("metrics.json"), "metrics.json was not created!")

    def test_accuracy_is_valid(self):
        """4. Verify that the recorded evaluation accuracy is a valid probability score."""
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        """5. Verify that the loaded model artifact produces a valid binary prediction [0, 1]."""
        model = joblib.load("hotel_booking_model.pkl")
        sample = pd.DataFrame([{
            "hotel": "City Hotel",
            "lead_time": 60,
            "arrival_date_year": 2016,
            "arrival_date_month": "March",
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "booking_changes": 0,
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "adr": 95.0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 1
        }])
        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_cancellation_risk_booking(self):
        """6. Verify that a high-risk booking profile is classified as CANCELED (1)."""
        model = joblib.load("hotel_booking_model.pkl")
        sample = pd.DataFrame([{
            "hotel": "City Hotel",
            "lead_time": 350,
            "arrival_date_year": 2016,
            "arrival_date_month": "August",
            "stays_in_weekend_nights": 0,
            "stays_in_week_nights": 3,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 3,
            "booking_changes": 0,
            "deposit_type": "Non Refund",
            "customer_type": "Transient",
            "adr": 120.0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0
        }])
        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)

    def test_low_cancellation_risk_booking(self):
        """7. Verify that a loyal, low-risk booking profile is classified as NOT CANCELED (0)."""
        model = joblib.load("hotel_booking_model.pkl")
        sample = pd.DataFrame([{
            "hotel": "Resort Hotel",
            "lead_time": 5,
            "arrival_date_year": 2016,
            "arrival_date_month": "July",
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "booking_changes": 1,
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "adr": 100.0,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 2
        }])
        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
