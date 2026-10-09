import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_cancellation_risk_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "lead_time": 300,
                "previous_cancellations": 2,
                "booking_changes": 0,
                "total_of_special_requests": 0,
                "required_car_parking_spaces": 0
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "CANCELED")
        self.assertEqual(response.get_json()["prediction_code"], 1)

    def test_low_cancellation_risk_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "lead_time": 15,
                "previous_cancellations": 0,
                "booking_changes": 2,
                "total_of_special_requests": 2,
                "required_car_parking_spaces": 1
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "NOT CANCELED")
        self.assertEqual(response.get_json()["prediction_code"], 0)

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "lead_time": 30
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
