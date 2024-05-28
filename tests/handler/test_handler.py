from fastapi.testclient import TestClient
import unittest
from run import app

client = TestClient(app)

class TestHandleRequest(unittest.TestCase):
    def test_empty_body(self):
        response = client.post("/dummy", data='')
        self.assertEqual(response.status_code, 422)  # FastAPI uses 422 for validation errors
        self.assertIn("request body cannot be empty", response.json()['detail'])

    def test_valid_payload(self):
        valid_payload = {
            "depot": "Central Depot",
            "shiftTime": ["2023-01-01T08:00:00Z", "2023-01-01T16:00:00Z"],
            "breakTime": ["2023-01-01T12:00:00Z", "2023-01-01T12:30:00Z"],
            "visits": [
                {
                    "patientId": "patient123",
                    "startTime": "2023-01-01T09:00:00Z",
                    "endTime": "2023-01-01T10:00:00Z",
                    "doubleStaffing": True,
                    "tasks": {
                        "timeHour": 1,
                        "timeMinute": 0,
                        "requestForCompetence": {
                            "code": 200
                        }
                    },
                    "patient": {
                        "language": "English",
                        "address": {
                            "city": "Sample City",
                            "streetAddress": "123 Sample St",
                            "zipCode": "12345"
                        }
                    }
                }
            ]
        }
        response = client.post("/dummy", json=valid_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Payload processed successfully", response.json()['message'])

    def test_invalid_time(self):
        invalid_time_payload = {
            "depot": "Central Depot",
            "shiftTime": [],
            "breakTime": [],
            "visits": [...]
        }
        response = client.post("/dummy", json=invalid_time_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("error: JSON does not match schema", response.json()['detail'])

    def test_missing_parameter(self):
        missing_parameter_payload = {
            "depot": "Central Depot",
            "visits": [...]
        }
        response = client.post("/dummy", json=missing_parameter_payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("error: JSON does not match schema", response.json()['detail'])

    def test_missing_section(self):
        missing_section_payload = '''
        {
            "depot": "Central Depot",
            "shiftTime": ["2023-01-01T08:00:00Z", "2023-01-01T16:00:00Z"],
            "breakTime": ["2023-01-01T12:00:00Z", "2023-01-01T12:30:00Z"]
        }'''
        response = client.post("/dummy", data=missing_section_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("invalid character '}'", response.json()['detail'])