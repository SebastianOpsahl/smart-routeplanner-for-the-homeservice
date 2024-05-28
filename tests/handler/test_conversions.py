from pytz import timezone
import unittest
from datetime import datetime
from handler.conversions import set_new_time_window
from tests.testdata import test_visits, Visit, Task, Patient, Address
from handler.conversions import handle_double_staffed

class test_distancematrix(unittest.TestCase):

    def test_handle_double_staffed(self):
        visits, err = test_visits()
        original_visit = visits[0]

        self.assertEqual(err, None)

        copy = handle_double_staffed(original_visit)
        self.assertEqual(original_visit, copy)

    def test_set_new_time_window(self):
        location = timezone("Europe/Oslo")
        visit_double_stacked = visit_double_stacked_for_test(location)
        set_new_time_window(0, visit_double_stacked, datetime(year=2024, month=1, day=1, hour=10, tzinfo=location))
        self.assertNotEqual(visit_double_stacked_result(location), visit_double_stacked)

def visit_double_stacked_for_test(location):
    return [
        Visit(
            visit_id=1,
            matrix_index=1,
            patient_id="0",
            patient=mock_patient(),
            start_time=datetime(year=2024, month=1, day=1, hour=14, tzinfo=location),
            end_time=datetime(year=2024, month=1, day=1, hour=14, minute=30, tzinfo=location),
            double_staffed=True,
            tasks=[Task(
                hours=0,
                minutes=19,
                competence=1
            )]
        ),
        Visit(
            visit_id=1,
            matrix_index=1,
            patient_id="0",
            patient=mock_patient(),
            start_time=datetime(year=2024, month=1, day=1, hour=14, tzinfo=location),
            end_time=datetime(year=2024, month=1, day=1, hour=14, minute=30, tzinfo=location),
            double_staffed=True,
            tasks=[Task(
                hours=0,
                minutes=19,
                competence=1
            )]
        )
    ]

def visit_double_stacked_result(location):
    return [
        Visit(
            visit_id=1,
            matrix_index=1,
            patient_id="0",
            patient=mock_patient(),
            start_time=datetime(year=2024, month=1, day=1, hour=10, tzinfo=location),
            end_time=datetime(year=2024, month=1, day=1, hour=10, tzinfo=location),
            double_staffed=True,
            tasks=[Task(
                hours=0,
                minutes=19,
                competence=1
            )]
        ),
        Visit(
            visit_id=1,
            matrix_index=1,
            patient_id="0",
            patient=mock_patient(),
            start_time=datetime(year=2024, month=1, day=1, hour=10, tzinfo=location),
            end_time=datetime(year=2024, month=1, day=1, hour=10, tzinfo=location),
            double_staffed=True,
            tasks=[Task(
                hours=0,
                minutes=19,
                competence=1
            )]
        )
    ]

def mock_patient():
    return Patient(address=Address(city="Gjøvik", street_address="Elvegata 2", zip_code="2815"))