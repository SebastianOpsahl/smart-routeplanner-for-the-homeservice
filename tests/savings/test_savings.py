from tests.testdata import test_data
from structures.structures import Shift, Visit
from structures.input import Task
from datetime import datetime
import unittest
from savings.savings import initial_solution, return_depot

class test_distancematrix(unittest.TestCase):

    def test_initial_solution(self):
        visits, _, matrix, location, err = test_data()
        assert err == None

        shift = Shift(
            depot="Skolevegen 8, 2827 Gjøvik",
            shift_time=[
                datetime(year=2024, month=1, day=1, hour=8, tzinfo=location),
                datetime(year=2024, month=1, day=1, hour=15, tzinfo=location)
            ],
            break_time=[
                datetime(year=2024, month=1, day=1, hour=12, tzinfo=location),
                datetime(year=2024, month=1, day=1, hour=12, minute=30, tzinfo=location)
            ],
            visits=visits
        )

        _, err = initial_solution(shift, matrix)
        self.assertEqual(err, None)
    
    def test_return_depot(self):
        depot = Visit(
            visit_id=0,
            matrix_index=0,
            patient_id="",
            patient=None,
            double_staffed=False,
            start_time=datetime(2024, 1, 1, 8, 0),
            end_time=datetime(2024, 1, 1, 10, 0),
            tasks= [
                Task(
                    hours=0,
                    minutes=30,
                    competence=0
                )
            ]
        )
        test_depot = return_depot()
        self.assertEqual(depot, test_depot)