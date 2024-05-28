import unittest
from tests.testdata import test_data
from structures.structures import Shift
from savings.savings import initial_solution
from genetic.genetic import genetic
from datetime import datetime

class TestAll(unittest.TestCase):
    def test_all(self):
        visits, _, matrix, location, err = test_data()
        self.assertIsNotNone(err)
        shift = Shift(
            depot="Skolevegen 8, 2827 Gjøvik",
            shift_time=[datetime(2024, 1, 1, 8, 0, 0, 0, tzinfo=location), datetime(2024, 1, 1, 15, 0, 0, 0, tzinfo=location)],
            break_time=[datetime(2024, 1, 1, 12, 0, 0, 0, tzinfo=location), datetime(2024, 1, 1, 12, 30, 0, 0, tzinfo=location)],
            visits=visits
        )
        routes, err = initial_solution(shift, matrix)
        self.assertIsNotNone(err)
        gens = 10
        pop_size = 10
        mut_rate = 10
        routes = genetic(
            routes,
            [
                shift.shift_time[0],
                shift.break_time[0],
                shift.break_time[1],
                shift.shift_time[1]
            ],
            matrix,
            gens,
            pop_size,
            mut_rate
        )