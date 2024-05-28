import unittest
from tests.testdata import actual_matrix, matrix_time
from structures.input import Visit, Task, Patient, Address
from structures.structures import Route, Solution
import pytz
from ml.calculations import *

class test_helper(unittest.TestCase):

    def test_normalize(self):
        shift_time, routes, matrix = solution_for_test()
        scaled_solution = normalize(shift_time, routes, matrix)
        expected_solution = solution_for_test_scaled()

        assert_solution_equal(self, scaled_solution, expected_solution)

    def test_scale_matrix(self):
        matrix = scale_matrix(actual_matrix())

        actual = [[0.0, 0.0028125, 0.0042014, 0.0020486, 0.0051505, 0.004294, 0.0055787, 0.0022454, 0.0021065],
            [0.002662, 0.0, 0.0029167, 0.0044213, 0.0042593, 0.0030208, 0.006794, 0.0016204, 0.0019676],
            [0.0042014, 0.0033449, 0.0, 0.0060532, 0.0036574, 0.0025, 0.0062616, 0.0030093, 0.0035185],
            [0.0021412, 0.0045949, 0.0060995, 0.0, 0.0061806, 0.0058333, 0.0060764, 0.0041551, 0.0031366],
            [0.0048148, 0.0046991, 0.0036574, 0.0058333, 0.0, 0.0031134, 0.0042824, 0.004456, 0.0041319],
            [0.001713, 0.0019329, 0.0029167, 0.0035532, 0.0044792, 0.0, 0.0058333, 0.0013542, 0.0010185],
            [0.0053588, 0.0066551, 0.0063542, 0.006088, 0.0043056, 0.0057986, 0.0, 0.0060995, 0.0052662],
            [0.0022569, 0.0010532, 0.0031597, 0.0040856, 0.0047222, 0.0032523, 0.0063773, 0.0, 0.0015625],
            [0.0014236, 0.0022106, 0.0035995, 0.003125, 0.0048727, 0.0036921, 0.0055556, 0.0016435, 0.0]]

        self.assertEqual(matrix, actual)

    def test_timedelta_to_scale(self):
        t = timedelta(hours=0)
        scale = timedelta_to_scale(t)
        self.assertEqual(scale, 0)

        t = timedelta(hours=12)
        scale = timedelta_to_scale(t)
        self.assertEqual(scale, 0.5)

        t = timedelta(hours=6)
        scale = timedelta_to_scale(t)
        self.assertEqual(scale, 0.25)

        t = timedelta(hours=10, minutes=10)
        scale = timedelta_to_scale(t)
        self.assertEqual(scale, 0.4236111)

    def test_datetime_to_scale(self):
        t = datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
        scale = datetime_to_scale(t)
        self.assertEqual(scale, 0)

        t = datetime(year=2022, month=1, day=1, hour=12, minute=0, second=0)
        scale = datetime_to_scale(t)
        self.assertEqual(scale, 0.5)

        t = datetime(year=2022, month=1, day=1, hour=6, minute=0, second=0)
        scale = datetime_to_scale(t)
        self.assertEqual(scale, 0.25)

        t = datetime(year=2022, month=1, day=1, hour=10, minute=10, second=0)
        scale = datetime_to_scale(t)
        self.assertEqual(scale, 0.4236111)

def solution_for_test():
    location = pytz.timezone('Europe/Oslo')
    matrix = matrix_small()
    routes = [
        Route(
            visits=[Visit(
                    visit_id=1,
                    matrix_index=1,
                    patient_id="0",
                    patient=Patient(address=Address(city="Gjøvik", street_address="Elvegata 2", zip_code="2815")),
                    start_time=datetime(2024, 1, 1, 14, 0, tzinfo=location),
                    end_time=datetime(2024, 1, 1, 14, 30, tzinfo=location),
                    double_staffed=False,
                    tasks=[Task(hours=0, minutes=19, competence=101)]
                )],
            valid=True
        ),
        Route(
            visits=[Visit(
                visit_id=2,
                matrix_index=2,
                patient_id="1",
                patient=Patient(address=Address(city="Gjøvik", street_address="Teknologivegen 22", zip_code="2815")),
                start_time=datetime(2024, 1, 1, 8, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 10, 0, tzinfo=location),
                double_staffed=True,
                tasks=[Task(hours=0, minutes=45, competence=102)]
            )],
            valid=True
        )
    ]

    shift_time = [
        datetime(2024, 1, 1, 6, 0, tzinfo=location),
        datetime(2024, 1, 1, 18, 0, tzinfo=location),
        datetime(2024, 1, 1, 12, 0, tzinfo=location),
        datetime(2024, 1, 1, 12, 30, tzinfo=location),
    ]

    return shift_time, routes, matrix

def matrix_small():
    return [
        [
            matrix_time(0, 0),
            matrix_time(4, 3),
            matrix_time(6, 3),
        ],
        [
            matrix_time(3, 50),
            matrix_time(0, 0),
            matrix_time(4, 12),
        ],
        [
            matrix_time(6, 3),
            matrix_time(4, 49),
            matrix_time(0, 0),
        ],
    ]

def solution_for_test_scaled():
    scaled_matrix = [
        [0.0, 0.0028125, 0.0042014], 
        [0.0026620, 0.0, 0.0029167],
        [0.0042014, 0.0033449, 0.0]
    ]

    routes = [
        EnvRoute(visits=[
            EnvVisit(
                visit_id=1,
                matrix_index=1,
                start_time=0.3333333,
                end_time=0.3541667,
                task_time=0.0131944,
                double_staffed=0.0
            ),
        ]),
        EnvRoute(visits=[
            EnvVisit(
                visit_id=2,
                matrix_index=2,
                start_time=0.0833333,
                end_time=0.1666667,
                task_time=0.03125,
                double_staffed=1.0
            )
        ]),
        EnvRoute(visits=[
            EnvVisit(
                visit_id=2,
                matrix_index=2,
                start_time=0.0833333,
                end_time=0.1666667,
                task_time=0.03125,
                double_staffed=1.0
            )
        ])
    ]

    shift_time = [0.0, 0.5]
    break_time = [0.25, 0.2708333]

    return [Solution(shift_time, break_time, routes, scaled_matrix)]

def assert_solution_equal(test_case, actual_solution: Solution, expected_solution: Solution):
    if actual_solution.matrix != expected_solution.matrix:
        test_case.fail(f"Matrix mismatch. \nActual: {actual_solution.matrix}\nExpected: {expected_solution.matrix}")

    if actual_solution.shift_time != expected_solution.shift_time:
        test_case.fail(f"Shift time mismatch. \nActual: {actual_solution.shift_time}\nExpected: {expected_solution.shift_time}")

    if actual_solution.break_time != expected_solution.break_time:
        test_case.fail(f"Shift time mismatch. \nActual: {actual_solution.break_time}\nExpected: {expected_solution.break_time}")
    
    if len(actual_solution.routes) != len(expected_solution.routes):
        test_case.fail(f"Route list length mismatch. \nActual: {len(actual_solution.routes)}\nExpected: {len(expected_solution.routes)}")
    
    for i, (actual_route, expected_route) in enumerate(zip(actual_solution.routes, expected_solution.routes)):
        if len(actual_route.visits) != len(expected_route.visits):
            test_case.fail(f"Visit list length mismatch in route {i}. \nActual: {len(actual_route.visits)}\nExpected: {len(expected_route.visits)}")
        
        for j, (actual_visit, expected_visit) in enumerate(zip(actual_route.visits, expected_route.visits)):
            if not actual_visit == expected_visit:
                test_case.fail(f"Visit mismatch in route {i}, visit {j}. \nActual: {actual_visit}\nExpected: {expected_visit}")

