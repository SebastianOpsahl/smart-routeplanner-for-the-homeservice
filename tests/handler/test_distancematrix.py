from handler.distancematrix import *
from tests.testdata import test_patients, test_patients_apartments
from datetime import timedelta
import unittest

class test_distancematrix(unittest.TestCase):
    def test_is_apartment(self):
        apartment, not_apartment = is_apartment("Sondresvei H0101", "Sebastiansvei 5")
        self.assertEqual(apartment, True) 
        self.assertEqual(not_apartment, False)

        apartment, also_apartment = is_apartment("Sebastiansvei 5 H0101", "Sondresvei 2 H0201")
        self.assertEqual(apartment, True)
        self.assertEqual(also_apartment, True)

        apartment, also_apartment = is_apartment("Elvegata 2", "Teknologivegen 22")
        self.assertEqual(apartment, False)
        self.assertEqual(also_apartment, False)

    def test_calculate_stairs(self):
        stairs, _ = calculate_stairs("H0202")
        self.assertEqual(stairs, 1.0)
        stairs, _ = calculate_stairs("L0201")
        self.assertEqual(stairs, 2.0)
        stairs, _ = calculate_stairs("K0101")
        self.assertEqual(stairs, 2.0)
        stairs, _ = calculate_stairs("U0101")
        self.assertEqual(stairs, 1.0)

    def test_extract_apartment_number(self):
        apartment_number = extract_apartment_number("Sondresvei 2 H0101")
        self.assertEqual(apartment_number, "H0101")
        apartment_number = extract_apartment_number("Sebastiansvei 5 L0101")
        self.assertEqual(apartment_number, "L0101")

    def test_extract_building_name(self):
        building = extract_building_name("Sondresvei 2 H0101")
        self.assertEqual(building, "Sondresvei 2")
        building = extract_building_name("Sebastiansvei 5 L0101")
        self.assertEqual(building, "Sebastiansvei 5")

    def test_biggest(self):
        bigger, smallest = biggest(10, 2)
        self.assertEqual(bigger, 10)
        self.assertEqual(smallest, 2)

    def test_matrix_greater(self):
        test_patients1 = test_patients()
        test_patients2 = test_patients_apartments()
        matrix1, _= distance_matrix_request(test_patients1)
        matrix2, _ = distance_matrix_request(test_patients2)
        sum1 = sum_matrix_durations(matrix1)
        sum2 = sum_matrix_durations(matrix2)
        
        self.assertGreater(sum2, sum1)

def sum_matrix_durations(self, matrix: list[list[timedelta]]):
    return sum([sum(row) for row in matrix])