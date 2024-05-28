from structures.input import Visit, Task, Patient, Address, Employee
from datetime import datetime, timedelta
# takes into account daylight time zone changes etc. which datetime does not
import pytz
import random


def test_data():
    try:
        visits, e = test_visits()
        employees = test_employees()
        matrix = actual_matrix()
        location = pytz.timezone('Europe/Oslo')
    except Exception as e:
        return None, None, None, None, e
    return visits, employees, matrix, location, None

def test_visits():
    try:
        patients = test_patients()
        location = pytz.timezone('Europe/Oslo')
    except Exception as e:
        return None, e

    visits = [
            Visit(
                visit_id=1,
                matrix_index=1,
                patient_id="0",
                patient=patients[0],
                start_time=datetime(2024, 1, 1, 14, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 14, 30, tzinfo=location),
                double_staffed=False,
                tasks=[Task(hours=0, minutes=19, competence=101)]
            ),
            Visit(
                visit_id=2,
                matrix_index=2,
                patient_id="1",
                patient=patients[1],
                start_time=datetime(2024, 1, 1, 8, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 10, 0, tzinfo=location),
                double_staffed=True,
                tasks=[Task(hours=0, minutes=45, competence=102)]
            ),
            Visit(
                visit_id=3,
                matrix_index=3,
                patient_id="2",
                patient=patients[2],
                start_time=datetime(2024, 1, 1, 8, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 8, 30, tzinfo=location),
                double_staffed=False,
                tasks=[Task(hours=0, minutes=30, competence=103)]
            ),
            Visit(
                visit_id=4,
                matrix_index=4,
                patient_id="3",
                patient=patients[3],
                start_time=datetime(2024, 1, 1, 8, 30, tzinfo=location),
                end_time=datetime(2024, 1, 1, 9, 30, tzinfo=location),
                double_staffed=True,
                tasks=[Task(hours=0, minutes=20, competence=201)]
            ),
            Visit(
                visit_id=5,
                matrix_index=5,
                patient_id="4",
                patient=patients[4],
                start_time=datetime(2024, 1, 1, 8, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 9, 30, tzinfo=location),
                double_staffed=False,
                tasks=[Task(hours=0, minutes=21, competence=105)]
            ),
            Visit(
                visit_id=6,
                matrix_index=6,
                patient_id="5",
                patient=patients[5],
                start_time=datetime(2024, 1, 1, 8, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 10, 0, tzinfo=location),
                double_staffed=True,
                tasks=[Task(hours=0, minutes=22, competence=202)]
            ),
            Visit(
                visit_id=7,
                matrix_index=7,
                patient_id="6",
                patient=patients[6],
                start_time=datetime(2024, 1, 1, 10, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 11, 0, tzinfo=location),
                double_staffed=False,
                tasks=[Task(hours=0, minutes=27, competence=301)]
            ),
            Visit(
                visit_id=8,
                matrix_index=8,
                patient_id="7",
                patient=patients[7],
                start_time=datetime(2024, 1, 1, 12, 0, tzinfo=location),
                end_time=datetime(2024, 1, 1, 14, 0, tzinfo=location),
                double_staffed=True,
                tasks=[Task(hours=0, minutes=45, competence=302)]
            ),
    ]

    for visit in visits:
        visit.set_task_time()

    return visits, None

def test_patients():
    return [
        Patient(address=Address(city="Gjøvik", street_address="Elvegata 2", zip_code="2815")),
        Patient(address=Address(city="Gjøvik", street_address="Teknologivegen 22", zip_code="2815")),
        Patient(address=Address(city="Gjøvik", street_address="Kronprins Olavs veg 35", zip_code="2819")),
        Patient(address=Address(city="Gjøvik", street_address="Skolevegen 8", zip_code="2827")),
        Patient(address=Address(city="Gjøvik", street_address="Skolegata 5", zip_code="2821")),
        Patient(address=Address(city="Gjøvik", street_address="Øverbyvegen 93", zip_code="2819")),
        Patient(address=Address(city="Gjøvik", street_address="Nedre Torvgate 4A", zip_code="2815")),
        Patient(address=Address(city="Gjøvik", street_address="Marcus Thranes gate 8", zip_code="2821")),
    ]

def test_patients_apartments():
    return [
        Patient(address=Address(city="Gjøvik", street_address="Elvegata 2", zip_code="2815")),
        Patient(address=Address(city="Gjøvik", street_address="Teknologivegen 22 H0101", zip_code="2815")),
        Patient(address=Address(city="Gjøvik", street_address="Teknologivegen 22 H0202", zip_code="2815")),
        Patient(address=Address(city="Gjøvik", street_address="Skolevegen 8", zip_code="2827")),
        Patient(address=Address(city="Gjøvik", street_address="Skolegata 5", zip_code="2821")),
        Patient(address=Address(city="Gjøvik", street_address="Øverbyvegen 93", zip_code="2819")),
        Patient(address=Address(city="Gjøvik", street_address="Nedre Torvgate 4A H0401", zip_code="2815")),
        Patient(address=Address(city="Gjøvik", street_address="Marcus Thranes gate 8 H0601", zip_code="2821")),
    ]

def test_cases():
    visits = test_visits()

    return [visits[0:10], visits[1:11], visits[2:12], visits[3:13], visits[4:14], visits[5:15], visits[6:16], visits[7:17], visits[8:18], visits[9:19],
        visits[10:20], visits[11:21], visits[12:22], visits[13:23], visits[14:24], visits[15:25], visits[16:26], visits[17:27], visits[18:28],
        visits[19:29], visits[20:30],]


def test_employees():
    return [Employee(i) for i in range(5)]

def generate_random_visit_subsets(visits, subset_size, num_subsets):
    subsets = []
    for _ in range(num_subsets):
        random_subset = random.sample(visits, subset_size)
        subsets.append(random_subset)
    return subsets

def actual_matrix():
    return [
        [
            matrix_time(0, 0),
            matrix_time(4, 3),
            matrix_time(6, 3),
            matrix_time(2, 57),
            matrix_time(7, 25),
            matrix_time(6, 11),
            matrix_time(8, 2),
            matrix_time(3, 14),
            matrix_time(3, 2)
        ],
        [
            matrix_time(3, 50),
            matrix_time(0, 0),
            matrix_time(4, 12),
            matrix_time(6, 22),
            matrix_time(6, 8),
            matrix_time(4, 21),
            matrix_time(9, 47),
            matrix_time(2, 20),
            matrix_time(2, 50)
        ],
        [
            matrix_time(6, 3),
            matrix_time(4, 49),
            matrix_time(0, 0),
            matrix_time(8, 43),
            matrix_time(5, 16),
            matrix_time(3, 36),
            matrix_time(9, 1),
            matrix_time(4, 20),
            matrix_time(5, 4)
        ],
        [
            matrix_time(3, 5),
            matrix_time(6, 37),
            matrix_time(8, 47),
            matrix_time(0, 0),
            matrix_time(8, 54),
            matrix_time(8, 24),
            matrix_time(8, 45),
            matrix_time(5, 59),
            matrix_time(4, 31)
        ],
        [
            matrix_time(6, 56),
            matrix_time(6, 46),
            matrix_time(5, 16),
            matrix_time(8, 24),
            matrix_time(0, 0),
            matrix_time(4, 29),
            matrix_time(6, 10),
            matrix_time(6, 25),
            matrix_time(5, 57),
        ],
        [
            matrix_time(2, 28),
            matrix_time(2, 47),
            matrix_time(4, 12),
            matrix_time(5, 7),
            matrix_time(6, 27),
            matrix_time(0, 0),
            matrix_time(8, 24),
            matrix_time(1, 57),
            matrix_time(1, 28)
        ],
        [
            matrix_time(7, 43),
            matrix_time(9, 35),
            matrix_time(9, 9),
            matrix_time(8, 46),
            matrix_time(6, 12),
            matrix_time(8, 21),
            matrix_time(0, 0),
            matrix_time(8, 47),
            matrix_time(7, 35)
        ],
        [
            matrix_time(3, 15),
            matrix_time(1, 31),
            matrix_time(4, 33),
            matrix_time(5, 53),
            matrix_time(6, 48),
            matrix_time(4, 41),
            matrix_time(9, 11),
            matrix_time(0, 0),
            matrix_time(2, 15)
        ],
        [
            matrix_time(2, 3),
            matrix_time(3, 11),
            matrix_time(5, 11),
            matrix_time(4, 30),
            matrix_time(7, 1),
            matrix_time(5, 19),
            matrix_time(8, 0),
            matrix_time(2, 22),
            matrix_time(0, 0)
        ]
    ]

def matrix_time(minutes: int, seconds: int):
    return timedelta(minutes=minutes, seconds=seconds)
