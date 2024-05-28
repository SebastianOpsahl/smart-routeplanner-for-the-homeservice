import unittest
from structures.input import Visit, Patient, Address, Task
from structures.structures import Route
from datetime import datetime, timedelta
from genetic.genetic import *

GENS = 1
MUT_RATE = 10
POP_SIZE = 1

def chrom_for_test():
    return [
            Route(
                [
                    Visit(
                        visit_id=1,
                        matrix_index=1,
                        patient_id=1,
                        patient=Patient(
                            address=Address(
                                city="Gjøvik",
                                street_address="Ibsens Gate 3",
                                zip_code="2821"
                            )
                        ),
                        start_time=datetime(year=2024, month=1, day=1, hour=9),
                        end_time=datetime(year=2024, month=1, day=1, hour=15),
                        double_staffed=False,
                        tasks=[
                            Task(0, 30)
                        ],
                        task_time=timedelta(minutes=30)
                    )
                ],
                True
            )
        ]

def time_for_test():
    return ([
        datetime(year=2024, month=1, day=1, hour=8),
        datetime(year=2024, month=1, day=1, hour=16)
    ],
    [
        datetime(year=2024, month=1, day=1, hour=12),
        datetime(year=2024, month=1, day=1, hour=12, minutes=30)
    ])

def matrix_for_test():
    return [
        [timedelta(minutes=0), timedelta(minutes=1)],
        [timedelta(minutes=1), timedelta(minutes=0)]
    ]

class test_genetic_functions(unittest.TestCase):
    def test_genetic(self):
        chrom = chrom_for_test()
        shift_time, break_time = time_for_test()
        matrix = matrix_for_test()
        result = genetic(
            chrom,
            shift_time,
            break_time,
            matrix,
            GENS,
            POP_SIZE,
            MUT_RATE
        )
        self.assertEqual(len(chrom), len(result))

    def test_generate_population(self):
        chrom = chrom_for_test()
        shift_time, break_time = time_for_test()
        matrix = matrix_for_test()
        pop_size = 1
        population = generate_population(
            chrom,
            shift_time,
            break_time,
            matrix,
            POP_SIZE,
            MUT_RATE
        )
        self.assertEqual(len(population), pop_size)

    def test_mutate(self):
        chrom = chrom_for_test()
        mut_chrom, _ = mutate(chrom, MUT_RATE)
        self.assertEqual(len(chrom), len(mut_chrom))

    def test_should_mutate(self):
        should = should_mutate(MUT_RATE)
        self.assertTrue(isinstance(should, bool))

    def test_random_up_to(self):
        x = 100
        rand_num = random_up_to(x)
        self.assertLessEqual(rand_num, x)

    def test_swap_genes(self):
        chrom = chrom_for_test()
        s = len(chrom)
        swap_genes(chrom, 0, 0)
        self.assertEqual(s, len(chrom))

    def test_random_index(self):
        route = chrom_for_test()[0]
        i = random_index(route)
        self.assertLess(i, len(route.visits))

    def test_reorder_genes(self):
        route = chrom_for_test()[0]
        s = len(route)
        reorder_genes(route)
        self.assertEqual(s, len(route))

    def test_feasible(self):
        chrom = chrom_for_test()
        shift_time, break_time = time_for_test()
        matrix = matrix_for_test()
        f = feasible(chrom, shift_time, break_time, matrix)
        self.assertTrue(f)

    def test_traverse_route(self):
        route = chrom_for_test()[0]
        shift_time, break_time = time_for_test()
        matrix = matrix_for_test()
        f, _ = traverse_route(route, shift_time, break_time, matrix)
        self.assertTrue(f)

    def test_feasible_meeting(self):
        d1 = datetime(year=2024, month=1, day=1, hour=10)
        d2 = datetime(year=2024, month=1, day=1, hour=10, minute=3)
        f = feasible_meeting(d1, d2)
        self.assertTrue(f)

    def test_sort_by_fitness(self):
        population = [chrom_for_test()]
        s = len(population)
        matrix = matrix_for_test()
        sort_by_fitness(population, matrix)
        self.assertEqual(s, len(population))

    def test_fitness(self):
        chrom = chrom_for_test()
        matrix = matrix_for_test()
        score = fitness(chrom, matrix)
        self.assertEqual(score, timedelta(minutes=2))

    def test_travel_time(self):
        route = chrom_for_test()[0]
        matrix = matrix_for_test()
        time = travel_time(route, matrix)
        self.assertEqual(time, 2)

    def test_mate(self):
        chrom = chrom_for_test()
        offspring = mate(chrom, chrom)
        self.assertEqual(len(chrom), len(offspring))

    def test_num_genes(self):
        chrom = chrom_for_test()
        num = num_genes(chrom)
        self.assertEqual(num, 1)

    def test_index(self):
        route = chrom_for_test()[0]
        gene = route[0]
        i = index(route, gene)
        self.assertEqual(i, 0)
    
    def random_new_gene(self):
        chrom = chrom_for_test()
        offspring = []
        gene = random_new_gene(chrom, offspring)
        self.assertEqual(gene.visit_id, chrom[0].visits[0].visit_id)

    def test_gene_in_chom(self):
        chrom = chrom_for_test()
        gene = chrom[0].visits[0]
        b = gene_in_chrom(chrom, gene)
        self.assertTrue(b)

    def test_random_gene(self):
        chrom = chrom_for_test()
        gene = random_gene(chrom)
        self.assertEqual(gene.visit_id, chrom[0].visits[0].visit_id)

    def test_random_index_chrom(self):
        chrom = chrom_for_test()
        i = random_index_chrom(chrom)
        self.assertEqual(i, 0)

    def test_isEven(self):
        x = 2
        b = isEven(x)
        self.assertTrue(b)

    def test_random_chromsome(self):
        chrom = chrom_for_test()
        chrom2 = random_chromosome(chrom, chrom)
        self.assertEqual(len(chrom), len(chrom2))