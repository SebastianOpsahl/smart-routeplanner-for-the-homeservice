from structures.input import Visit, Task, Patient, Address
from structures.structures import Route, EnvRoute, Solution
from typing import TypeAlias, Dict
from datetime import timedelta, datetime
from savings.savings import BREAK_DIFFERENCE
import numpy as np
from copy import deepcopy, copy
from random import randint
from savings.savings import return_depot

BREAK_DIFFERENCE_NORMALIZED = np.float32(0.003472222)

Gene: TypeAlias = Visit
Chromosome: TypeAlias = list[Route]
Population: TypeAlias = list[Chromosome]
DistanceMatrix: TypeAlias = list[list[timedelta]]

class MutRate:
    '''Class for holding all mutation rates as one parameter'''
    def __init__(self, swap_rate: int, reorder_rate: int, move_rate: int):
        self.swap_rate = swap_rate
        self.reorder_rate = reorder_rate
        self.move_rate = move_rate

def genetic(solution: Solution, gens: int, pop_size: int, mut_rate: MutRate, max_mut_attempts: int):
    '''Main function of the genetic module'''
    chromosome = import_chrom(solution.routes)
    population = generate_population(chromosome, solution.shift_time, solution.break_time, solution.matrix, pop_size, mut_rate, max_mut_attempts)
    for i in range(gens):
        population = sort_by_fitness(population, solution.matrix)
        s = int(10 * pop_size / 100)
        next_gen = population[:s]
        s = int(90 * pop_size / 100)
        highest = int(pop_size / 2) - 1
        for _ in range(s):
            parent1 = population[random_up_to(highest)]
            parent2 = population[random_up_to(highest)]
            offspring = mate(parent1, parent2)

            feasible, break_indices = genetic_feasible(offspring, solution.shift_time, solution.break_time, solution.matrix)
            if feasible:
                next_gen.append(add_break_indices(offspring, break_indices))
            else:
                next_gen.append(random_chromosome(parent1, parent2))
        population = next_gen
    new_solution = solution
    new_solution.routes = export_chrom(population[0], solution.break_time[0])
    return new_solution

def add_break_indices(chromosome: Chromosome, break_indices: list[int]):
    '''Gives every route a break_index based on the break_indices list'''
    new_chrom = chromosome
    for i in range(len(chromosome)):
        route = chromosome[i]
        route.break_index = break_indices[i]
        new_chrom[i] = route
    return new_chrom

def import_chrom(chromosome: Chromosome):
    '''Prepares the chromosome for the genetic module by removings the break visit and adding it as break_index'''
    new_chrom = chromosome
    for i in range(len(chromosome)):
        route = chromosome[i]
        found_break = False
        for j in range(len(route.visits)):
            if route.visits[j].visit_id == -1:
                route.visits.pop(j)
                route.break_index = j
                found_break = True
                break
        if not found_break:
            route.break_index = -1
        new_chrom[i] = route
    return new_chrom

def export_chrom(chromosome: Chromosome, break_start: datetime):
    '''Prepares the chromosome for returned solution by adding the break visit where the break_index points'''
    new_chrom = chromosome
    for i in range(len(chromosome)):
        route = chromosome[i]
        new_chrom[i].visits.insert(route.break_index, return_depot(break_start))
    return new_chrom

def generate_population(chromosome: Chromosome, shift_time: list[datetime], break_time: list[datetime],
                    matrix: DistanceMatrix, pop_size: int, mut_rate: MutRate, max_mut_attempts: int):
    '''Generates a population of mutated chromosomes based on the original chromosome'''
    population = [chromosome]
    for i in range(1, pop_size):
        mut_chrom, mutated = mutate(chromosome, mut_rate)
        if mutated:
            found_feasible = False
            for _ in range(max_mut_attempts):
                mut_chrom, mutated = mutate(chromosome, mut_rate)
                feasible, break_indices = genetic_feasible(mut_chrom, shift_time, break_time, matrix)
                if feasible and mutated:
                    population.append(add_break_indices(mut_chrom, break_indices))
                    found_feasible = True
                    break
            if not found_feasible:
                population.append(chromosome)
        else:
            population.append(chromosome)
    return population

def mutate(chromosome: Chromosome, mut_rate: MutRate):
    '''Mutates a chromosome by the use of multiple types of mutations'''
    mut_chrom = deepcopy(chromosome)
    mutated = False
    for i in range(len(mut_chrom)):
        if i == len(mut_chrom):
            break
        for j in range(len(mut_chrom[i].visits)):
            for k in range(j + 1, len(mut_chrom[i].visits)):
                if should_mutate(mut_rate.reorder_rate):
                    reorder_genes(mut_chrom[i], j, k)
                    mutated = True
        for j in range(i+1, len(mut_chrom)):
            if i >= len(mut_chrom) or j >= len(mut_chrom):
                break
            if should_mutate(mut_rate.swap_rate):
                swap_genes(mut_chrom, i, j)
                mutated = True
            if should_mutate(mut_rate.move_rate):
                move_gene(mut_chrom, i, j)
                mutated = True
    return mut_chrom, mutated

def should_mutate(mut_rate: int):
    '''Checks if a mutation should happen based on a mut_rate/100 chance'''
    return random_up_to(100) <= mut_rate

def random_up_to(max: int):
    '''Returns a random number between and including 0 and max'''
    return randint(0, max)

def move_gene(chromosome: Chromosome, a: int, b: int):
    '''Moves a random gene from route a to route b'''
    i = random_index(chromosome[a])
    j = random_up_to(len(chromosome[b].visits))
    gene = chromosome[a].visits[i]
    chromosome[a].visits.pop(i)
    chromosome[b].visits.insert(j, gene)
    if len(chromosome[a].visits) == 0:
        chromosome.pop(a)

def swap_genes(chromosome: Chromosome, a: int, b: int):
    '''Swaps two random genes between rote a and route b'''
    i = random_index(chromosome[a])
    j = random_index(chromosome[b])
    gene1 = chromosome[a].visits[i]
    gene2 = chromosome[b].visits[j]
    chromosome[a].visits[i] = gene2
    chromosome[b].visits[j] = gene1

def random_index(route: Route):
    '''Returns a random index of route.visits'''
    return random_up_to(len(route.visits) - 1)

def reorder_genes(route: Route, a: int, b: int):
    '''Switches the place of two genes within one route'''
    gene1 = route.visits[a]
    gene2 = route.visits[b]
    route.visits[a] = gene2
    route.visits[b] = gene1

def genetic_feasible(chromosome: Chromosome, shift_time: list[datetime], break_time: list[datetime], matrix: DistanceMatrix):
    '''feasible() function for use by genetic module'''
    double_staffed: Dict[int, datetime] = {}
    break_indices: list[int] = []
    for route in chromosome:
        feasible, m, index = genetic_traverse_route(route, shift_time, break_time, matrix)
        if not feasible:
            return False, None
        for id in m:
            if id in double_staffed:
                if not feasible_meeting(m[id], double_staffed[id]):
                    return False, None
                else:
                    double_staffed[id] = m[id]
        break_indices.append(index)
    return True, break_indices

def genetic_traverse_route(route: Route, shift_time: list[datetime], break_time: list[datetime], matrix: DistanceMatrix):
    '''traverse_route() function for use by genetic module'''
    double_staffed: Dict[int, datetime | np.float32] = {}
    break_index = -1 #Temp value
    current_time = shift_time[0]
    last_location = 0
    trigger = True
    for i in range(len(route.visits)):
        visit = route.visits[i]
        current_location = visit.matrix_index
        current_time += matrix[last_location][current_location]
        if current_time > visit.end_time:
            return False, None, None
        if current_time < visit.start_time:
            current_time = visit.start_time
        v_time = visit.task_time
        current_time += v_time
        if current_time + matrix[current_location][0] > break_time[0] and trigger:
            break_index = i
            trigger = False
            current_time = break_time[1]
            last_location = 0
            current_time += matrix[last_location][current_location]
            if current_time > visit.end_time:
                return False, None, None
            if current_time < visit.start_time:
                current_time = visit.start_time
            current_time += v_time
        double_staffed[visit.visit_id] = current_time - v_time
        last_location = current_location
    return current_time + matrix[last_location][0] < shift_time[1], double_staffed, break_index

def feasible(chromosome: Chromosome | list[EnvRoute], shift_time: list[datetime] | list[np.float32], break_time: list[datetime] 
            | list[np.float32], matrix: DistanceMatrix | list[list[np.float32]]):
    '''Checks if the chromosome is a possible solution'''
    double_staffed: Dict[int, datetime] = {}
    for route in chromosome:
        feasible, m = traverse_route(route, shift_time, break_time, matrix)
        if not feasible:
            return False
        for id in m:
            if id in double_staffed:
                if not feasible_meeting(m[id], double_staffed[id]):
                    return False
                else:
                    double_staffed[id] = m[id]
    return True

def traverse_route(route: Route | EnvRoute, shift_time: list[datetime] | list[np.float32], break_time: list[datetime] | 
                list[np.float32], matrix: DistanceMatrix | list[list[np.float32]]):
    '''Traverses route and checks if the route is possible while also finding all the doublestaffed visits'''
    double_staffed: Dict[int, datetime | np.float32] = {}
    b_difference = BREAK_DIFFERENCE if isinstance(matrix[0][0], timedelta) else BREAK_DIFFERENCE_NORMALIZED
    current_time = shift_time[0]
    trigger = True
    last_location = 0
    for visit in route.visits:
        current_location = visit.matrix_index
        current_time += matrix[last_location][current_location]
        if current_time > visit.end_time:
            return False, None
        if current_time < visit.start_time:
            current_time = visit.start_time
        current_time += visit.task_time

        if current_time + matrix[current_location][0] > break_time[0] - b_difference and trigger:
            if visit.matrix_index != 0 and len(route.visits) > 1:
                return False, None
            trigger = False
        
        if visit.matrix_index != 0:
            double_staffed[visit.visit_id] = current_time - visit.task_time
        last_location = current_location    
    return current_time + matrix[last_location][0] < shift_time[1], double_staffed

MINDIFF = timedelta(minutes=15)
def feasible_meeting(time1: datetime, time2: datetime):
    '''Checks if two epmloyees arrive without too much difference in arrival time'''
    diff = time1 - time2
    return diff >= -MINDIFF and diff <= MINDIFF

def sort_by_fitness(population: Population, matrix: DistanceMatrix):
    '''Sorts the population by fitness from low to high'''
    return sorted(population, key=lambda x: genetic_fitness(x, matrix))

def genetic_fitness(chromosome: Chromosome, matrix: DistanceMatrix):
    '''fitness() function for use by genetic module'''
    time = timedelta(0)
    extra = timedelta(days=len(chromosome))
    return sum([genetic_travel_time(route, matrix) for route in chromosome], time) + extra

def genetic_travel_time(route: Route, matrix: DistanceMatrix):
    '''travel_time() function for use by genetic module'''
    time = timedelta(0)
    matrix_indices = [visit.matrix_index for visit in route.visits]
    if route.break_index > -1:
        matrix_indices.insert(route.break_index, 0)

    last = 0
    for i in matrix_indices:
        current = i
        time += matrix[last][current]
        last = current
    return time + matrix[last][0]

def fitness(chromosome: Chromosome | list[EnvRoute], matrix: DistanceMatrix | list[list[np.float32]]):
    '''Calculates the fitness of the chromosome by total travel time plus ammount of vehicles as days.
    The lower fitness, the better solution'''
    time = timedelta(0) if isinstance(matrix[0][0], timedelta) else np.float32(0)
    extra = timedelta(days=len(chromosome)) if isinstance(matrix[0][0], timedelta) else np.float32(len(chromosome))
    return sum([travel_time(route, matrix) for route in chromosome], time) + extra


def travel_time(route: Route | EnvRoute, matrix: DistanceMatrix | list[list[np.float32]]):
    '''Calculates how much time goes to drive the route'''
    time = timedelta(0) if isinstance(matrix[0][0], timedelta) else np.float32(0)

    last = 0
    for visit in route.visits:
        current = visit.matrix_index
        time += matrix[last][current]
        last = current
    return time + matrix[last][0]

def mate(parent1: Chromosome, parent2: Chromosome):
    '''Generates a new chromosome by combining parent1 and parent2'''
    offspring = deepcopy(parent1)
    for i in range(len(parent1)):
        visits = copy(parent1[i].visits)
        if len(visits) > 1:
            offspring[i].visits = [visits[0], visits[1]]
        else:
            offspring[i].visits = visits
    parent_size = num_genes(parent1)

    loop = True
    while loop:
        if num_genes(offspring) == parent_size:
            loop = False
            break
        for i in range(len(offspring)):
            parent = parent1 if isEven(len(offspring[i].visits)) else parent2
            offspring[i].visits.append(next_gene(parent, offspring, offspring[i]))
            if num_genes(offspring) == parent_size:
                loop = False
                break
    return offspring
            
def num_genes(chromosome: Chromosome):
    '''Returns the number of genes in all the routes of a chromosome'''
    return sum([len(route.visits) for route in chromosome])

def next_gene(parent: Chromosome, offspring: Chromosome, current_route: Route):
    '''Finds the next gene in the parent chromosome that should be added to the offspring'''
    last_gene = current_route.visits[-1]
    for i in range(len(parent)):
        j = index(parent[i], last_gene)
        if j != -1:
            if j == len(parent[i].visits) - 1:
                return random_new_gene(parent, offspring)
            new_gene = parent[i].visits[j+1]
            if gene_in_chrom(offspring, new_gene):
                return random_new_gene(parent, offspring)
            return new_gene
    return random_new_gene(parent, offspring)

def index(route: Route, gene: Gene):
    '''Finds the route.visits index of a gene'''
    try:
        i = route.visits.index(gene)
        return i
    except ValueError:
        return -1
    
def random_new_gene(parent: Chromosome, offspring: Chromosome):
    '''Returns a random_gene from parent that is not in offspring'''
    missing = sum([[visit for visit in route.visits if not gene_in_chrom(offspring, visit)] for route in parent], [])
    return random_gene_from_gene_list(missing)

def gene_in_chrom(chromsome: Chromosome, gene: Gene):
    '''Checks if one of the chromosome's routes containts a gene'''
    found_double_staffed = False
    for route in chromsome:
        for visit in route.visits:
            if visit.visit_id == gene.visit_id:
                if gene.double_staffed:
                    if found_double_staffed:
                        return True
                    else:
                        found_double_staffed = True
                else:
                    return True
    return False

def random_gene(chromosome: Chromosome):
    '''Returns a ranodm gene from a random route from the chromosome'''
    route = chromosome[random_index_chrom(chromosome)]
    return route.visits[random_index(route)]

def random_gene_from_gene_list(genes: list[Gene]):
    '''Returns a random gene from a list of genes'''
    return genes[random_up_to(len(genes) - 1)]

def random_index_chrom(chromosome: Chromosome):
    '''Returns a random index of the chromosome'''
    return random_up_to(len(chromosome) - 1)

def isEven(x: int):
    '''Checks if an integer is an even number'''
    return x % 2 == 0

def random_chromosome(chrom1: Chromosome, chrom2: Chromosome):
    '''Picks a radom chromosome of two'''
    if randint(0, 1) == 0:
        return chrom1
    return chrom2