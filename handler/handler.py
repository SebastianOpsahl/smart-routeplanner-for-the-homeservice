from structures.structures import Shift, SolutionOut, Input
from handler.conversions import handle_double_staffed
from typing import Dict
from handler.distancematrix import distance_matrix_request
from ml.calculations import normalize, unormalize
from fastapi import status
from fastapi.responses import JSONResponse
from savings.savings import initial_solution
from ml.model import ml_solution


from genetic.genetic import fitness

def input2shift_and_key(input: Input):
    return Shift(
        depot=input.depot,
        shift_time=input.shift_time,
        break_time=input.break_time,
        visits=input.visits
    ), input.api_key

def shift_handler(input: Input) -> SolutionOut:
    '''Function which handles the API request and response'''
    shift, api_key = input2shift_and_key(input)
    visits = []
    visited: Dict[str, int] = {}
    i = 1
    for visit in shift.visits:
        visit.set_task_time()
        if visit.patient.address.street_address in visited:
            visit.matrix_index = visited[visit.patient.address.street_address]
        else:
            visit.matrix_index = i
            visited[visit.patient.address.street_address] = i
            i += 1
        visits.append(visit)
        if visit.double_staffed:
            cloned_visit = handle_double_staffed(visit)
            visits.append(cloned_visit)
    shift.visits = visits

    matrix, err = distance_matrix_request(shift.depot, [visit.patient for visit in shift.visits], api_key)
    if err != None:
        return JSONResponse(status_code=status.HTTP_501_NOT_IMPLEMENTED, content=str(err))

    solution, err = initial_solution(shift, matrix)
    if err != None:
        return JSONResponse(status_code=status.HTTP_501_NOT_IMPLEMENTED, content=str(err))

    print("Initial solution fitness: ", fitness(solution.routes, solution.matrix))

    normalized_solution = normalize(solution)

    normalized_complete_solution = ml_solution(normalized_solution)

    complete_solution = unormalize(normalized_complete_solution, shift.shift_time, shift.break_time)

    return complete_solution