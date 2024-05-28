from structures.structures import Shift, Saving, Route, BreakRestrained, ShiftRestrained, TryReverse, Solution
from structures.input import Visit, Patient, Address, Task
from datetime import timedelta, datetime
from typing import Dict

BREAK_DIFFERENCE = timedelta(minutes=5)

def initial_solution(shift: Shift, matrix: list[list[timedelta]]):
    '''Creates savings pairs s consisting of s.i and s.j. Then tries to connect these
    into savings pairs. These are merged into routes. Only valid routes (routes with something)
    within them are taken further'''
    savings = compute_savings(shift, matrix)
    routes = [Route([visit], True) for visit in shift.visits]
    assigned: Dict[int, bool] = {}

    for i in range(len(shift.visits)):
        assigned[i] = True

    while not all(assigned.values()):
        made_changes = False
        savings: list[Saving] = [s for s in savings if not assigned[s.i] and not assigned[s.j]]
        for s in savings:
            if routes[s.i].valid and routes[s.j].valid:
                can_merge, can_add, merged_routes = False, False, None
                if assigned[s.i] and assigned[s.j]:
                    combined_visits = routes[s.i].visits + routes[s.j].visits
                    combined_routes = Route(combined_visits, False)
                    merged_routes, can_merge = can_merge_routes(combined_routes, shift.shift_time, shift.break_time, matrix)
                else:
                    can_add, made_changes = handle_single_values(routes, s, matrix, assigned, shift)    
                if can_add or can_merge:
                    if can_merge:
                        routes[s.i] = merged_routes
                    if can_add:
                        routes[s.i].visits.append(shift.visits[s.j])
                    routes[s.j].valid = False
                    assigned[s.j] = True
                    assigned[s.i] = True
                    made_changes = True
        if not made_changes:
            break

    valid_routes = [route for route in routes if route.valid]

    return Solution (
        shift.shift_time,
        shift.break_time,
        valid_routes,
        matrix,
    ), None

def handle_single_values(routes: list[Route], s: Saving, matrix: list[list[timedelta]], assigned: Dict[int, bool], shift: Shift):
    '''Handle connections between savings where either both or atleast one is an object not yet a list'''
    can_add, err = can_add_visit(routes[s.i].visits, s.j, matrix, shift)
    made_changes = False
    if err is not None:
        match err:
            case TryReverse():
                if not assigned[s.i] and not assigned[s.j]:
                    can_add, err = can_add_visit(routes[s.j].visits, s.i, matrix, shift)
                    if err:
                        handle_visit_connection_error(routes, s, shift, assigned)
                    made_changes = True
            case BreakRestrained():
                routes[s.i].visits.append(return_depot(shift.break_time[0]))
                made_changes = True
            case _:
                handle_visit_connection_error(routes, s, shift, assigned)
                made_changes = True
    return can_add, made_changes

def compute_savings(shift: Shift, matrix: list[list[timedelta]]):
    '''Compute savings based on distance and time'''
    savings: list[Saving] = []
    num_visits = len(shift.visits)
    for i in range(num_visits):
        i_matrix_idx = shift.visits[i].matrix_index
        for j in range(i + 1, num_visits):
            j_matrix_idx = shift.visits[j].matrix_index
            a, b = shift.visits[i], shift.visits[j]

            if a.start_time + matrix[i_matrix_idx][j_matrix_idx] > b.end_time:
                continue

            saving_value = matrix[0][i_matrix_idx] + matrix[0][j_matrix_idx] - matrix[i_matrix_idx][j_matrix_idx]

            # adding time constraints as a factor
            time_diff = a.start_time - (b.start_time + b.task_time + matrix[i_matrix_idx][j_matrix_idx])
            if time_diff > timedelta(0):
                saving_value -= time_diff
            time_diff = b.start_time - (a.start_time + a.task_time + matrix[i_matrix_idx][j_matrix_idx])
            if time_diff > timedelta(0):
                saving_value -= time_diff
            
            s = Saving(
                i,
                j,
                saving_value,
                valid = True
            )
            savings.append(s)
    savings.sort(key=value_for_sort)
    return savings

def value_for_sort(saving: Saving):
    return -saving.value

def current_time(visits: list[Visit], matrix: list[list[timedelta]], shift: Shift):
    '''Helper function to iterate trough list upto one point. Adding break as neccessary'''
    trigger = True
    current_time = shift.shift_time[0]
    last_location = 0
    i = 0

    while i < len(visits):
        visit = visits[i]        
        if visit.matrix_index == 0:
            if trigger:
                trigger = False
            else:
                visits.pop(i)
                continue

        current_location = visit.matrix_index
        current_time += matrix[last_location][current_location]

        if current_time > visit.end_time:
            return None
        if current_time < visit.start_time:
            current_time = visit.start_time
        current_time += visit.task_time

        if current_time + matrix[current_location][0] > shift.break_time[0] - BREAK_DIFFERENCE and trigger:
            trigger = False
            depot = return_depot(shift.break_time[0])
            visits.insert(i, depot)
            current_time = shift.break_time[1]
            last_location = 0
            i += 1

        last_location = current_location
        i += 1
    
    return current_time

def can_add_visit(route: list[Visit], next_visit_index: int, matrix: list[list[timedelta]], shift: Shift):
    '''Checks feasibility of adding a visit to a route'''
    # If visit is already in route (to make on route not being able to handle one double staffed visit)
    for visit in route:
        if visit.visit_id == shift.visits[next_visit_index].visit_id:
            return False, None

    time = current_time(route, matrix, shift)
    if time is None:
        return False, None
    last_visit_index = shift.visits[route[-1].visit_id].visit_id

    start_service_time = time + matrix[shift.visits[last_visit_index].matrix_index][shift.visits[next_visit_index].matrix_index]

    # if can't reach current visits time window
    if shift.visits[next_visit_index].end_time < start_service_time:
        return False, TryReverse()
    
    time_to_depot = matrix[shift.visits[next_visit_index].matrix_index][0]
    time_end = start_service_time + shift.visits[next_visit_index].task_time

    # handles the break, as it is mandatory
    if (start_service_time < shift.break_time[0] and (time_end + time_to_depot) > shift.break_time[1] or
        start_service_time < shift.break_time[0] and (shift.break_time[0] < (time_end + time_to_depot) > shift.break_time[1])
        or time_end < shift.break_time[0] < shift.visits[next_visit_index].start_time):
        return False, BreakRestrained()

    # wait at location if it have to
    if start_service_time < shift.visits[next_visit_index].start_time:
        start_service_time = shift.visits[next_visit_index].start_time

    # check if the service can be done and be at the depot before shift end time
    if shift.shift_time[1] < time_end + time_to_depot:
        return False, ShiftRestrained()
    
    return True, None
    
def handle_visit_connection_error(routes: list[Route], s: Saving, shift: Shift, assigned: dict):
    '''Creates new Route list'''
    new_route = Route(
        visits=[shift.visits[s.j]],
        valid=True
    )
    routes.append(new_route)
    assigned[s.j] = True

def can_merge_routes(route: Route, shift_time: list[datetime], break_time: list[datetime], matrix: list[list[datetime]]):
    '''Checks if two Route lists can merge based on their savings connection'''
    trigger = True
    current_time = shift_time[0]
    last_location = 0
    i = 0

    while i < len(route.visits):
        visit = route.visits[i]        
        if visit.matrix_index == 0:
            if trigger:
                trigger = False
            else:
                route.visits.pop(i)
                continue

        current_location = visit.matrix_index
        current_time += matrix[last_location][current_location]

        if current_time > visit.end_time:
            return None, False
        if current_time < visit.start_time:
            current_time = visit.start_time
        current_time += visit.task_time

        if current_time + matrix[current_location][0] > break_time[0] - BREAK_DIFFERENCE and trigger:
            trigger = False
            depot = return_depot(break_time[0])
            route.visits.insert(i, depot)
            current_time = break_time[1]
            last_location = 0
            i += 1

        last_location = current_location
        i += 1
    
    if current_time + matrix[last_location][0] < shift_time[1]:
        return Route(
            visits=route.visits,
            valid=True
        ), True
    
    return None, False

def return_depot(break_start: datetime):
    '''Function to return the depot as a visit object'''
    return Visit(
        visit_id=-1,
        matrix_index=0,
        patient_id=-1,
        patient=Patient(address=Address(city="", street_address="", zip_code="")),
        double_staffed=False,
        start_time=break_start - BREAK_DIFFERENCE,
        end_time=break_start + BREAK_DIFFERENCE,
        task_time=timedelta(minutes=30),
        tasks= [
            Task(
                hours=0,
                minutes=30,
            )
        ]
    )