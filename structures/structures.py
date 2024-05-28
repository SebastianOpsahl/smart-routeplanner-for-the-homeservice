from datetime import datetime, timedelta
from structures.input import Visit, Patient, Task
from pydantic import BaseModel
import numpy as np

# Input from API
class Input(BaseModel):
    depot: str
    shift_time: list[datetime]
    break_time: list[datetime]
    visits: list[Visit]
    api_key: str

class Shift(BaseModel):
    depot: str
    shift_time: list[datetime]
    break_time: list[datetime]
    visits: list[Visit]

class Saving:
    def __init__(self, i: int, j: int, value: timedelta, valid: bool):
        self.i = i
        self.j = j
        self.value = value
        self.valid = valid

class Route:
    visits: list[Visit]
    valid: bool
    break_index: int = None

    def __init__(self, visits: list[Visit], valid: bool):
        self.visits = visits
        self.valid = valid

    # For nicely printing for testing purposes
    def __str__(self):
        visits_str = ', '.join(str(visit) for visit in self.visits)
        return f"Route with visits [{visits_str}] and validity is {'valid' if self.valid else 'invalid'}"

# Classes marked with 'env' are used by the AI component, these needs to be normalized
class EnvVisit:
    def __init__(self, visit_id: int, matrix_index: int, patient: Patient, start_time: np.float32, 
                 end_time: np.float32, task_time: np.float32, double_staffed: np.float32, tasks: list[Task]):    
        self.visit_id = visit_id
        self.matrix_index = matrix_index
        self.patient = patient
        self.start_time = start_time
        self.end_time = end_time
        self.task_time = task_time
        self.double_staffed = double_staffed
        self.tasks = tasks
    
    def __str__(self):
        return (f"Visit ID: {self.visit_id}, "
                f"Start Time: {self.start_time}, End Time: {self.end_time}, "
                f"Task Time: {self.task_time}, Double Staffed: {self.double_staffed}")

    def __eq__(self, other):
        if not isinstance(other, EnvVisit):
            return NotImplemented
        return (self.visit_id == other.visit_id and
                self.start_time == other.start_time and
                self.end_time == other.end_time and
                self.task_time == other.task_time and
                self.double_staffed == other.double_staffed)

class EnvRoute:
    def __init__(self, visits: list[EnvVisit]):
        self.visits = visits
    
    def __str__(self):
        visits_str = "\n".join(str(visit) for visit in self.visits)
        return f"Route with visits:\n{visits_str}"

    def __eq__(self, other):
        if not isinstance(other, EnvRoute):
            return NotImplemented
        return self.visits == other.visits

class Solution:
    shift_time: list[datetime]
    break_time: list[datetime]
    routes: list[Route]
    matrix: list[list[timedelta]]

    def __init__(self, shift_time: list[datetime], break_time: list[datetime], routes: list[Route], matrix: list[list[datetime]]):
        self.shift_time=shift_time
        self.break_time = break_time
        self.routes = routes
        self.matrix = matrix

    def __str__(self):
        routes_str = "\n\n".join(str(route) for route in self.routes)
        matrix_str = "\n".join(" ".join(str(cell) for cell in row) for row in self.matrix)
        return f"Solution:\nRoutes:\n{routes_str}\n\nMatrix:\n{matrix_str}"

    def __eq__(self, other):
        if not isinstance(other, Solution):
            return NotImplemented
        return self.routes == other.routes and self.matrix == other.matrix
    
    def swap_visits(self, visit1_id: int, visit2_id: int):
        route1, idx1 = self.find_visit_by_id(visit1_id)
        route2, idx2 = self.find_visit_by_id(visit2_id)

        if route1 is not None and route2 is not None:
            route1.visits[idx1], route2.visits[idx2] = route2.visits[idx2], route1.visits[idx1]

    def move_visit(self, visit1_id: int, visit2_id: int, before_after):
        after = (before_after.item() == 1)
        route1, idx1 = self.find_visit_by_id(visit1_id)
        route2, idx2 = self.find_visit_by_id(visit2_id)            

        if route1 is not None and route2 is not None:
            visit_to_move = route1.visits.pop(idx1)

            if after:
                if idx2 + 1 < len(route2.visits):
                    route2.visits.insert(idx2+1, visit_to_move)
                else:
                    route2.visits.append(visit_to_move)
            else:
                if idx2 - 1 >= 0:
                    route2.visits.insert(idx2, visit_to_move)
                else:
                    route2.visits.insert(0, visit_to_move)
        
    def find_visit_by_id(self, visit_id: int):
        for route in self.routes:
            for idx, visit in enumerate(route.visits):
                if visit.visit_id == visit_id:
                    return route, idx
        return None, None

# Classes used for return of data
class VisitOut(BaseModel):
    patient: Patient
    start_time: datetime
    end_time: datetime
    double_staffed: bool
    task_time: timedelta
    double_staffed: bool
    tasks: list[Task]

class RouteOut(BaseModel):
    visits: list[VisitOut]

class SolutionOut(BaseModel):
    shift_time: list[datetime]
    break_time: list[datetime]
    routes: list[RouteOut]

# Predefined errors
class BreakRestrained(Exception):
    def __init__(self):
        super().__init__("break restrained")

class ShiftRestrained(Exception):
    def __init__(self):
        super().__init__("shift restrained")

class TryReverse(Exception):
    def __init__(self):
        super().__init__("a->b didn't work try b->a")