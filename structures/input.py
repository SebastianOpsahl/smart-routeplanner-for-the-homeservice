from datetime import datetime, timedelta
from pydantic import BaseModel

# api/v2/employees returns a list of employee can be filtered by {role}
class Employee:
    def __init__(self, id: int):
        self.id = id

class Address(BaseModel):
    city: str
    street_address: str
    zip_code: str

# Retrieved via /api/v2/patients/{id}
# Have to be retrieved when retrieving each visit in order to obtain the address of the visit
class Patient(BaseModel):
    address: Address

class Task(BaseModel):
    hours: int
    minutes: int

# Retrieved by API call to /api/v2/visits/{forDate}
class Visit(BaseModel):
    visit_id: int = None
    matrix_index: int = None
    patient_id: int = None
    patient: Patient
    start_time: datetime
    end_time: datetime
    double_staffed: bool
    tasks: list[Task]
    # temporarily optional as it is set later on
    task_time: timedelta = None

    def set_task_time(self):
        total_duration = timedelta()
        for task in self.tasks:
            total_duration += timedelta(hours=task.hours, minutes=task.minutes)
        self.task_time = total_duration
    
    def __str__(self):
        return (f"Visit ID: {self.visit_id}, Start: {self.start_time}, End: {self.end_time}, ")