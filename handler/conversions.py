from datetime import timedelta, datetime
from structures.input import Visit

TIMEWINDOWDIFF = timedelta(minutes=3)

def handle_double_staffed(visit: Visit):
    return visit

def set_new_time_window(visit_id: int, visits: list[Visit], new_time: datetime):
    '''Sets new time window for double staffed visits'''
    for i in range(len(visits)):
        if visits[i].visit_id == visit_id:
            visits[i].start_time = new_time - TIMEWINDOWDIFF
            visits[i].end_time = new_time + TIMEWINDOWDIFF
