from structures.input import Patient, Address
from requests import get
from datetime import timedelta
import re

# Standing for the floor 'types' in a building
floor_hierarchy = {
    'K': 2,
    'U': 1,
    'H': 0,
    'L': 1
}

TIME_PER_FLOOR = 20
API_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
MAX_DESTINATIONS = 25

def distance_matrix_request(depot: str, patients: list[Patient], api_key: str):
    '''Requests the distance matrix API'''
    if len(patients) == 0:
        return None, Exception("Patient list empty")
    addresses = [depot] + remove_duplicate_addresses([full_address(patient.address) for patient in patients])
    n = len(addresses)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        origin = [addresses[i]]
        destinations = addresses[i+1:]
        if len(destinations) == 0:
            break
        elements, err = get_elements(origin, destinations, api_key)
        if err != None:
            return None, err

        for j, element in enumerate(elements):
            stairs = handle_apartments(origin[0], destinations[j])
            duration_value = float(element['duration']['value']) + (stairs * TIME_PER_FLOOR)
            matrix[i][i+1+j] = duration_value
            matrix[i+1+j][i] = duration_value

    return convert_matrix_to_duration(matrix), None

def get_elements(origin: str, destinations: str, api_key: str):
    '''Handling to not exceed limitations'''
    if len(destinations) > MAX_DESTINATIONS:
        d1, d2 = split_list(destinations)
        e1, err = get_elements(origin, d1, api_key)
        if err != None:
            return None, err
        e2, err = get_elements(origin, d2, api_key)
        if err != None:
            return None, err
        return e1 + e2, None
    return get_row(origin, destinations, api_key)
    
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def get_row(origin: str, destinations: list[str], api_key: str):
    '''Returns one row from the matrix API response'''
    params = {
        "units": "imperial",
        "origins": '|'.join(origin),
        "destinations": '|'.join(destinations),
        "key": api_key
    }
    resp = get(url=API_URL, params=params)
    data = resp.json()
    rows = data["rows"]
    if len(rows) == 0:
        return None, Exception("No rows recieved from Google Distance Matrix API")
    err = check_matrix_feasibility(rows)
    if err:
        return None, err
    return data['rows'][0]['elements'], None


def remove_duplicate_addresses(addresses: list[str]):
    '''Since they will be given the same matrix_id anyways'''
    new_addresses: list[str] = []
    for address in addresses:
        if not address in new_addresses:
            new_addresses.append(address)
    return new_addresses

def full_address(address: Address):
    return address.street_address + ", " + address.city + ", " + address.zip_code

def check_matrix_feasibility(rows):
    for row in rows:
        elements = row["elements"]
        for element in elements:
            status = element["status"]
            if status == "ZERO_RESULTS":
                return Exception("ZERO_RESULTS")
            elif status == "NOT FOUND":
                return Exception("NOT FOUND")
    return None

def handle_apartments(i_address: str, j_address: str):
    row_apartment, element_apartent = is_apartment(i_address, j_address)
    # If both are apartements
    if row_apartment and element_apartent:
        i_stairs, i_up = add_appartment_distance(i_address)
        j_stairs, j_up = add_appartment_distance(j_address)
        # If in the same building
        if extract_building_name(i_address) == extract_building_name(j_address):
            # The same side of level 0
            if (i_up and j_up) or (not i_up and not j_up):
                biggest, smallest = biggest(i_stairs, j_stairs)
                return biggest - smallest
            else:
                return i_stairs + j_stairs
        else:
            return i_stairs + j_stairs
    # If either one is apartement
    elif row_apartment and not element_apartent:
        i_stairs, i_up = add_appartment_distance(i_address)
        return i_stairs
    elif not row_apartment and element_apartent:
        j_stairs, j_up = add_appartment_distance(j_address)
        return j_stairs
    return 0

def is_apartment(address: str, address2: str):
    pattern = r"^(.*?)\s[LHUK][0-9]{2}[0-9]{2}$"
    return fits_pattern(pattern, address), fits_pattern(pattern, address2)

def fits_pattern(pattern: str, txt: str):
    '''Pattern matching addresses'''
    if re.search(pattern, txt):
        return True
    return False

def add_appartment_distance(address: str):
    apartment = extract_apartment_number(address)
    stairs, up = calculate_stairs(apartment)
    return stairs, up

def extract_apartment_number(address: str):
    pattern = r"[LHUK][0-9]{2}[0-9]{2}$"
    match = re.search(pattern, address)
    if match:
        return match.group(0)
    return ""

def calculate_stairs(apartment: str):
    '''Calculates stairs based on relative position in the building'''
    floor_letter = apartment[0]
    floor_number = int(apartment[1:3])
    stairs_time = floor_hierarchy[floor_letter] + floor_number - 1
    return float(stairs_time), floor_letter == 'H' or floor_letter == 'L'

def extract_building_name(address: str):
    pattern = r"^(.*?)\s[LHUK][0-9]{2}[0-9]{2}$"
    match = re.search(pattern, address)
    if match:
        return match.group(1)
    return ""

def biggest(a: float, b: float):
    if a > b:
        return a, b
    return b, a

def convert_matrix_to_duration(matrix: list[list[int]]):
    return [[timedelta(seconds=seconds) for seconds in row] for row in matrix]