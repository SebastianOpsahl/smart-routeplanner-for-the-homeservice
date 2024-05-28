from structures.structures import Input
from handler.handler import shift_handler
from ml.model import load_model
from ml.enviroment import combined_feature_size

from fastapi import FastAPI

app = FastAPI()

load_model()

@app.get("/calculate-routeplan")
async def calculate_shift(input: Input):
   return shift_handler(input)