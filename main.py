from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()
class IrisSpecies(BaseModel):
    quantity: int
    volume: int
    distance: float


@app.post('/predict')
async def predict_species(iris: IrisSpecies):
    data = iris.dict()
    loaded_model = pickle.load(open('exp.pkl', 'rb'))
    data_in = [[data['quantity'], data['volume'], data['distance']]]
    prediction = loaded_model.predict(data_in)

    return {
    'prediction': prediction[0]
    }
