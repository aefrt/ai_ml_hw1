from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import data_transform
import pickle
import uvicorn
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    df = pd.DataFrame([item.__dict__]).drop(columns='selling_price')
    df = data_transform.preprocess_data(df)
    return model.predict(df)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    df = pd.concat([pd.DataFrame([item.__dict__]) for item in items]).reset_index().drop(columns=['index', 'selling_price'])
    df = data_transform.preprocess_data(df)
    return model.predict(df)

if __name__ == "__main__":
    
    uvicorn.run(
        "model_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )