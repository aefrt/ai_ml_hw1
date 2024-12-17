import requests
import io
import csv
import pandas as pd

data = {
    'name': 'Maruti Swift Dzire VDI',
    'year': 2014,
    'selling_price': 450000,
    'km_driven': 145500,
    'fuel': 'Diesel',
    'seller_type': 'Individual',
    'transmission': 'Manual',
    'owner': 'First Owner',
    'mileage': '23.4 kmpl',
    'engine': '1248 CC',
    'max_power': '74 bhp',
    'torque': '190Nm@ 2000rpm',
    'seats': 5.0
}

response = requests.post(
    'http://localhost:8000/predict_item', 
    json=data
)

print(response.text)

df = pd.read_csv('df_train.csv').head(3)
if 'Unnamed: 0' in df.columns.tolist():
    df.drop(columns='Unnamed: 0', inplace=True)

response = requests.post(
    'http://localhost:8000/predict_items', 
    json=[df.iloc[i,:].to_dict() for i in range(df.shape[0])]
)

print(response.text)