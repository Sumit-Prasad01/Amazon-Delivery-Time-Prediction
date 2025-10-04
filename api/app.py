from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import numpy as np
from config.paths_config import *

# Load trained model
model = joblib.load(MODEL_OUTPUT_PATH)

# Define feature schema (based on your dataset)
class DeliveryFeatures(BaseModel):
    Category_Grocery: bool
    Agent_Rating: float
    Traffic_Low: bool
    Distance_km: float
    Agent_Age: int
    Weather_Sunny: bool
    Weather_Cloudy: bool
    Weather_Fog: bool
    Vehicle_motorcycle: bool
    Traffic_Medium: bool
    Drop_Longitude: float
    Traffic_Jam: bool
    Drop_Latitude: float
    Store_Longitude: float
    Store_Latitude: float
    Order_Hour: int
    Day_Of_Week: int
    Area_Metropolitian: bool
    Pickup_Delay: float
    Area_Semi_Urban: bool
    Weather_Stormy: bool
    Weather_Sandstorms: bool
    Weather_Windy: bool
    Category_Cosmetics: bool

app = FastAPI(title = "Amazon Delivery Time Prediction API")

@app.get("/")
def home():
    return {"message": "Delivery Time Prediction API is running!"}

@app.post("/predict")
def predict(features: DeliveryFeatures):
    # Convert input to numpy array
    data = np.array([[ 
        features.Category_Grocery,
        features.Agent_Rating,
        features.Traffic_Low,
        features.Distance_km,
        features.Agent_Age,
        features.Weather_Sunny,
        features.Weather_Cloudy,
        features.Weather_Fog,
        features.Vehicle_motorcycle,
        features.Traffic_Medium,
        features.Drop_Longitude,
        features.Traffic_Jam,
        features.Drop_Latitude,
        features.Store_Longitude,
        features.Store_Latitude,
        features.Order_Hour,
        features.Day_Of_Week,
        features.Area_Metropolitian,
        features.Pickup_Delay,
        features.Area_Semi_Urban,
        features.Weather_Stormy,
        features.Weather_Sandstorms,
        features.Weather_Windy,
        features.Category_Cosmetics
    ]])

    # Predict
    prediction = model.predict(data)[0]
    return {"predicted_delivery_time": float(prediction)}
def main():
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)

if __name__ == "__main__":
    main()