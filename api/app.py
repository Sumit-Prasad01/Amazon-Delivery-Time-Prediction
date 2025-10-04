import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from config.paths_config import MODEL_OUTPUT_PATH


# Define Input Schema

class DeliveryInput(BaseModel):
    Category : str
    Agent_Rating : float
    Weather : str
    Distance_km : float
    Agent_Age : int
    Order_Time : str
    Traffic : str
    Vehicle : str
    Pickup_Time : str
    Order_Date : str
    Drop_Longitude : float
    Drop_Latitude: float
    Store_Longitude: float
    Store_Latitude: float


# Initialize App and Load Model

app = FastAPI(title = "Amazon Delivery Time Prediction API")

# Load model 

model = joblib.load(MODEL_OUTPUT_PATH)
