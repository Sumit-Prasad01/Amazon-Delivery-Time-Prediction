
import streamlit as st
import requests
from config.paths_config import *
import os 
from dotenv import load_dotenv
load_dotenv()

# FastAPI endpoint
API_URL = os.getenv("API_URL")

st.set_page_config(page_title="Delivery Time Predictor", page_icon="📦", layout="centered")

st.title("🚚 Delivery Time Prediction App")
st.markdown("Enter the delivery details below to predict the estimated delivery time.")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    Category_Grocery = st.checkbox("Category: Grocery")
    Category_Cosmetics = st.checkbox("Category: Others")
    Agent_Rating = st.number_input("Agent Rating", min_value=0.0, max_value=5.0, value=4.5)
    Distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0)
    Agent_Age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
    Pickup_Delay = st.number_input("Pickup Delay (min)", min_value=0.0, value=10.0)
    Order_Hour = st.number_input("Order Hour (0-23)", min_value=0, max_value=23, value=12)
    Day_Of_Week = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=3)

with col2:
    Traffic_Low = st.checkbox("Traffic: Low")
    Traffic_Medium = st.checkbox("Traffic: Medium")
    Traffic_Jam = st.checkbox("Traffic: Jam")

    Weather_Sunny = st.checkbox("Sunny")
    Weather_Cloudy = st.checkbox("Cloudy")
    Weather_Fog = st.checkbox("Fog")
    Weather_Stormy = st.checkbox("Stormy")
    Weather_Sandstorms = st.checkbox("Sandstorms")
    Weather_Windy = st.checkbox("Windy")

    Vehicle_motorcycle = st.checkbox("Vehicle: Motorcycle")
    Area_Metropolitian = st.checkbox("Area: Metropolitan")
    Area_Semi_Urban = st.checkbox("Area: Semi-Urban")

Drop_Longitude = st.number_input("Drop Longitude", value=77.5946)
Drop_Latitude = st.number_input("Drop Latitude", value=12.9716)
Store_Longitude = st.number_input("Store Longitude", value=77.5800)
Store_Latitude = st.number_input("Store Latitude", value=12.9600)

# --- Predict button ---
if st.button("🔮 Predict Delivery Time"):
    # Prepare input
    input_data = {
        "Category_Grocery": Category_Grocery,
        "Agent_Rating": Agent_Rating,
        "Traffic_Low": Traffic_Low,
        "Distance_km": Distance_km,
        "Agent_Age": Agent_Age,
        "Weather_Sunny": Weather_Sunny,
        "Weather_Cloudy": Weather_Cloudy,
        "Weather_Fog": Weather_Fog,
        "Vehicle_motorcycle": Vehicle_motorcycle,
        "Traffic_Medium": Traffic_Medium,
        "Drop_Longitude": Drop_Longitude,
        "Traffic_Jam": Traffic_Jam,
        "Drop_Latitude": Drop_Latitude,
        "Store_Longitude": Store_Longitude,
        "Store_Latitude": Store_Latitude,
        "Order_Hour": Order_Hour,
        "Day_Of_Week": Day_Of_Week,
        "Area_Metropolitian": Area_Metropolitian,
        "Pickup_Delay": Pickup_Delay,
        "Area_Semi_Urban": Area_Semi_Urban,
        "Weather_Stormy": Weather_Stormy,
        "Weather_Sandstorms": Weather_Sandstorms,
        "Weather_Windy": Weather_Windy,
        "Category_Cosmetics": Category_Cosmetics
    }

    # Call API
    with st.spinner("Predicting..."):
        response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        prediction = response.json()["predicted_delivery_time"]
        st.success(f"🕒 Predicted Delivery Time: **{prediction:.2f} minutes**")
    else:
        st.error("⚠️ Error: Unable to get prediction. Check API connection.")
