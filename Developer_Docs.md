# Developer Documentation - Amazon Delivery Time Prediction 🚚

## 1. Project Overview

This project predicts **delivery time for Amazon orders** using machine learning. It leverages historical data containing order, agent, location, and environmental features to estimate the delivery duration in minutes.

The system includes:
- Data preprocessing and encoding (LabelEncoder / OneHotEncoder)
- Exploratory Data Analysis (EDA)
- Feature importance extraction using Random Forest
- Model training with multiple regressors (LightGBM, XGBoost, etc.)
- Hyperparameter tuning using RandomizedSearchCV
- Deployment-ready FastAPI for real-time prediction
- Streamlit UI Frontend
---

## 2. Business Use Case

E-commerce and logistics companies like Amazon rely heavily on **accurate delivery time prediction** to enhance operational efficiency and customer satisfaction.

### Key Business Benefits:
1. **Improved Customer Experience:** Provides realistic delivery estimates, reducing dissatisfaction from delayed orders.
2. **Optimized Resource Allocation:** Helps logistics teams schedule and assign deliveries based on predicted duration and conditions.
3. **Agent Performance Insights:** Identifies key factors affecting delays (agent rating, area, vehicle type, etc.).
4. **Strategic Planning:** Enables better route planning and resource distribution in high-demand regions.

This ML-based system helps automate decision-making and improve logistics performance through data-driven insights.

---

## 3. Architecture Overview

1. **Data Layer**
   - Raw data in CSV format.
   - Cleaned and preprocessed using pandas and sklearn.

2. **Modeling Layer**
   - Label encoding for categorical features.
   - Training multiple models (LightGBM, XGBoost, RandomForest).
   - Evaluation using RMSE, MAE, and R² metrics.

3. **Serving Layer**
   - Best model and encoder saved as `.pkl` files.
   - Real-time predictions via FastAPI REST API.
   - Streamlit Frontend

4. **Deployment**
   - Deployable on **AWS EC2**, **Azure**, or **Docker containers**.

---

## 4. Setup & Environment

### Prerequisites
- Python 3.10+
- pip or conda
- Libraries: `scikit-learn`, `lightgbm`, `xgboost`, `pandas`, `fastapi`, `uvicorn`, `joblib`, `streamlit`

### Installation

```bash
git clone https://github.com/Sumit-Prasad01/Amazon-Delivery-Time-Prediction.git
cd Amazon-Delivery-Time-Prediction
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
```

---

## 5. Model Pipeline

### a. Data Preprocessing
- Handle missing values and irrelevant columns (`Order_ID` dropped).
- Encode categorical variables with `LabelEncoder` or `OneHotEncoder`.
- Extract numeric features (hour, date, etc.) from time fields.

### b. Feature Engineering
- Calculate distance between store and drop location.
- Select top 25 important features using RandomForest.

### c. Model Training
Trained models include:
- Linear Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM (best performer)
- and other regression model as well.

### d. Model Saving
```python
joblib.dump(best_model, "models/lgbm_model.pkl")
```

### e. API Deployment
- Implement FastAPI in `api/app.py`.
- Load model.
- Expose `/predict` endpoint for JSON requests.

### f. Streamlit App
- Implement Streamlit App in `frontend/ui.py`.
- run streamlit app `streamlit run frontend/ui.py`
---

## 6. Example Prediction Request

```json
POST /predict
{
  "Category_Grocery": false,
  "Agent_Rating": 4.8,
  "Traffic_Low": true,
  "Distance_km": 8.942314,
  "Agent_Age": 20,
  "Weather_Sunny": true,
  "Weather_Cloudy": false,
  "Weather_Fog": false,
  "Vehicle_motorcycle": true,
  "Traffic_Medium": true,
  "Drop_Longitude": 77.5946,
  "Traffic_Jam": false,
  "Drop_Latitude": 12.9716,
  "Store_Longitude": 77.5800,
  "Store_Latitude": 12.9600,
  "Order_Hour": 17,
  "Day_Of_Week": 6,
  "Area_Metropolitian": true,
  "Pickup_Delay": 5.0,
  "Area_Semi_Urban": false,
  "Weather_Stormy": false,
  "Weather_Sandstorms": false,
  "Weather_Windy": false,
  "Category_Cosmetics": false
}
```

**Response:**
```json
{
  "Delivery_Time_Predicted": 119.45,
  "Unit": "minutes"
}
```

---

## 7. Model Performance Summary

| Model | RMSE | MAE | R² |
|--------|------|-----|-----|
| LightGBM | 21.98 | 17.11 | 0.816 |
| Random Forest | 22.58 | 17.34 | 0.806 |
| XGBoost | 22.76 | 17.72 | 0.803 |

LightGBM achieved the best accuracy and stability.

---

## 8. Future Improvements

- Integrate live **traffic/weather API** for real-time predictions.
- Use **Optuna** for automatic hyperparameter tuning.
- Add **continuous retraining pipeline** with new data.

---

## 9. Contributors

**Sumit Prasad** — Developer & Machine Learning Engineer  
GitHub: [Sumit-Prasad01](https://github.com/Sumit-Prasad01)

---

© 2025 Sumit Prasad | All Rights Reserved
