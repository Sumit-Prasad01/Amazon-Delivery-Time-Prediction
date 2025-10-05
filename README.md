# Amazon Delivery Time Prediction 🚚

Predict the delivery time for e-commerce orders using machine learning models. This repository contains end-to-end code for data preprocessing, modeling, evaluation, experiment tracking (Mlflow) ,deployment (API) and Streamlit (Frontend).

---

## 📂 Project Structure

```bash
├── api/                  ← Fastapi Api 
├── artifacts/
│ ├── raw/                ← Raw input datasets (CSV)
│ ├── processed/          ← Cleaned & preprocessed data
│ └── models/             ← trained models
├── config/               ← Config Files
├── frontend/             ← Streamlit Frontend
├── notebooks/            ← EDA and Model Experiment Jupyter notebooks
├── pipeline/             ← end to end trainig pipeline script
├── src/                  ← python scripts for data ingestion, preprocessing and training etc.
├── utils/                ← helper functions
├── requirements.txt
└── README.md
```

---
## 🧩 Features Used

Used the following input features for prediction:
```bash
[ 
    Category_Grocery, Agent_Rating, Traffic_Low, Distance_km, Agent_Age,
    Weather_Sunny, Weather_Cloudy, Weather_Fog, Vehicle_motorcycle, Traffic_Medium,
    Drop_Longitude, Traffic_Jam, Drop_Latitude, Store_Longitude, Store_Latitude,
    Order_Hour, Day_Of_Week, Area_Metropolitian, Pickup_Delay, Area_Semi_Urban,
    Weather_Stormy, Weather_Sandstorms, Weather_Windy, Category_Cosmetics
]
```

---

## 🚀 How to Use

### 1. Setup environment

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
### 2. Preprocess & Encode

- Run the preprocessing script to clean and encode features.

- Save the preprocessing pipeline or label encoders (preprocessor.pkl, label_encoders.pkl).

### 3. Train Models

- Use the training.py to train multiple regression models (Linear, Random Forest, LightGBM, etc.).

- Use feature_selection.py to extract top 15 features.

- Compare performance using full features vs top features.

### 4. Hyperparameter Tuning

- Use tuning.py to tune hyperparameters (e.g. for LightGBM) via RandomizedSearchCV or similar.

### 5. Save Artifacts

- After finding the best model, save it as models/best_model.pkl.

- Save preprocessing pipelines / encoders under models/.

### 6. Launch API (FastAPI)

- Navigate into the api/ folder.

- Ensure main.py refers to the correct paths for the model and encoder files.

- Run:
```bash
uvicorn app:app --reload
```
- The API docs will be available at http://127.0.0.1:8000/docs.

- You can send a JSON request with your input features to the /predict endpoint.

### 7. Streamlit App
- Navigate into the frontend/ folder.
- Run:
```bash
Streamit run ui.py
```

## 📈 Model Performance (Example)
```bash
Setting	             Model	     RMSE	 MSE	  R²

All     Features	LightGBM	~21.98	~17.11	~0.82
Top 25  Features	LightGBM	~22.75	~16.89	~0.82

```
- These numbers indicate that using top 15 features gives nearly similar performance with fewer features.

## 📝 Notes & Best Practices

- For categorical variables, you can use LabelEncoder or OneHotEncoder — but be consistent between training and inference.

- Avoid including Order_ID or unique identifier columns as features — they do not generalize.

- If using to_csv, always save with index=False to avoid “Unnamed: 0” columns.

- In your API, handle unseen category values gracefully (e.g. default encoding or fallback).

- Remember to drop or convert raw date/time strings into numeric features (hour, day-of-week) to avoid KeyErrors.

## 🛠️ Future Improvements

- Add real-time features like traffic or weather API integration

- Use more advanced models (e.g. ensemble, neural networks)

- Monitor model drift over time

- Deploy to cloud (AWS, Azure, GCP) or containerize (Docker)
