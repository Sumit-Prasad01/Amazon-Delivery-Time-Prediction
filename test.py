import pandas as pd
import joblib
import numpy as np
from config.paths_config import *


# Load label encoders
encoders = joblib.load(ENCODED_FEATURES_MODEL_PATH)
print("✅ LabelEncoders loaded successfully!")

# Sample input (may include unseen values)
sample = {
    "Category": "Clothing",
    "Agent_Rating": 4.7,
    "Weather": "Fog",
    "Distance_km": 12.5,
    "Agent_Age": 34,
    "Order_Time": "17:55:00",   # unseen possible value   # <----------------  Giving Error
    "Traffic": "Low",
    "Vehicle": "van",
    "Pickup_Time": "11:45:00",   # <----------------  Giving Error
    "Order_Date": "2024-08-21",
    "Drop_Longitude": 75.912471,
    "Drop_Latitude": 22.765049,
    "Store_Longitude": 75.892471,
    "Store_Latitude": 22.745049
}

df_sample = pd.DataFrame([sample])

# print(df_sample)


# # Safe transform function
def safe_label_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        # Handle unseen category
        print(f"⚠️ Unseen category '{value}' detected — assigning fallback class 0.")
        return 0  # or np.nan

# Apply encoders safely
for col, le in encoders.items():
    print(le.classes_)
    df_sample[col] = df_sample[col].apply(lambda x: safe_label_transform(le, x))


# Apply label encoders to categorical columns
# for col, le in encoders.items():
#     df_sample[col] = le.transform(df_sample[col])



print("✅ Encoding successful!")
print(df_sample.head())



# def data_processing(self, df : pd.DataFrame):
#         try:
#             logger.info("Starting our data processing step")
#             logger.info("Dropping the columns")

#             df.drop(columns = ["Order_ID"], inplace = True)
#             df.drop_duplicates(inplace = True)

#             cat_cols = self.config['data_processing']['categorical_columns']
#             num_cols = self.config['data_processing']['numerical_columns']

#             logger.info("Applying LabelEncoding")

#             label_encoder = LabelEncoder()
#             mappings = {}

#             for col in cat_cols:
#                 df[col] = label_encoder.fit_transform(df[col])
#                 mappings[col] = {label : code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

#             logger.info("Label Mappings are : ")

#             for col, mapping in mappings.items():
#                 logger.info(f"{col} : {mapping}")

#             logger.info("Encoding done.")

#             return df

#         except Exception as e:
#             logger.error(f"Error During data preprocessing {e}")
#             raise CustomException("Error while preprocessing data.", e)




































    # def encode_features(self):
    #     try:
    #         logger.info("Loading dataset...")
    #         df = pd.read_csv(self.df_path)

            
    #         desired_columns = [
    #             "Category", "Agent_Rating", "Weather", "Distance_km", "Agent_Age",
    #             "Order_Time", "Traffic", "Vehicle", "Pickup_Time", "Order_Date",
    #             "Drop_Longitude", "Drop_Latitude", "Store_Longitude", "Store_Latitude"
    #         ]
    #         X = df[desired_columns].copy()

            
    #         cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    #         num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    #         logger.info(f"Categorical Columns: {cat_cols}")
    #         logger.info(f"Numeric Columns: {num_cols}")

            
    #         logger.info("Applying Label Encoding...")
    #         encoders = {}
    #         mappings = {}

    #         for col in cat_cols:
    #             le = LabelEncoder()
    #             X[col] = le.fit_transform(X[col])
    #             encoders[col] = le
    #             mappings[col] = {
    #                 label: code for label, code in zip(le.classes_, le.transform(le.classes_))
    #             }

            
    #         # joblib.dump(encoders, "label_encoders.pkl")
    #         # logger.info("Label encoders saved to label_encoders.pkl")

            
    #         encoded_path = "encoded_dataset.csv"
    #         X.to_csv(encoded_path, index=False)
    #         logger.info(f"Encoded dataset saved to {encoded_path}")

    #         logger.info("Encoding complete.")
    #         return encoders

    #     except Exception as e:
    #         logger.error(f"Error while encoding features: {e}")
    #         raise CustomException("Failed to encode features", e)

    
    # def save_model(self, encoder_dict):
        
    #     try:
    #         os.makedirs(os.path.dirname(self.encoder_output_path), exist_ok=True)

    #         logger.info("Saving the fitted LabelEncoders dictionary...")

            
    #         joblib.dump(encoder_dict, self.encoder_output_path)
    #         logger.info(f" LabelEncoders Saved to {self.encoder_output_path}")

    #     except Exception as e:
    #         logger.error(f"Error while saving encoders: {e}")
            
    #         raise CustomException("Failed to Save encoders", e)