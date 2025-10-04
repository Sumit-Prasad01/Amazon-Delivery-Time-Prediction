import pandas as pd
import numpy as np
import joblib 
from config.paths_config import *
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

class EncoderFeatures:

    def __init__(self, file_path, encoder_output_path):
        self.df_path = file_path
        self.encoder_output_path = encoder_output_path




    def encode_features(self):
        try:
            logger.info("Starting feature encoding with OneHotEncoder...")
            df = pd.read_csv(self.df_path)

            desired_columns = [
                "Category", "Agent_Rating", "Weather", "Distance_km", "Agent_Age",
                "Order_Time", "Traffic", "Vehicle", "Pickup_Time", "Order_Date",
                "Drop_Longitude", "Drop_Latitude", "Store_Longitude", "Store_Latitude"
            ]
            X = df[desired_columns].copy()

            cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

            logger.info(f"Categorical Columns for OHE: {cat_cols}")
            logger.info(f"Numeric Columns (left untouched by OHE): {num_cols}")


            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "one_hot_encode",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        cat_cols,
                    )
                ],
                
                remainder="passthrough",
                
                verbose_feature_names_out=False
            ).set_output(transform="pandas") 

            logger.info("Fitting and transforming data with ColumnTransformer...")
            
            
            X_encoded = preprocessor.fit_transform(X)

           
            logger.info("Encoding complete.")
            
 
            encoded_path = "encoded_dataset.csv"
            X_encoded.to_csv(encoded_path, index=False)
            logger.info(f"Encoded dataset saved to {encoded_path}")
            
            
            return preprocessor

        except Exception as e:
            logger.error(f"Error while encoding features: {e}")
            raise CustomException("Failed to encode features", e)

    

    def save_model(self, preprocessor):
        
        try:
            os.makedirs(os.path.dirname(self.encoder_output_path), exist_ok=True)

            
            logger.info("Saving the fitted ColumnTransformer preprocessor...")

            
            joblib.dump(preprocessor, self.encoder_output_path)
            logger.info(f" Preprocessor Saved to {self.encoder_output_path}")

        except Exception as e:
            logger.error(f"Error while saving preprocessor: {e}")
            
            
            raise CustomException("Failed to Save preprocessor", e)

        
    
    def run(self):
        try:
            logger.info("Starting Feature Encoding")
            encoded_features = self.encode_features()
            self.save_model(encoded_features)
        
        except Exception as e:
            logger.error(f"Error in Encoding pipeline {e}")
            raise CustomException("Failed to run encoding pipeline.",e)



if __name__ == "__main__":

    encoder = EncoderFeatures(RAW_FILE_PATH, ENCODED_FEATURES_MODEL_PATH)
    encoder.run()
