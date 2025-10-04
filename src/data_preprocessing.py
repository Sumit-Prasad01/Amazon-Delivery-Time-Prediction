import os
import numpy as np
import pandas as pd

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


logger = get_logger(__name__)


class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def data_processing(self, df: pd.DataFrame):
        try:
            logger.info("Starting our data processing step with One-Hot Encoding")
            logger.info("Dropping the columns")

            
            df = df.copy()

            df.drop(columns=["Order_ID"], inplace=True)
            df.drop_duplicates(inplace=True)

            
            cat_cols = self.config['data_processing']['categorical_columns']
            
            num_cols = self.config['data_processing']['numerical_columns'] 

            logger.info("Applying One-Hot Encoding (using pd.get_dummies)")

            
            df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

            
            logger.info("One-Hot Encoding complete.")
            logger.info(f"DataFrame shape after encoding: {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Error During data preprocessing {e}")
            raise CustomException("Error while preprocessing data.", e)
        
    

    def select_features(self, df : pd.DataFrame):
        try:
            logger.info("Starting Our Feature Selection")

            X = df.drop(columns = 'Delivery_Time')
            y = df['Delivery_Time']

            model = RandomForestRegressor(random_state = 42)
            model.fit(X,y)

            feature_importance = model.feature_importances_
            features_importance_df = pd.DataFrame({
                'features' : X.columns,
                'importance' : feature_importance
            })

            top_important_features_df = features_importance_df.sort_values(by = 'importance', ascending = False)
            num_features_to_select = self.config['data_processing']["no_of_features"]

            top_25_features = top_important_features_df['features'].head(num_features_to_select).values

            logger.info(f"Features Selected : {top_25_features}")

            top_15_df = df[top_25_features.tolist() + ['Delivery_Time']]

            logger.info("Feature Selection Completed Successfully.")

            return top_15_df
        
        except Exception as e:
            logger.error(f"Error During feature selection {e}")
            raise CustomException("Error while feature selection.", e)
        
    
    def save_data(self, df: pd.DataFrame, file_path: str):
        try:
            logger.info("Saving data into processed folder")

            # Drop any unnamed index-like columns before saving
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            df.to_csv(file_path, index=False)

            logger.info(f"Data Saved Successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error During Saving Data {e}")
            raise CustomException("Error while Saving Data.", e)
        

    
    def process(self):
        try:
            logger.info("Loading the data from raw dir")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.data_processing(train_df)
            test_df = self.data_processing(test_df)

            train_df = self.select_features(train_df)

            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data Processing completed successfully.")

        
        except Exception as e:
            logger.error(f"Error During processing data pipeline {e}")
            raise CustomException("Error while Processing Data Pipeline.", e)
    

if __name__ == "__main__":

    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()