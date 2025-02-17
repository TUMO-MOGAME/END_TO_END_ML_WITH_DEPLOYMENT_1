import os, sys
from src.exception import ProjectException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# in injestion wheever we perofming data injection component  thers should be some import like wrer you have to same raw data train data and test data and we save this data here
# in calss data ingestion config 
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')



class DataIngestion:
    def __init__(self):
        # Initialize the ingestion configuration
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        # This method will read data from a CSV file, split it into training and testing datasets, and save them
        logging.info("Entered the data ingestion method/component.")
        
        try:
            # Read the dataset into a dataframe from the provided file path
            df = pd.read_csv('notebook/data/data.csv')
            logging.info('Dataset read into DataFrame.')

            # Create the necessary directories if they don't already exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw dataset saved.')

            # Split the dataset into training and testing sets (80/20 split)
            logging.info("Train-test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test datasets to their respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is complete.")
            
            # Return the paths to the training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            # If an error occurs, raise a custom exception
            raise ProjectException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()