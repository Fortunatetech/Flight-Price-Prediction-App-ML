import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataCleaningConfig:
    train_data_path_cleaned: str=os.path.join('artifacts',"train_cleaned.csv")
    test_data_path_cleaned: str=os.path.join('artifacts',"test_cleaned.csv")

class DataCleaning:
    def __init__(self):
        self.cleaning_config=DataCleaningConfig()
    
    def initiate_data_cleaning(self):
        logging.info("Entered the data cleaning method or component")
        try:
            df_train=pd.read_csv("artifacts/train.csv")
            df_test=pd.read_csv("artifacts/test.csv")
        
            logging.info('Read the dataset as a pandas dataframe')
                # Extract day, month, and weekday from 'Date_of_Journey'
            df_train['Journey_Day'] = pd.to_datetime(df_train['Date_of_Journey'], format='%d/%m/%Y').dt.day.astype(int)
            df_train['Journey_Month'] = pd.to_datetime(df_train['Date_of_Journey'], format='%d/%m/%Y').dt.month.astype(int)
            df_train['Journey_Weekday'] = pd.to_datetime(df_train['Date_of_Journey'], format='%d/%m/%Y').dt.weekday.astype(int)
            logging.info('processing the training dataset as dataframe')
                # Calculate the duration in minutes from 'Duration'
            def duration_to_minutes(duration):
                    duration_list = duration.split()
                    if len(duration_list) == 2:
                        return int(duration_list[0][:-1]) * 60 + int(duration_list[1][:-1])
                    elif 'h' in duration:
                        return int(duration_list[0][:-1]) * 60
                    else:
                        return int(duration_list[0][:-1])

            df_train['Duration_minutes'] = df_train['Duration'].apply(duration_to_minutes).astype(int)

                # Convert 'Dep_Time' and 'Arrival_Time' to datetime format and extract hour and minute
            df_train['Dep_Time'] = pd.to_datetime(df_train['Dep_Time'])
            df_train['Dep_Hour'] = df_train['Dep_Time'].dt.hour.astype(int)
            df_train['Dep_Minute'] = df_train['Dep_Time'].dt.minute.astype(int)

            df_train['Arrival_Time'] = pd.to_datetime(df_train['Arrival_Time'])
            df_train['Arrival_Hour'] = df_train['Arrival_Time'].dt.hour.astype(int)
            df_train['Arrival_Minute'] = df_train['Arrival_Time'].dt.minute.astype(int)
            df_train.dropna(inplace=True)
                # Drop the following columns 'Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration'
            df_train= df_train.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route'], axis = 1)
            

            # Extract day, month, and weekday from 'Date_of_Journey'
            df_test['Journey_Day'] = pd.to_datetime(df_test['Date_of_Journey'], format='%d/%m/%Y').dt.day.astype(int)
            df_test['Journey_Month'] = pd.to_datetime(df_test['Date_of_Journey'], format='%d/%m/%Y').dt.month.astype(int)
            df_test['Journey_Weekday'] = pd.to_datetime(df_test['Date_of_Journey'], format='%d/%m/%Y').dt.weekday.astype(int)

                # Calculate the duration in minutes from 'Duration'
            def duration_to_minutes(duration):
                duration_list = duration.split()
                if len(duration_list) == 2:
                    return int(duration_list[0][:-1]) * 60 + int(duration_list[1][:-1])
                elif 'h' in duration:
                    return int(duration_list[0][:-1]) * 60
                else:
                    return int(duration_list[0][:-1])

            df_test['Duration_minutes'] = df_test['Duration'].apply(duration_to_minutes).astype(int)

            # Convert 'Dep_Time' and 'Arrival_Time' to datetime format and extract hour and minute
            df_test['Dep_Time'] = pd.to_datetime(df_test['Dep_Time'])
            df_test['Dep_Hour'] = df_test['Dep_Time'].dt.hour.astype(int)
            df_test['Dep_Minute'] = df_test['Dep_Time'].dt.minute.astype(int)

            df_test['Arrival_Time'] = pd.to_datetime(df_test['Arrival_Time'])
            df_test['Arrival_Hour'] = df_test['Arrival_Time'].dt.hour.astype(int)
            df_test['Arrival_Minute'] = df_test['Arrival_Time'].dt.minute.astype(int)
            df_test.dropna(inplace=True)
                # Drop the following columns 'Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration'
            df_test = df_test.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route'], axis = 1)
                

            
            logging.info('ended he loop') 
        
            logging.info('returning done') 

            os.makedirs(os.path.dirname(self.cleaning_config.train_data_path_cleaned),exist_ok=True)
            logging.info('directory created')


            df_train.to_csv(self.cleaning_config.train_data_path_cleaned,index=False,header=True)
            logging.info('Preprocessed Train data saved')
            df_test.to_csv(self.cleaning_config.test_data_path_cleaned,index=False,header=True)
            logging.info('Preprocessed test data saved')

            logging.info("Train test data cleaned")
            return df_train, df_test
            logging.info("returned df_train and df_test")       
      
        except Exception as e:
            raise CustomException(e,sys)
    





            

            

        
           


