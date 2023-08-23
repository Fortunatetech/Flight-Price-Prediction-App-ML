import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.model_selection import train_test_split
#from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):

        '''
        This Function is responsible for data Transformation which include:
        1. Feature Encoding using one-hot-encoding Technique
        2. Standard Scaling for Numeric Features

        '''

        try:
            # Define lists of categorical and numerical features
            categorical_features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info']
            numerical_features = ['Journey_Day', 'Journey_Month', 'Journey_Weekday', 
                                  'Duration_minutes',
                                   'Dep_Hour', 'Dep_Minute',
                                   'Arrival_Hour', 'Arrival_Minute'
                                   ]


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            preprocessor=ColumnTransformer(
                [
                ("cat_pipelines",cat_pipeline, categorical_features),
                ("num_pipeline",num_pipeline, numerical_features)
                
                ]            
                
                )

           
            return preprocessor
            logging.info("Categorical Columns Encoding Completed")
            logging.info("Numerical Columns Standard Scaling Completed")
        
        except Exception as e:
            raise CustomException(e,sys) 
       
    def inititate_data_transformation(self):
        try:

            train_df=pd.read_csv("artifacts/train_cleaned.csv")
            test_df=pd.read_csv("artifacts/test_cleaned.csv")
            #train_set,test_set=train_test_split(cleaned_data,test_size=0.2,random_state=42)         
        
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation()
            target_column_name ="Price"
            numerical_columns = ["Journey_day", "Journey_month","Journey_year","hours","minutes","Arrival_hour","Arrival_min","duration_mins","duration_hours","Total_Stops"]
            input_feature_train_df= train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Saved preprocessing object.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                         )

            return (
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            
            )

        except Exception as e:
            raise CustomException(e,sys)





    

    





    








    

        
