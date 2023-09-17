import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # This function is responsible for data transformation
        try:
            # Numerical Pipeline
            num_column = ['age','trestbps','chol','thalach','oldpeak']
            num_pipeline = Pipeline([
                    ('scaling', RobustScaler())
            ])

            #Categorical Pipeline
            cat_column = ['cp','slope','thal']
            cat_pipeline = Pipeline([
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ])

            # Combines both pipelines into Transformer
            preprocessor = ColumnTransformer([
                    ('categorical', cat_pipeline, cat_column),
                    ('numerical', num_pipeline, num_column)],
                    remainder = 'passthrough'
            )

            return preprocessor

            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            preprocessing_obj = self.get_data_transformer_object()  

            target_column_name = 'target'
            num_column = ['age','trestbps','chol','thalach','oldpeak']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # X_train
            target_feature_train_df = train_df[target_column_name]                        # y_train
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)    # X_trest
            target_feature_test_df = test_df[target_column_name]                          # y_test

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            # concatenate arrays
            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)