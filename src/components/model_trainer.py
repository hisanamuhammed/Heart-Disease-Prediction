import os
import sys
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import( 
    RandomForestClassifier,
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier )

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,x_test,y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                    "Gaussian NB": GaussianNB(),

                    "Random Forest": RandomForestClassifier(random_state=99, n_jobs=-1),
            }

            # Define the parameters for each model
            params = {
                "Gaussian NB": {
                    'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-6]
                },
                
                "Random Forest": {
                    'max_depth': [1, 2, 3, 4, 5]                
                }      
            }
            

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, x_test=x_test,
                                                y_test=y_test, models=models, param=params)
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]   
            best_model = models[best_model_name]  
            print(best_model)       

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Found best model on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)

            accuracy = accuracy_score(y_test, predicted)

            return accuracy


        except Exception as e:
            raise CustomException(e,sys)