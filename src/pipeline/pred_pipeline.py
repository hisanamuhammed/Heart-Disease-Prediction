import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 age: int,
                 trestbps: int,
                 chol: int,
                 thalach: int,
                 oldpeak: float,
                 cp: str,
                 slope: str,
                 thal: str, 
                 sex: bool,
                 fbs: bool,
                 restecg: int,
                 exang: bool,
                 ca: int):
        self.age = age
        self.trestbps = trestbps
        self.chol = chol
        self.thalach = thalach
        self.oldpeak = oldpeak
        self.cp = cp
        self.slope = slope
        self.thal = thal
        self.sex = sex
        self.fbs = fbs
        self.restecg = restecg
        self.exang = exang
        self.ca = ca

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "trestbps" : [self.trestbps],
                "chol" : [self.chol],
                "thalach" : [self.thalach],
                "oldpeak" : [self.oldpeak],
                "cp" : [self.cp],
                "slope" : [self.slope],
                "thal" : [self.thal],
                "sex" : [self.sex],
                "fbs" : [self.fbs],
                "restecg" : [self.restecg],
                "exang" : [self.exang],
                "ca" : [self.ca]
            }
            return pd.DataFrame(custom_data_input_dict  )

        except Exception as e:
            raise CustomException(e,sys)