from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from src.pipeline.pred_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
                age = request.form.get('age'),
                trestbps= request.form.get('trestbps'),
                chol= request.form.get('chol'),
                thalach= request.form.get('thalach'),
                oldpeak= float(request.form.get('oldpeak')),
                cp= request.form.get('cp'),
                slope= request.form.get('slope'),
                thal= request.form.get('thal'),

                sex = request.form.get('sex'),
                fbs = request.form.get('fbs'),
                restecg = request.form.get('restecg'),
                exang = request.form.get('exang'),
                ca = request.form.get('ca')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=str(int(results[0])))
    

if __name__=="__main__":
    app.run(host="0.0.0.0")