import joblib
import pandas as pd
import json

def init():
    global model
    model = joblib.load('churn_model_pipeline.pkl')

def run(raw_data):
    data = pd.DataFrame(json.loads(raw_data))
    predictions = model.predict_proba(data)[:, 1]  # churn probability
    return predictions.tolist()
