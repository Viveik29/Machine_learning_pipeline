import os
import sys
import pandas as pd
import mlflow
import yaml
import dagshub
import pickle
from sklearn.metrics import accuracy_score


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/viveik16693/Machine_learning_pipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="viveik16693"
os.environ["MLFLOW_TRACKING_PASSWORD"]="265a35ffa313931d4d42d67acb23b1b5ddc92b08"

mlflow.set_tracking_uri("https://dagshub.com/viveik16693/Machine_learning_pipeline.mlflow")

def evaluate_model(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    Y = data['Outcome']

    model =pickle.load(open(model_path,'rb'))
    Ypred = model.predict(X)
    accuracy = accuracy_score(Ypred , Y)
    mlflow.log_metric('accuracy',accuracy)
    print("accuracy",accuracy)
    
if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["train"]
    evaluate_model(params['data'],params['model'])
                   
