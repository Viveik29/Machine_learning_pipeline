import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from mlflow.models import infer_signature 
import dagshub
import yaml
from sklearn.metrics import accuracy_score , confusion_matrix,classification_report
from urllib.parse import urlparse 
import mlflow
import pickle

# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/viveik16693/Machine_learning_pipeline.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"]="dvc remote modify origin --local user viveik16693"
# os.environ["MLFLOW_TRACKING_PASSWORD"]="dvc remote modify origin --local password 265a35ffa313931d4d42d67acb23b1b5ddc92b08"
import dagshub

def hyperparameter_tunning(xtrain,ytrain,param_grid):
    rf = RandomForestClassifier()
    gridsearch = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,verbose=2,n_jobs=-1)
    gridsearch.fit(xtrain,ytrain)
    return gridsearch





def model_trainning(input_path,model_path,random_state,n_estimaters,max_depth):
    data=pd.read_csv(input_path)
    
    x = data.drop('Outcome',axis=1)
    print(x.shape)
    y = data['Outcome']
    print(y.shape)
    

    #mlflow logging info
    dagshub.init(repo_owner='viveik16693', repo_name='Machine_learning_pipeline', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/viveik16693/Machine_learning_pipeline.mlflow")
    #start mlflow run
    with mlflow.start_run():
     
        xtrain,xtest,ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=42)
        print(f"xtrain.shape:{xtrain.shape} and ytrain.shape: {ytrain.shape}")
        signature=infer_signature(xtrain,ytrain)

        #hyperparameter grid
        param_grid = {
            'n_estimators': [20],
            'max_depth': [2],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }

    gridsearch = hyperparameter_tunning(xtrain,ytrain,param_grid)
    best_model = gridsearch.best_estimator_
    ypred = best_model.predict(xtest)
    accuracy = accuracy_score(ypred,ytest)
    print(f"Accuracy:{accuracy}") 
    #loging started
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_param("best_n_estimatios",gridsearch.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", gridsearch.best_params_['max_depth'])
    mlflow.log_param("best_sample_split", gridsearch.best_params_['min_samples_split'])
    mlflow.log_param("best_samples_leaf", gridsearch.best_params_['min_samples_leaf'])

    ## log the confusion matrix and classification report

    cm=confusion_matrix(ytest,ypred )
    cr=classification_report(ytest,ypred)

    mlflow.log_text(str(cm),"confusion_matrix.txt")
    mlflow.log_text(cr,"classification_report.txt")

    tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store!='file':
        mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model")
    else:
        mlflow.sklearn.log_model(best_model, "model",signature=signature)

    ## create the directory to save the model
    os.makedirs(os.path.dirname(model_path),exist_ok=True)

    filename=model_path
    pickle.dump(best_model,open(filename,'wb'))

    print(f"Model saved to {model_path}")


if __name__ =="__main__":
    params= yaml.safe_load(open("params.yaml"))['train']
    model_trainning(params['data'],params['model'],params['random_state'],params['n_estimaters'],params['max_depth'])


    

        
        


