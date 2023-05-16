# Load libraries
import importlib
import json
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import process

importlib.reload(process)

from dotenv import load_dotenv

importlib.reload(mlflow)

load_dotenv()
TRACKING_URI = os.environ['MLFLOW_TRACKING_URI'] 
EXPERIMENT_NAME = os.environ['MLFLOW_EXPERIMENT_NAME'] +'_PROD'
SEED = int(os.environ['SEED'])
TARGET = 'is_fraud'


# Setup MLflow
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(name=EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
mlflow.set_tracking_uri(TRACKING_URI)
print(experiment)


test_data = pd.read_csv('fraudTest.csv')
y_test = test_data[TARGET]
X_test = test_data.drop(TARGET, axis=1)


logged_model = 'runs:/22915a34f27b47f2af7169c1698339a2/rf_models'
encoder_artifact = 'mlartifacts/2/22915a34f27b47f2af7169c1698339a2/artifacts/encoder.pickle'
scaler_artifact = 'mlartifacts/2/22915a34f27b47f2af7169c1698339a2/artifacts/encoder.pickle'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)



# log parameters into MLflow
with mlflow.start_run(experiment_id=experiment.experiment_id):

    with open(encoder_artifact, "rb") as f:
        encoder = pickle.load(f)
    with open(scaler_artifact, "rb") as f:
        scaler = pickle.load(f)
    feature_logic = process.FeatureEngineering(encoder=encoder, scaler=scaler)
    added_features_test = feature_logic.transform(X_test)
    final_x_test, variables = feature_logic.preprocess(added_features_test)
    y_pred = loaded_model.predict(final_x_test)

    # get predictions
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # log parameters
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall
        }

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(sk_model=loaded_model)

