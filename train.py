# Load libraries
import importlib
import json
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import process

importlib.reload(process)
importlib.reload(mlflow)

load_dotenv()
TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
EXPERIMENT_NAME = os.environ['MLFLOW_EXPERIMENT_NAME'] + '_PROD'
SEED = int(os.environ['SEED'])
TARGET = os.environ['TARGET']


# Setup MLflow
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(name=EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
mlflow.set_tracking_uri(TRACKING_URI)
print(experiment)


train_data = pd.read_csv('fraudTrain.csv')
y_train = train_data.iloc[:10000][TARGET]
X_train = train_data.iloc[:10000].drop(TARGET, axis=1)

test_data = pd.read_csv('fraudTest.csv')
y_test = test_data[TARGET]
X_test = test_data.drop(TARGET, axis=1)


with open('best_params.json', 'r') as json_file:
    best_params = json.load(json_file)

# log parameters into MLflow
with mlflow.start_run(experiment_id=experiment.experiment_id):

    mlflow.log_params(best_params)

    feature_logic = process.FeatureEngineering()
    added_features = feature_logic.transform(X_train)
    final_x, variables = feature_logic.preprocess(added_features)

    clf = RandomForestClassifier(n_jobs=-1,
                                 class_weight='balanced_subsample',
                                 random_state=SEED,
                                 **best_params
                                 )
    clf.fit(final_x, y_train)

    added_features_test = feature_logic.transform(X_test)
    final_x_test, _ = feature_logic.preprocess(added_features_test)
    y_pred = clf.predict(final_x_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall
        }

    mlflow.log_metrics(metrics)
    print({**metrics, **best_params})

    with open('encoder.pickle', 'wb') as handle:
        pickle.dump(feature_logic.encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact('encoder.pickle')
    with open('scaler.pickle', 'wb') as handle:
        pickle.dump(feature_logic.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact('scaler.pickle')

    input_example = {var: final_x[0][idx] for idx, var in enumerate(variables)}

    mlflow.sklearn.log_model(sk_model=clf,
                             artifact_path='rf_models',
                             registered_model_name='fraud_rf_model',
                             input_example=input_example)
