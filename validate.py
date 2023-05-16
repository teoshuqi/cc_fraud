# Load libraries
import importlib
import json
import os

import mlflow
import random
import mlflow.sklearn
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

import process

importlib.reload(process)
importlib.reload(mlflow)

load_dotenv()
TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
EXPERIMENT_NAME = os.environ['MLFLOW_EXPERIMENT_NAME']
SEED = int(os.environ['SEED'])
TARGET = os.environ['TARGET']


# Setup MLflow
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(name=EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
mlflow.set_tracking_uri(TRACKING_URI)
print(experiment)


train_data = pd.read_csv('fraudTrain.csv')
y_train = train_data.iloc[:20000][TARGET]
X_train = train_data.iloc[:20000].drop(TARGET, axis=1)


# log parameters into MLflow
N = 5
for i in range(N):
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        n_estimator = random.choice(np.arange(50, 300))
        max_depth = random.choice(np.arange(2, 20))
        min_samples_leaf = random.choice(np.arange(2, 200))
        estimators = {
            'n_estimator': n_estimator,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf
        }
        mlflow.log_params(estimators)
        print(n_estimator, max_depth, min_samples_leaf)

        feature_logic = process.FeatureEngineering()
        added_features = feature_logic.transform(X_train)
        final_x, variables = feature_logic.preprocess(added_features)

        clf = RandomForestClassifier(n_jobs=-1,
                                     class_weight='balanced_subsample',
                                     random_state=SEED,
                                     n_estimators=n_estimator,
                                     max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf
                                     )

        cv_results = cross_validate(clf, final_x, y_train, cv=7,
                                    error_score='raise',
                                    scoring=('recall', 'precision', 'f1'),
                                    n_jobs=-1)
        recall = np.mean(cv_results['test_recall'])
        precision = np.mean(cv_results['test_precision'])
        f1 = np.mean(cv_results['test_f1'])
        metrics = {
            'cv_recall': recall,
            'cv_recision': precision,
            'cv_f1': f1
        }
        mlflow.log_metrics(metrics)
        print({**metrics, **estimators})
