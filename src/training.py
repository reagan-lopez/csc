import warnings

import pandas as pd
import numpy as np

import utils
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

from logger import logging

models = {
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(),
        "parameters": {
            "model__learning_rate": [0.1, 0.01, 0.05, 0.001],
            "model__n_estimators": [8, 16, 32, 64, 128, 256],
            "model__subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=10000),
        "parameters": {},
    },
    "AdaBoostClassifier": {
        "model": AdaBoostClassifier(),
        "parameters": {
            "model__learning_rate": [0.1, 0.01, 0.05, 0.001],
            "model__n_estimators": [8, 16, 32, 64, 128, 256],
        },
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(),
        "parameters": {"model__n_estimators": [8, 16, 32, 64, 128, 256]},
    },
}


def eval_models(model_scores):
    best_model = ""
    max_roc_auc = 0
    min_brier_score = 1
    for name, scores in model_scores.items():
        if scores[0] >= max_roc_auc and scores[1] <= min_brier_score:
            best_model = name
            max_roc_auc = scores[0]
            min_brier_score = scores[1]

    if best_model:
        print(f"\n***************************************")
        print(f"Best model for the dataset is:")
        print(f"{best_model}")
        print(f"  ROC AUC: {max_roc_auc}")
        print(f"  Brier Score: {min_brier_score}")
        print(f"***************************************")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    dataset = "artifacts/training_data.csv"
    try:
        df = pd.read_csv(dataset, sep=",")
    except Exception as e:
        logging.exception("Unable to read dataset. Error: %s", e)

    utils.format_data(df)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(df)

    # The predicted column is "Returned" which is boolean
    X_train = train.drop(["Returned"], axis=1)
    X_test = test.drop(["Returned"], axis=1)
    y_train = train["Returned"].ravel()
    y_test = test["Returned"].ravel()

    preprocessor = utils.create_preprocessor()

    model_scores = {}
    for model_name, values in models.items():
        print(f"\nEvaluating model: {model_name}")
        model, params = values["model"], values["parameters"]

        if params:
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            gs = GridSearchCV(pipeline, params, cv=3)
            gs.fit(X_train, y_train)
            pipeline.set_params(**gs.best_params_)

        # Create the full pipeline including the classifier
        # pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        with mlflow.start_run():
            # Fit the pipeline on the training data
            pipeline.fit(X_train, y_train)

            # Make probability predictions on the testing data
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            roc_auc, brier_score = utils.eval_metrics(y_test, y_prob)
            model_scores[model_name] = [roc_auc, brier_score]
            print(f"  ROC AUC: {roc_auc}")
            print(f"  Brier Score: {brier_score}")

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("brier_score", brier_score)

            predictions = pipeline.predict_proba(X_train)[:, 1]
            signature = infer_signature(X_train, predictions)

            mlflow.sklearn.log_model(pipeline, model_name, signature=signature)

    eval_models(model_scores)
