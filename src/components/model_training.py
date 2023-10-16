from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

from src.logger import logging

from dataclasses import dataclass
import os


@dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")


class ModelTraining:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.mtc = ModelTrainingConfig()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def eval_metrics(self, actual, pred):
        roc_auc = roc_auc_score(actual, pred)
        brier_score = brier_score_loss(actual, pred)
        return roc_auc, brier_score

    def eval_models(self, model_scores):
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

    def initiate_model_training(self):
        try:
            models = {
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "LogisticRegression": LogisticRegression(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
            }

            model_scores = {}
            for model_name in models:
                print(f"\nEvaluating model: {model_name}")
                model = models[model_name]

                with mlflow.start_run():
                    # Fit the pipeline on the training data
                    model.fit(self.X_train, self.y_train)

                    # Make probability predictions on the testing data
                    y_prob = model.predict_proba(self.X_test)[:, 1]

                    roc_auc, brier_score = self.eval_metrics(self.y_test, y_prob)
                    model_scores[model_name] = [roc_auc, brier_score]
                    print(f"  ROC AUC: {roc_auc}")
                    print(f"  Brier Score: {brier_score}")

                    mlflow.log_metric("roc_auc", roc_auc)
                    mlflow.log_metric("brier_score", brier_score)

                    predictions = model.predict_proba(self.X_train)[:, 1]
                    signature = infer_signature(self.X_train, predictions)

                    mlflow.sklearn.log_model(model, model_name, signature=signature)

            self.eval_models(model_scores)

        except Exception as e:
            logging.info("Exception occured during model training. Error: %s", e)
