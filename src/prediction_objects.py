from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import os
import warnings

from logger import logging
import utils

model_name, model = "RandomForestClassifier", RandomForestClassifier()

training_data_path = os.path.join("artifacts", "training_data.csv")
prediction_data_path = os.path.join("artifacts", "prediction_data.csv")
model_path = os.path.join("artifacts", "model.pkl")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    logging.info("Start training..")
    try:
        df = pd.read_csv(training_data_path, sep=",")
    except Exception as e:
        logging.info("Unable to read dataset. Error: %s", e)

    utils.format_data(df)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(df)

    # The predicted column is "Returned" which is boolean
    X_train = train.drop(["Returned"], axis=1)
    X_test = test.drop(["Returned"], axis=1)
    y_train = train["Returned"].ravel()
    y_test = test["Returned"].ravel()

    preprocessor = utils.create_preprocessor()

    print(f"\nTraining model '{model_name}' using '{training_data_path}'")

    # Create the full pipeline including the classifier
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    with mlflow.start_run():
        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)
        utils.save_object(file_path=model_path, obj=pipeline)

        # Make probability predictions on the testing data
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        roc_auc, brier_score = utils.eval_metrics(y_test, y_prob)
        print(f"ROC AUC: {roc_auc}")
        print(f"Brier Score: {brier_score}")

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("brier_score", brier_score)

        predictions = pipeline.predict_proba(X_train)[:, 1]
        signature = infer_signature(X_train, predictions)

        mlflow.sklearn.log_model(pipeline, model_name, signature=signature)

    logging.info("Start prediction..")
    try:
        df = pd.read_csv(prediction_data_path, sep=",")
    except Exception as e:
        logging.info("Unable to read dataset. Error: %s", e)
    print(f"\nPredicting returns in '{prediction_data_path}' using the trained model")

    utils.format_data(df)

    # Drop target column
    df = df.drop("Returned", axis=1)

    pipeline = utils.load_object(model_path)
    y_prob = pipeline.predict_proba(df)[:, 1]

    # Create a DataFrame with 'ID' and 'Prediction' columns
    results_df = pd.DataFrame({"ID": df["ID"], "Prediction": y_prob})

    # Save the results to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    predictions_filename = f"results/predictions_{timestamp}.csv"
    results_df.to_csv(predictions_filename, index=False)

    logging.info(f"Results saved to '{predictions_filename}'")
    print(f"Results saved to '{predictions_filename}'")
