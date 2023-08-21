import warnings
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

from dataclasses import dataclass

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

models = {"GradientBoostingClassifier": GradientBoostingClassifier()}


def format_data(df):
    # Calculate MSRP column
    df["MSRP"] = (df["PurchasePrice"] / (1 - df["DiscountPct"])).round()

    # Calculate ProductID column
    df["ProductID"] = (
        df["ProductDepartment"]
        + "_"
        + df["ProductCost"].astype(str)
        + "_"
        + df["MSRP"].astype(str)
    )

    # Convert date columns to datetime format
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["CustomerBirthDate"] = pd.to_datetime(df["CustomerBirthDate"])

    # Calculate CustomerAge at the time of OrderDate
    df["CustomerAge"] = (df["OrderDate"] - df["CustomerBirthDate"]).dt.days // 365

    return df


def create_preprocessor():
    # Define categorical and numeric features
    categorical_features = [
        "CustomerState",
        "ProductDepartment",
        "ProductSize",
        "ProductID",
    ]
    numeric_features = [
        "CustomerAge",
        "ProductCost",
        "DiscountPct",
        "PurchasePrice",
        "MSRP",
        "CustomerID_encoded",
    ]

    # Define the age bins
    age_bins = [0, 18, 30, 40, 50, 60, 70, float("inf")]

    # Create the ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("num", "passthrough", numeric_features),
            (
                "age_binning",
                KBinsDiscretizer(n_bins=len(age_bins) - 1, encode="onehot-dense"),
                ["CustomerAge"],
            ),
        ]
    )

    return preprocessor


def eval_metrics(actual, pred):
    roc_auc = roc_auc_score(actual, pred)
    brier_score = brier_score_loss(actual, pred)
    return roc_auc, brier_score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    dataset = "data/train.csv"
    try:
        df = pd.read_csv(dataset, sep=",")
    except Exception as e:
        logger.exception("Unable to read dataset. Error: %s", e)

    df = format_data(df)

    # Create a new column 'CustomerID_encoded' with target encodings set to mean of 'Returned'
    customer_id_mean = df.groupby("CustomerID")["Returned"].mean()
    df["CustomerID_encoded"] = df["CustomerID"].map(customer_id_mean)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(df)

    # The predicted column is "Returned" which is boolean
    X_train = train.drop(["Returned"], axis=1)
    X_test = test.drop(["Returned"], axis=1)
    y_train = train[["Returned"]]
    y_test = test[["Returned"]]

    preprocessor = create_preprocessor()

    for model_name in models:
        print(f"\nTraining model '{model_name}' using '{dataset}'")

        # Create the full pipeline including the classifier
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", models[model_name])]
        )

        with mlflow.start_run():
            # Fit the pipeline on the training data
            pipeline.fit(X_train, y_train)

            # Make probability predictions on the testing data
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            roc_auc, brier_score = eval_metrics(y_test, y_prob)
            print(f"ROC AUC: {roc_auc}")
            print(f"Brier Score: {brier_score}")

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("brier_score", brier_score)

            predictions = pipeline.predict_proba(X_train)[:, 1]
            signature = infer_signature(X_train, predictions)

            mlflow.sklearn.log_model(pipeline, model_name, signature=signature)

    dataset = "data/test.csv"
    try:
        df = pd.read_csv(dataset, sep=",")
    except Exception as e:
        logger.exception("Unable to read dataset. Error: %s", e)
    print(f"\nPredicting returns in '{dataset}' using the trained model")
    df = format_data(df)

    # Apply target encoding to 'CustomerID' and replace NaN values with a default value
    df["CustomerID_encoded"] = df["CustomerID"].map(customer_id_mean)
    df["CustomerID_encoded"].fillna(0.5, inplace=True)

    # Drop target column
    df = df.drop("Returned", axis=1)

    # Make predictions on the test data using the trained pipeline
    y_prob = pipeline.predict_proba(df)[:, 1]

    # Create a DataFrame with 'ID' and 'Prediction' columns
    results_df = pd.DataFrame({"ID": df["ID"], "Prediction": y_prob})

    # Save the results to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    predictions_filename = f"results/predictions_{timestamp}.csv"
    results_df.to_csv(predictions_filename, index=False)

    print(f"Results saved to '{predictions_filename}'")
