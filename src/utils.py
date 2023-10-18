from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from logger import logging
import os
import dill
from sklearn.metrics import roc_auc_score, brier_score_loss


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info("Unable to save object. Error: %s", e)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info("Unable to load object. Error: %s", e)


def format_data(df):
    # Calculate MSRP column
    df["MSRP"] = (df["PurchasePrice"] / (1 - df["DiscountPct"])).round()

    # Calculate ProductID column
    df["ProductID"] = (
        df["ProductDepartment"]
        + "-"
        + df["ProductCost"].astype(str)
        + "-"
        + df["MSRP"].astype(str)
    )

    # Convert date columns to datetime format
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["CustomerBirthDate"] = pd.to_datetime(df["CustomerBirthDate"])

    # Calculate CustomerAge at the time of OrderDate
    df["CustomerAge"] = (df["OrderDate"] - df["CustomerBirthDate"]).dt.days // 365


def create_preprocessor():
    # Define categorical and numeric features
    categorical_features = [
        "CustomerState",
        "ProductDepartment",
        "ProductSize",
        "ProductID",
    ]
    numerical_features = [
        "CustomerAge",
        "ProductCost",
        "DiscountPct",
        "PurchasePrice",
        "MSRP",
    ]

    # Create transformers for each feature type
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    # Create a ColumnTransformer to apply the transformers to the appropriate feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def eval_metrics(actual, pred):
    roc_auc = roc_auc_score(actual, pred)
    brier_score = brier_score_loss(actual, pred)
    return roc_auc, brier_score
