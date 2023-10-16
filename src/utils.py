import os
import dill
from src.logger import logging
import pandas as pd


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

    return df
