import mlflow
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    dataset = "data/test.csv"
    try:
        df = pd.read_csv(dataset, sep=",")
    except Exception as e:
        logging.info("Unable to read dataset. Error: %s", e)
    print(f"\nPredicting returns in '{dataset}' using the trained model")
    df = format_data(df)
    df = df.drop("Returned", axis=1)  # Drop target column

    logged_model = "runs:/2569045c934b4ea592b85ccf4b567d43/RandomForestClassifier"
    loaded_model = mlflow.sklearn.load_model(logged_model)

    y_prob = loaded_model.predict_proba(df)[:, 1]
    results_df = pd.DataFrame({"ID": df["ID"], "Prediction": y_prob})

    # Save the results to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    predictions_filename = f"results/predictions_{timestamp}.csv"
    results_df.to_csv(predictions_filename, index=False)

    print(f"Results saved to '{predictions_filename}'")
