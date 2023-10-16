import os
import pandas as pd
from logger import logging
from utils import load_object, format_data
from datetime import datetime


prediction_data_path = os.path.join("artifacts", "prediction_data.csv")


class PredictionPipeline:
    def __init__(self, df):
        self.df = df

    def predict(self):
        try:
            preprocessor_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            transformed_data = preprocessor.transform(self.df)
            pred = model.predict_proba(transformed_data)[:, 1]
            return pred
        except Exception as e:
            logging.info("Exception occured during prediction. Error: %s", e)


if __name__ == "__main__":
    df = pd.read_csv(prediction_data_path)
    df = format_data(df)
    df = df.drop(["Returned"], axis=1)

    pred_pipeline = PredictionPipeline(df)
    pred = pred_pipeline.predict()

    results_df = pred({"ID": df["ID"], "Prediction": pred})

    # Save the results to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    predictions_filename = f"results/predictions_{timestamp}.csv"
    results_df.to_csv(predictions_filename, index=False)

    print(f"Results saved to '{predictions_filename}'")
