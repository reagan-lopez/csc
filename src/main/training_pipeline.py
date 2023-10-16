import os
import pandas as pd
from src.utils import format_data
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

training_data_path = os.path.join("artifacts", "training_data.csv")

if __name__ == "__main__":
    df = pd.read_csv(training_data_path)
    df = format_data(df)

    data_transformation = DataTransformation(df)
    (
        X_train,
        y_train,
        X_test,
        y_test,
        _,
    ) = data_transformation.initate_data_transformation()

    model_training = ModelTraining(X_train, y_train, X_test, y_test)
    model_training.initiate_model_training()
