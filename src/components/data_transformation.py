from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, df):
        self.dtc = DataTransformationConfig()
        self.df = df

    def create_preprocessor(self):
        """
        This function is responsible for data transformation
        """
        try:
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
            ]

            # Create the ColumnTransformer for preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(), categorical_features),
                    ("num", "passthrough", numeric_features),
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Exception occured during data transformation. Error: %s", e)

    def initate_data_transformation(self):
        try:
            preprocessor = self.create_preprocessor()

            save_object(
                file_path=self.dtc.preprocessor_path,
                obj=preprocessor,
            )
            logging.info("Preprocessor pickle file saved.")

            train, test = train_test_split(self.df)
            logging.info("Split into train and test.")

            X_train = train.drop(["Returned"], axis=1)
            X_test = test.drop(["Returned"], axis=1)
            y_train = train[["Returned"]]
            y_test = test[["Returned"]]

            logging.info("Created X_train and X_test.")

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
            logging.info("Transformed train and test.")

            return (
                X_train,
                y_train,
                X_test,
                y_test,
                self.dtc.preprocessor_path,
            )

        except Exception as e:
            logging.exception(
                "Exception occured during data transformation. Error: %s", e
            )
