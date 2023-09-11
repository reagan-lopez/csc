import warnings
from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    QuantileDiscretizer,
)
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    LogisticRegression,
    GBTClassifier,
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.spark import log_model

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

models = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GBTClassifier": GBTClassifier,
}


def format_data(df):
    # Calculate MSRP column
    df = df.withColumn(
        "MSRP", (df["PurchasePrice"] / (1 - df["DiscountPct"])).cast("integer")
    )

    # Calculate ProductID column
    df = df.withColumn(
        "ProductID",
        (
            df["ProductDepartment"]
            + "-"
            + df["ProductCost"].cast("string")
            + "-"
            + df["MSRP"].cast("string")
        ),
    )

    # Convert date columns to datetime format
    df = df.withColumn("OrderDate", df["OrderDate"].cast("date"))
    df = df.withColumn("CustomerBirthDate", df["CustomerBirthDate"].cast("date"))

    # Calculate CustomerAge at the time of OrderDate
    df = df.withColumn(
        "CustomerAge",
        (
            (
                (
                    df["OrderDate"].cast("timestamp").cast("long")
                    - df["CustomerBirthDate"].cast("timestamp").cast("long")
                )
                / (365 * 24 * 60 * 60)
            )
        ).cast("integer"),
    )

    return df


def eval_metrics(actual, pred):
    roc_auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    brier_score_evaluator = BinaryClassificationEvaluator(metricName="brierScore")

    roc_auc = roc_auc_evaluator.evaluate(pred)
    brier_score = brier_score_evaluator.evaluate(pred)

    return roc_auc, brier_score


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

    # Create indexers for categorical features
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep")
        for col in categorical_features
    ]

    # One-hot encode categorical features
    encoders = [
        OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded")
        for col in categorical_features
    ]

    # Assemble the feature vector
    assembler = VectorAssembler(
        inputCols=numeric_features + [col + "_encoded" for col in categorical_features],
        outputCol="features",
    )

    # Define the age discretizer
    discretizer = QuantileDiscretizer(
        numBuckets=6, inputCol="CustomerAge", outputCol="CustomerAge_discretized"
    )

    # Create the preprocessor stages
    stages = indexers + encoders + date_converters + [assembler, discretizer]

    # Create the preprocessor pipeline
    preprocessor = Pipeline(stages=stages)

    return preprocessor


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    spark = SparkSession.builder.appName("YourAppName").getOrCreate()

    dataset = "data/training.csv"
    try:
        df = spark.read.csv(dataset, header=True, inferSchema=True)
    except Exception as e:
        logger.exception("Unable to read dataset. Error: %s", e)

    df = format_data(df)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = df.randomSplit([0.75, 0.25], seed=42)

    # The predicted column is "Returned" which is boolean
    feature_cols = df.columns
    feature_cols.remove("Returned")

    # Create a list of feature columns
    feature_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Create a pipeline for preprocessing
    preprocessor = Pipeline(stages=[feature_assembler])

    # Fit the preprocessor on the training data
    preprocessor_model = preprocessor.fit(train)

    # Transform the training and test data using the preprocessor
    train = preprocessor_model.transform(train)
    test = preprocessor_model.transform(test)

    model_scores = {}
    for model_name in models:
        print(f"\nEvaluating model: {model_name}")

        # Create the classifier
        classifier = models[model_name]

        # Create a pipeline including the classifier
        pipeline = Pipeline(stages=[classifier])

        with mlflow.start_run():
            # Fit the pipeline on the training data
            model = pipeline.fit(train)

            # Make predictions on the testing data
            predictions = model.transform(test)

            roc_auc, brier_score = eval_metrics(test["Returned"], predictions)
            model_scores[model_name] = [roc_auc, brier_score]
            print(f"  ROC AUC: {roc_auc}")
            print(f"  Brier Score: {brier_score}")

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("brier_score", brier_score)

            # Log the model to MLflow
            log_model(model, model_name)

    eval_models(model_scores)

    spark.stop()
