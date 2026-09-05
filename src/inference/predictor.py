"""
Model inference utilities.

This module loads the Champion model from MLflow
and provides the Spark runtime required for inference.
"""

import mlflow
from pyspark.sql import SparkSession
from src.transformation.feature_engineering import create_date_features
from src.utils.logger import logger


MODEL_NAME = "LoanDefaultModel"
MODEL_ALIAS = "champion"


def create_inference_spark_session() -> SparkSession:
    """
    Create and return the Spark session used for inference.
    """

    logger.info("Creating Spark session for inference")

    spark = (
        SparkSession.builder
        .appName("LoanDefaultInference")
        .master("local[*]")
        .getOrCreate()
    )

    logger.info("Inference Spark session created successfully")

    return spark


def load_champion_model():
    """
    Load the current Champion model from MLflow.

    Returns
    -------
    PipelineModel
        Spark ML PipelineModel assigned to the Champion alias.
    """

    logger.info(
        "Loading Champion model: %s@%s",
        MODEL_NAME,
        MODEL_ALIAS,
    )

    model = mlflow.spark.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    logger.info("Champion model loaded successfully")

    return model



def predict(spark, model, input_df):
    """
    Generate a prediction for an input Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session used for inference.

    model : PipelineModel
        Loaded Champion Spark model.

    input_df : DataFrame
        Input loan data before date feature engineering.

    Returns
    -------
    dict
        Prediction result containing the predicted class
        and probability of default.
    """

    # Create the same date features used during training.
    input_df = create_date_features(input_df)

    # Generate model predictions.
    predictions = model.transform(input_df)

    # Extract the first prediction row.
    result = predictions.select(
        "prediction",
        "probability",
    ).first()

    prediction = int(result["prediction"])
    probability = float(result["probability"][1])

    return {
        "prediction": prediction,
        "probability": probability,
    }