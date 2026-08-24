from pyspark.sql import SparkSession

from src.config.configuration import (
    GOLD_DATA_PATH,
    TRAINING_SAMPLE_FRACTION,
    TRAINING_SAMPLE_SEED,
    CLASSIFICATION_THRESHOLD,
)
from src.pipeline.silver_pipeline import run_silver_pipeline
from src.pipeline.gold_pipeline import run_gold_pipeline
from src.ingestion.bronze_ingestion import ingest_to_bronze
from src.split.train_sampling import sample_training_data
from src.split.train_test_split import time_based_split
from src.transformation.feature_engineering import create_date_features
from src.ml.evaluation import calculate_binary_metrics
from src.ml.model_training import build_logistic_regression_pipeline
from src.utils.logger import logger


def create_spark_session() -> SparkSession:
    """
    Create and return a local Spark session for the ML pipeline.

    Returns
    -------
    SparkSession
        Active Spark session used by the pipeline.
    """

    logger.info("Creating Spark session")

    spark = (
        SparkSession.builder
        .appName("EndToEndMLPipeline")
        .master("local[*]")
        .getOrCreate()
    )

    logger.info("Spark session created successfully")

    return spark


def run_pipeline() -> None:
    """
    Execute the end-to-end data and ML pipeline.
    """

    spark = create_spark_session()

    try:
        # Execute Bronze ingestion.
        ingest_to_bronze(spark)

        # Transform and validate Bronze data into Silver.
        run_silver_pipeline(spark)

        # Transform Silver data into the Gold ML dataset.
        run_gold_pipeline(spark)

        # Read the persisted Gold ML dataset.
        logger.info("Reading Gold dataset for model training")

        gold_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(GOLD_DATA_PATH))
        )

        logger.info(
            "Gold dataset loaded with %d columns",
            len(gold_df.columns),
        )

        # Split chronologically before removing the date columns.
        train_df, test_df = time_based_split(gold_df)

        logger.info(
            "Time-based split completed: train=%d rows, test=%d rows",
            train_df.count(),
            test_df.count(),
        )

        # Sample the training data to control local resource usage.
        train_df = sample_training_data(
            train_df,
            fraction=TRAINING_SAMPLE_FRACTION,
            seed=TRAINING_SAMPLE_SEED,
        )

        logger.info(
            "Training sample created using fraction=%.2f",
            TRAINING_SAMPLE_FRACTION,
        )

        # Create model-friendly date features.
        train_df = create_date_features(train_df)
        test_df = create_date_features(test_df)

        # Build and train the Logistic Regression pipeline.
        logger.info("Starting Logistic Regression training")

        pipeline = build_logistic_regression_pipeline(train_df)
        model = pipeline.fit(train_df)

        logger.info("Logistic Regression training completed")

        # Generate predictions on the future test period.
        predictions = model.transform(test_df)

        logger.info(
            "Generated predictions for %d test rows",
            predictions.count(),
        )

        # Evaluate the model using the configured classification threshold.
        metrics = calculate_binary_metrics(
            predictions,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        logger.info(
            "Evaluation results at threshold %.2f: "
            "precision=%.4f, recall=%.4f, f1=%.4f",
            CLASSIFICATION_THRESHOLD,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        logger.info("Pipeline execution completed successfully")

    except Exception:
        logger.exception("Pipeline execution failed")
        raise

    finally:
        # Always stop Spark, even when an exception occurs.
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    run_pipeline()