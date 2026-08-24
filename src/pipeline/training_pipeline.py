from pyspark.sql import SparkSession

from src.pipeline.silver_pipeline import run_silver_pipeline
from src.pipeline.gold_pipeline import run_gold_pipeline
from src.ingestion.bronze_ingestion import ingest_to_bronze
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