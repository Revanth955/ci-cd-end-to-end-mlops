from pyspark.sql import SparkSession

from src.config.configuration import RAW_DATA_FILE, BRONZE_DATA_PATH
from src.utils.logger import logger


def ingest_to_bronze(spark: SparkSession) -> None:
    """
    Read the raw CSV dataset and write it to the Bronze layer.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session used for reading and writing the dataset.

    Raises
    ------
    FileNotFoundError
        If the configured raw dataset does not exist.
    """

    # Validate that the configured raw dataset exists before starting ingestion.
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_FILE}")

    logger.info("Raw dataset found: %s", RAW_DATA_FILE)
    logger.info("Starting Bronze ingestion")

    # Read the raw CSV without inferring data types.
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(RAW_DATA_FILE))
    )

    logger.info("Raw dataset loaded successfully with Spark")
    logger.info("Bronze dataset contains %d columns", len(df.columns))

    # Write the raw data to the Bronze layer without changing its structure.
    (
        df.write
        .mode("overwrite")
        .option("header", True)
        .csv(str(BRONZE_DATA_PATH))
    )

    logger.info(
        "Bronze ingestion completed successfully: %s",
        BRONZE_DATA_PATH,
    )