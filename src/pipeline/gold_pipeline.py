from pyspark.sql import SparkSession

from src.config.configuration import GOLD_DATA_PATH, SILVER_DATA_PATH
from src.transformation.gold_transformation import transform_to_gold
from src.utils.logger import logger


def run_gold_pipeline(spark: SparkSession) -> None:
    """
    Transform the Silver dataset into the Gold ML dataset
    and persist the result.
    """

    if not SILVER_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Silver data not found: {SILVER_DATA_PATH}"
        )

    logger.info("Silver data found: %s", SILVER_DATA_PATH)
    logger.info("Starting Gold pipeline")

    # Read the Silver dataset.
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(SILVER_DATA_PATH))
    )

    logger.info(
        "Silver dataset contains %d columns",
        len(df.columns),
    )

    # Transform Silver data into the Gold ML dataset.
    gold_df = transform_to_gold(df)

    logger.info(
        "Gold dataset contains %d columns",
        len(gold_df.columns),
    )

    # Persist the Gold dataset.
    (
        gold_df.write
        .mode("overwrite")
        .option("header", True)
        .csv(str(GOLD_DATA_PATH))
    )

    logger.info(
        "Gold dataset written successfully: %s",
        GOLD_DATA_PATH,
    )