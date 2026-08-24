from pyspark.sql import SparkSession

from src.config.configuration import  BRONZE_DATA_PATH, SILVER_DATA_PATH
from src.utils.logger import logger
from src.transformation.silver_transformation import transform_to_silver
from src.validation.silver_validation import validate_silver

def run_silver_pipeline(spark: SparkSession) -> None:
    if not BRONZE_DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {BRONZE_DATA_PATH}")
    logger.info("Data found: %s", BRONZE_DATA_PATH)
    logger.info("Starting Silver pipeline")
    # Read the raw Bronze data without inferring data types.
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(str(BRONZE_DATA_PATH))
    )

    logger.info("Bronze dataset contains %d columns", len(df.columns))

    df = transform_to_silver(df)
    logger.info("Silver dataset contains %d columns", len(df.columns))
    validation_results = validate_silver(df)

    for check_name, violations in validation_results.items():
        violation_count = violations.limit(1).count()

        if violation_count > 0:
            logger.error("Silver validation failed: %s", check_name)
            raise ValueError(f"Silver validation failed: {check_name}")

    logger.info("Silver validation passed")


    # Write the validated Silver data to the Silver layer.
    (
        df.write
        .mode("overwrite")
        .option("header", True)
        .csv(str(SILVER_DATA_PATH))
    )

    logger.info(
        "Silver ingestion completed successfully: %s",
        SILVER_DATA_PATH,
    )

