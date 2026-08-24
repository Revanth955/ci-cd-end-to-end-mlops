from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    datediff,
    year,
    month,
)


def create_date_features(df: DataFrame) -> DataFrame:
    """
    Create model-friendly features from loan date columns.

    Features created:
    - issue_year
    - issue_month
    - credit_history_months
    """

    df = df.withColumn(
        "issue_year",
        year(col("issue_d")),
    )

    df = df.withColumn(
        "issue_month",
        month(col("issue_d")),
    )

    df = df.withColumn(
        "credit_history_months",
        datediff(
            col("issue_d"),
            col("earliest_cr_line"),
        ) / 30.44,
    )

    # Raw dates are no longer required after feature creation.
    df = df.drop(
        "issue_d",
        "earliest_cr_line",
    )

    return df