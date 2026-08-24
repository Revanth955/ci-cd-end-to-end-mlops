from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum


def check_duplicate_ids(df: DataFrame) -> DataFrame:
    """
    Identify loan IDs that occur more than once.
    """
    return (
        df.groupBy("id")
        .count()
        .filter(col("count") > 1)
    )

def check_invalid_values(df: DataFrame) -> DataFrame:
    """
    Identify records containing invalid business values.
    """
    return df.filter(
        (col("loan_amnt") <= 0)
        | (col("int_rate") < 0)
        | (col("installment") <= 0)
        | (col("dti") < 0)
    )
def check_target_values(df: DataFrame) -> DataFrame:
    """
    Identify records with unexpected loan status values.
    """
    valid_statuses = [
    "Fully Paid",
    "Charged Off",
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",]

    return df.filter(
        ~col("loan_status").isin(valid_statuses)
    )
def analyze_nulls(df: DataFrame) -> DataFrame:
    """
    Calculate NULL counts and NULL percentages for every column.
    """

    null_counts = df.select(
        [
            sum(col(column).isNull().cast("int")).alias(column)
            for column in df.columns
        ]
    )

    stack_expr = ", ".join(
        [f"'{column}', `{column}`" for column in df.columns]
    )

    return (
        null_counts
        .selectExpr(
            f"stack({len(df.columns)}, {stack_expr}) "
            "as (column_name, null_count)"
        )
        .withColumn(
            "null_percentage",
            (col("null_count") / df.count()) * 100
        )
        .orderBy(col("null_percentage").desc())
    )

def validate_silver(df: DataFrame) -> dict:
    """
    Run all Silver-layer validation checks.
    """

    duplicate_ids = check_duplicate_ids(df)
    invalid_values = check_invalid_values(df)
    invalid_targets = check_target_values(df)

    return {
        "duplicate_ids": duplicate_ids,
        "invalid_values": invalid_values,
        "invalid_targets": invalid_targets,
    }