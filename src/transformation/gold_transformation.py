from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_replace


def transform_to_gold(df: DataFrame) -> DataFrame:
    """
    Transform Silver data into the Gold ML dataset.

    Creates a binary default target and removes identifiers,
    post-origination leakage, and features not suitable for
    origination-time default prediction.
    """

    # Keep only loans with a final outcome.
    df = df.filter(
        col("loan_status").isin(
            "Fully Paid",
            "Does not meet the credit policy. Status:Fully Paid",
            "Charged Off",
            "Default",
            "Does not meet the credit policy. Status:Charged Off",
        )
    )

    # Create binary target.
    df = df.withColumn(
        "target",
        when(
            col("loan_status").isin(
                "Charged Off",
                "Default",
                "Does not meet the credit policy. Status:Charged Off",
            ),
            1,
        ).otherwise(0),
    )

    # Remove target, identifiers, and post-origination leakage.
    columns_to_drop = [
        "loan_status",
        "id",
        "url",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "hardship_flag",
        "debt_settlement_flag",
        "emp_title",
        "zip_code",
        "pymnt_plan",
        "annual_inc_joint",
        "dti_joint",
        "verification_status_joint",
        "revol_bal_joint",
    ]

    df = df.drop(*columns_to_drop)

    df = df.withColumn(
    "emp_length",
    when(col("emp_length") == "< 1 year", 0)
    .when(col("emp_length") == "10+ years", 10)
    .otherwise(
        regexp_replace(col("emp_length"), " years?", "").cast("double")
    )
)

    return df