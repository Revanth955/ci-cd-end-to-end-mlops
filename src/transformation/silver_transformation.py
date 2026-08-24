from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, to_date


def transform_to_silver(df: DataFrame) -> DataFrame:
    """
    Transform Bronze data into the Silver layer.

    Transformations:
    - Normalize column names.
    - Cast required numeric columns.
    - Convert date columns to Spark DateType.
    - Remove unnecessary/high-null columns.
    - Remove records with a missing target.
    """

    # Normalize column names to lowercase.
    for column in df.columns:
        df = df.withColumnRenamed(column, column.lower())

    # Cast numeric columns.
    # Use double first because Bronze values may contain decimal notation
    # such as "24000.0".
    df = df.withColumn(
        "loan_amnt",
        col("loan_amnt").cast("double")
    )

    df = df.withColumn(
        "int_rate",
        col("int_rate").cast("double")
    )

    df = df.withColumn(
        "installment",
        col("installment").cast("double")
    )
    df = df.withColumn(
    "dti",
    col("dti").cast("double"))


    # Convert issue date from MMM-yyyy to Spark DateType.
    df = df.withColumn(
        "issue_d",
        to_date(col("issue_d"), "MMM-yyyy")
    )

    # Remove " months" and convert term to numeric.
    # Cast to double because values may be represented as "36.0".
    df = df.withColumn(
        "term",
        regexp_replace(col("term"), " months", "").cast("double")
    )

    # Columns removed because they are unnecessary for the
    # baseline ML dataset or contain extremely high NULL rates.
    columns_to_drop = [
        "member_id",
        "orig_projected_additional_accrued_interest",

        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",

        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",

        "sec_app_mths_since_last_major_derog",
        "sec_app_revol_util",
        "sec_app_fico_range_low",
        "sec_app_fico_range_high",
        "sec_app_earliest_cr_line",
        "sec_app_inq_last_6mths",
        "sec_app_mort_acc",
        "sec_app_open_acc",
        "sec_app_open_act_il",
        "sec_app_num_rev_accts",
        "sec_app_chargeoff_within_12_mths",
        "sec_app_collections_12_mths_ex_med",

        "desc",
    ]

    # Remove unnecessary columns.
    df = df.drop(*columns_to_drop)

    # Remove records where the target is missing.
    df = df.dropna(subset=["loan_status"])
    df = df.filter(col("dti") >= 0)

    # Convert remaining MMM-yyyy columns to Spark DateType.
    df = df.withColumn(
        "earliest_cr_line",
        to_date(col("earliest_cr_line"), "MMM-yyyy")
    )

    df = df.withColumn(
        "last_pymnt_d",
        to_date(col("last_pymnt_d"), "MMM-yyyy")
    )

    df = df.withColumn(
        "next_pymnt_d",
        to_date(col("next_pymnt_d"), "MMM-yyyy")
    )

    df = df.withColumn(
        "last_credit_pull_d",
        to_date(col("last_credit_pull_d"), "MMM-yyyy")
    )


    return df