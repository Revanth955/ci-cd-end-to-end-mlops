"""
FastAPI application for loan default inference.
"""

from api.schemas import LoanRequest
from fastapi import FastAPI, HTTPException
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.functions import to_date

from src.inference.predictor import load_champion_model, predict


app = FastAPI(
    title="Loan Default Prediction API",
    version="1.0.0",
)


# Create the Spark runtime once when the API process starts.
spark = (
    SparkSession.builder
    .appName("LoanDefaultInferenceAPI")
    .master("local[*]")
    .getOrCreate()
)

# Define the schema expected by the trained model.
# Explicit types are required because some incoming JSON fields may be null.
INPUT_SCHEMA = StructType([
    StructField("loan_amnt", DoubleType(), True),
    StructField("funded_amnt", DoubleType(), True),
    StructField("funded_amnt_inv", DoubleType(), True),
    StructField("term", DoubleType(), True),
    StructField("int_rate", DoubleType(), True),
    StructField("installment", DoubleType(), True),
    StructField("grade", StringType(), True),
    StructField("sub_grade", StringType(), True),
    StructField("emp_length", DoubleType(), True),
    StructField("home_ownership", StringType(), True),
    StructField("annual_inc", DoubleType(), True),
    StructField("verification_status", StringType(), True),
    StructField("issue_d", StringType(), True),
    StructField("purpose", StringType(), True),
    StructField("title", StringType(), True),
    StructField("addr_state", StringType(), True),
    StructField("dti", DoubleType(), True),
    StructField("delinq_2yrs", DoubleType(), True),
    StructField("earliest_cr_line", StringType(), True),
    StructField("fico_range_low", DoubleType(), True),
    StructField("fico_range_high", DoubleType(), True),
    StructField("inq_last_6mths", DoubleType(), True),
    StructField("mths_since_last_delinq", DoubleType(), True),
    StructField("mths_since_last_record", DoubleType(), True),
    StructField("open_acc", DoubleType(), True),
    StructField("pub_rec", DoubleType(), True),
    StructField("revol_bal", DoubleType(), True),
    StructField("revol_util", DoubleType(), True),
    StructField("total_acc", DoubleType(), True),
    StructField("initial_list_status", StringType(), True),
    StructField("collections_12_mths_ex_med", DoubleType(), True),
    StructField("mths_since_last_major_derog", DoubleType(), True),
    StructField("policy_code", DoubleType(), True),
    StructField("application_type", StringType(), True),
    StructField("acc_now_delinq", DoubleType(), True),
    StructField("tot_coll_amt", DoubleType(), True),
    StructField("tot_cur_bal", DoubleType(), True),
    StructField("open_acc_6m", DoubleType(), True),
    StructField("open_act_il", DoubleType(), True),
    StructField("open_il_12m", DoubleType(), True),
    StructField("open_il_24m", DoubleType(), True),
    StructField("mths_since_rcnt_il", DoubleType(), True),
    StructField("total_bal_il", DoubleType(), True),
    StructField("il_util", DoubleType(), True),
    StructField("open_rv_12m", DoubleType(), True),
    StructField("open_rv_24m", DoubleType(), True),
    StructField("max_bal_bc", DoubleType(), True),
    StructField("all_util", DoubleType(), True),
    StructField("total_rev_hi_lim", DoubleType(), True),
    StructField("inq_fi", DoubleType(), True),
    StructField("total_cu_tl", DoubleType(), True),
    StructField("inq_last_12m", DoubleType(), True),
    StructField("acc_open_past_24mths", DoubleType(), True),
    StructField("avg_cur_bal", DoubleType(), True),
    StructField("bc_open_to_buy", DoubleType(), True),
    StructField("bc_util", DoubleType(), True),
    StructField("chargeoff_within_12_mths", DoubleType(), True),
    StructField("delinq_amnt", DoubleType(), True),
    StructField("mo_sin_old_il_acct", DoubleType(), True),
    StructField("mo_sin_old_rev_tl_op", DoubleType(), True),
    StructField("mo_sin_rcnt_rev_tl_op", DoubleType(), True),
    StructField("mo_sin_rcnt_tl", DoubleType(), True),
    StructField("mort_acc", DoubleType(), True),
    StructField("mths_since_recent_bc", DoubleType(), True),
    StructField("mths_since_recent_bc_dlq", DoubleType(), True),
    StructField("mths_since_recent_inq", DoubleType(), True),
    StructField("mths_since_recent_revol_delinq", DoubleType(), True),
    StructField("num_accts_ever_120_pd", DoubleType(), True),
    StructField("num_actv_bc_tl", DoubleType(), True),
    StructField("num_actv_rev_tl", DoubleType(), True),
    StructField("num_bc_sats", DoubleType(), True),
    StructField("num_bc_tl", DoubleType(), True),
    StructField("num_il_tl", DoubleType(), True),
    StructField("num_op_rev_tl", DoubleType(), True),
    StructField("num_rev_accts", DoubleType(), True),
    StructField("num_rev_tl_bal_gt_0", DoubleType(), True),
    StructField("num_sats", DoubleType(), True),
    StructField("num_tl_120dpd_2m", DoubleType(), True),
    StructField("num_tl_30dpd", DoubleType(), True),
    StructField("num_tl_90g_dpd_24m", DoubleType(), True),
    StructField("num_tl_op_past_12m", DoubleType(), True),
    StructField("pct_tl_nvr_dlq", DoubleType(), True),
    StructField("percent_bc_gt_75", DoubleType(), True),
    StructField("pub_rec_bankruptcies", DoubleType(), True),
    StructField("tax_liens", DoubleType(), True),
    StructField("tot_hi_cred_lim", DoubleType(), True),
    StructField("total_bal_ex_mort", DoubleType(), True),
    StructField("total_bc_limit", DoubleType(), True),
    StructField("total_il_high_credit_limit", DoubleType(), True),
    StructField("disbursement_method", StringType(), True),
])

# Load the current Champion model once at startup.
model = load_champion_model()


@app.get("/health")
def health_check() -> dict:
    """
    Return the API health status.
    """

    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict_default(loan: LoanRequest) -> dict:
    """
    Generate a loan default prediction.

    Parameters
    ----------
    loan : LoanRequest
        Validated loan feature values supplied by the API client.

    Returns
    -------
    dict
        Prediction and probability of default.
    """

    # Convert the validated Pydantic request into a dictionary
    # before creating the Spark DataFrame.
    input_df = spark.createDataFrame(
        [loan.model_dump()],
        schema=INPUT_SCHEMA,
    )

    # Convert date strings into Spark date columns.
    input_df = (
        input_df
        .withColumn("issue_d", to_date("issue_d"))
        .withColumn("earliest_cr_line", to_date("earliest_cr_line"))
    )

    # Generate the prediction using the Champion model.
    return predict(
        spark=spark,
        model=model,
        input_df=input_df,
    )