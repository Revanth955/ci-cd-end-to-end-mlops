"""
Pydantic request schemas for the loan default prediction API.
"""

from typing import Optional

from pydantic import BaseModel


class LoanRequest(BaseModel):
    """
    Validate and type-check loan features received by the prediction API.

    Numerical features are converted to float values when possible.
    Missing numerical values are allowed because the ML preprocessing
    pipeline handles missing values through imputation.
    """

    loan_amnt: Optional[float] = None
    funded_amnt: Optional[float] = None
    funded_amnt_inv: Optional[float] = None
    term: Optional[float] = None
    int_rate: Optional[float] = None
    installment: Optional[float] = None

    grade: Optional[str] = None
    sub_grade: Optional[str] = None

    emp_length: Optional[float] = None
    home_ownership: Optional[str] = None
    annual_inc: Optional[float] = None
    verification_status: Optional[str] = None
    issue_d: Optional[str] = None
    purpose: Optional[str] = None
    title: Optional[str] = None
    addr_state: Optional[str] = None

    dti: Optional[float] = None
    delinq_2yrs: Optional[float] = None
    earliest_cr_line: Optional[str] = None
    fico_range_low: Optional[float] = None
    fico_range_high: Optional[float] = None
    inq_last_6mths: Optional[float] = None
    mths_since_last_delinq: Optional[float] = None
    mths_since_last_record: Optional[float] = None
    open_acc: Optional[float] = None
    pub_rec: Optional[float] = None
    revol_bal: Optional[float] = None
    revol_util: Optional[float] = None
    total_acc: Optional[float] = None

    initial_list_status: Optional[str] = None
    collections_12_mths_ex_med: Optional[float] = None
    mths_since_last_major_derog: Optional[float] = None
    policy_code: Optional[float] = None
    application_type: Optional[str] = None
    acc_now_delinq: Optional[float] = None
    tot_coll_amt: Optional[float] = None
    tot_cur_bal: Optional[float] = None
    open_acc_6m: Optional[float] = None
    open_act_il: Optional[float] = None
    open_il_12m: Optional[float] = None
    open_il_24m: Optional[float] = None
    mths_since_rcnt_il: Optional[float] = None
    total_bal_il: Optional[float] = None
    il_util: Optional[float] = None
    open_rv_12m: Optional[float] = None
    open_rv_24m: Optional[float] = None
    max_bal_bc: Optional[float] = None
    all_util: Optional[float] = None
    total_rev_hi_lim: Optional[float] = None
    inq_fi: Optional[float] = None
    total_cu_tl: Optional[float] = None
    inq_last_12m: Optional[float] = None
    acc_open_past_24mths: Optional[float] = None
    avg_cur_bal: Optional[float] = None
    bc_open_to_buy: Optional[float] = None
    bc_util: Optional[float] = None
    chargeoff_within_12_mths: Optional[float] = None
    delinq_amnt: Optional[float] = None
    mo_sin_old_il_acct: Optional[float] = None
    mo_sin_old_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_tl: Optional[float] = None
    mort_acc: Optional[float] = None
    mths_since_recent_bc: Optional[float] = None
    mths_since_recent_bc_dlq: Optional[float] = None
    mths_since_recent_inq: Optional[float] = None
    mths_since_recent_revol_delinq: Optional[float] = None
    num_accts_ever_120_pd: Optional[float] = None
    num_actv_bc_tl: Optional[float] = None
    num_actv_rev_tl: Optional[float] = None
    num_bc_sats: Optional[float] = None
    num_bc_tl: Optional[float] = None
    num_il_tl: Optional[float] = None
    num_op_rev_tl: Optional[float] = None
    num_rev_accts: Optional[float] = None
    num_rev_tl_bal_gt_0: Optional[float] = None
    num_sats: Optional[float] = None
    num_tl_120dpd_2m: Optional[float] = None
    num_tl_30dpd: Optional[float] = None
    num_tl_90g_dpd_24m: Optional[float] = None
    num_tl_op_past_12m: Optional[float] = None
    pct_tl_nvr_dlq: Optional[float] = None
    percent_bc_gt_75: Optional[float] = None
    pub_rec_bankruptcies: Optional[float] = None
    tax_liens: Optional[float] = None
    tot_hi_cred_lim: Optional[float] = None
    total_bal_ex_mort: Optional[float] = None
    total_bc_limit: Optional[float] = None
    total_il_high_credit_limit: Optional[float] = None

    disbursement_method: Optional[str] = None