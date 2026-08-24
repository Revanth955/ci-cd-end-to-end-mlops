from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def time_based_split(
    df: DataFrame,
    split_date: str = "2018-01-01",
) -> tuple[DataFrame, DataFrame]:
    """
    Split Gold data chronologically using the loan issue date.

    Records before the split date are used for training.
    Records on or after the split date are used for testing.

    Parameters
    ----------
    df : DataFrame
        Gold ML dataset.
    split_date : str
        Date separating historical training data from future test data.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Training and test DataFrames.
    """

    train_df = df.filter(
        col("issue_d") < split_date
    )

    test_df = df.filter(
        col("issue_d") >= split_date
    )

    return train_df, test_df