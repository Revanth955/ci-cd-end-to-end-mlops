from pyspark.sql import DataFrame


def sample_training_data(
    train_df: DataFrame,
    fraction: float = 1.0,
    seed: int = 42,
) -> DataFrame:
    """
    Sample the training DataFrame for resource-constrained training.

    Parameters
    ----------
    train_df : DataFrame
        Historical training DataFrame.
    fraction : float
        Fraction of training data to retain.
        A value of 1.0 keeps the complete training set.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    DataFrame
        Sampled training DataFrame.

    Raises
    ------
    ValueError
        If fraction is not between 0 and 1.
    """

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and less than or equal to 1.")

    if fraction == 1.0:
        return train_df

    return train_df.sample(
        withReplacement=False,
        fraction=fraction,
        seed=seed,
    )