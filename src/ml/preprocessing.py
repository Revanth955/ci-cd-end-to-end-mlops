from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
    StandardScaler,
)
from pyspark.sql import DataFrame


def get_feature_columns(df: DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical feature columns.

    The target column is excluded from both feature groups.
    """

    numerical_columns = [
        field.name
        for field in df.schema.fields
        if field.dataType.simpleString()
        in ["double", "int", "bigint", "float", "long"]
        and field.name != "target"
    ]

    categorical_columns = [
        field.name
        for field in df.schema.fields
        if field.dataType.simpleString() == "string"
        and field.name != "title"
    ]

    return numerical_columns, categorical_columns


def build_preprocessing_pipeline(
    df: DataFrame,
    scale_numeric: bool = False,
) -> Pipeline:
    """
    Build the Spark ML preprocessing pipeline.

    Numerical features:
        Median imputation.

    Categorical features:
        String indexing followed by one-hot encoding.

    Parameters
    ----------
    df : DataFrame
        Training DataFrame used to identify feature types.
    scale_numeric : bool
        Whether numerical features should be standardized.

    Returns
    -------
    Pipeline
        Spark ML preprocessing pipeline.
    """

    numerical_columns, categorical_columns = get_feature_columns(df)

    imputed_columns = [
        f"{column}_imputed"
        for column in numerical_columns
    ]

    imputer = Imputer(
        inputCols=numerical_columns,
        outputCols=imputed_columns,
        strategy="median",
    )

    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_indexed",
            handleInvalid="keep",
        )
        for column in categorical_columns
    ]

    indexed_columns = [
        f"{column}_indexed"
        for column in categorical_columns
    ]

    encoder = OneHotEncoder(
        inputCols=indexed_columns,
        outputCols=[
            f"{column}_encoded"
            for column in categorical_columns
        ],
    )

    encoded_columns = [
        f"{column}_encoded"
        for column in categorical_columns
    ]

    assembler = VectorAssembler(
        inputCols=imputed_columns + encoded_columns,
        outputCol="features_unscaled",
        handleInvalid="keep",
    )

    stages = [
        imputer,
        *indexers,
        encoder,
        assembler,
    ]

    if scale_numeric:
        scaler = StandardScaler(
            inputCol="features_unscaled",
            outputCol="features",
            withMean=False,
            withStd=True,
        )

        stages.append(scaler)

    return Pipeline(stages=stages)