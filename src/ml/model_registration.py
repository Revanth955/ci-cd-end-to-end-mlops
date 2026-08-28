"""
Model registration utilities.

This module registers an already-validated MLflow model artifact
as a version of a registered model.

The module does not:
- train models,
- calculate model metrics,
- validate model performance,
- deploy models.

Validation must be completed before this module is called.
"""

import mlflow

from src.utils.logger import logger


def register_model(run_id: str, model_name: str):
    """
    Register a model artifact from an existing MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the logged model artifact.

    model_name : str
        Name of the registered model.

    Returns
    -------
    ModelVersion
        MLflow registered model version.
    """

    logger.info(
        "Registering model from MLflow run: %s",
        run_id,
    )

    # The model artifact was logged under the "model" artifact path
    # by the training pipeline.
    model_uri = f"runs:/{run_id}/model"

    # Register the existing model artifact.
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    logger.info(
        "Model registered successfully: %s version %s",
        registered_model.name,
        registered_model.version,
    )

    return registered_model