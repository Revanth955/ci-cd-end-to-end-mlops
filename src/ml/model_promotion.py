"""
Model promotion orchestration.

This module coordinates model validation and registration.

The module does not:
- train models,
- calculate evaluation metrics,
- implement validation rules,
- deploy models.

It orchestrates the workflow:
    Validate → Register if validation passes.
"""

from src.ml.model_validation import validate_model
from src.ml.model_registration import register_model
from src.utils.logger import logger


def should_register(validation_result: dict) -> bool:
    """
    Determine whether a validated model is eligible for registration.

    Parameters
    ----------
    validation_result : dict
        Detailed result returned by the model validation component.

    Returns
    -------
    bool
        True when the model passes validation; otherwise False.
    """

    return validation_result["passed"]


def promote_model(run_id: str, model_name: str):
    """
    Validate a model run and register it if validation passes.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the candidate model and metrics.

    model_name : str
        Name of the registered model.

    Returns
    -------
    object or dict
        Registered MLflow model version when validation passes.
        Validation result when validation fails.
    """

    logger.info(
        "Starting model promotion workflow for run: %s",
        run_id,
    )

    # Validate the candidate model.
    validation_result = validate_model(run_id)

    # Stop the workflow when the candidate fails validation.
    if not should_register(validation_result):
        logger.warning(
            "Model validation failed. Registration skipped for run: %s",
            run_id,
        )

        return validation_result

    # Register the validated model artifact.
    registered_model = register_model(
        run_id=run_id,
        model_name=model_name,
    )

    logger.info(
        "Model promotion workflow completed successfully: "
        "%s version %s",
        registered_model.name,
        registered_model.version,
    )

    return registered_model