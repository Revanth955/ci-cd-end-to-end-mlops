"""
Tests for model promotion orchestration.
"""

from types import SimpleNamespace
from unittest.mock import patch

from src.ml.model_promotion import promote_model


RUN_ID = "test-run-id"
MODEL_NAME = "LoanDefaultModel"


@patch("src.ml.model_promotion.register_model")
@patch("src.ml.model_promotion.validate_model")
def test_promote_model_registers_when_validation_passes(
    mock_validate_model,
    mock_register_model,
):
    """A validated model should proceed to registration."""

    # Simulate a successful validation result.
    mock_validate_model.return_value = {
        "passed": True,
    }

    # Simulate the object returned by MLflow registration.
    mock_register_model.return_value = SimpleNamespace(
        name=MODEL_NAME,
        version="2",
    )

    result = promote_model(
        run_id=RUN_ID,
        model_name=MODEL_NAME,
    )

    # Validation must be performed first.
    mock_validate_model.assert_called_once_with(RUN_ID)

    # A passing model must be sent for registration.
    mock_register_model.assert_called_once_with(
        run_id=RUN_ID,
        model_name=MODEL_NAME,
    )

    # The registered model version should be returned.
    assert result.version == "2"


@patch("src.ml.model_promotion.register_model")
@patch("src.ml.model_promotion.validate_model")
def test_promote_model_does_not_register_when_validation_fails(
    mock_validate_model,
    mock_register_model,
):
    """A failed model should not proceed to registration."""

    validation_result = {
        "passed": False,
    }

    # Simulate a failed validation result.
    mock_validate_model.return_value = validation_result

    result = promote_model(
        run_id=RUN_ID,
        model_name=MODEL_NAME,
    )

    # Validation must still be performed.
    mock_validate_model.assert_called_once_with(RUN_ID)

    # A failed model must never be registered.
    mock_register_model.assert_not_called()

    # The validation result should be returned to the caller.
    assert result == validation_result