"""Tests for loss_per_learning_rate.py."""

# mypy: disable-error-code=no-untyped-def
# flake8: noqa=DCO030

import copy

import numpy as np
import pytest
import torch
from jaxtyping import Float, jaxtyped
from numpy.typing import NDArray
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.generalized_newtons_method.types import Criterion, Optimizer
from src.generalized_newtons_method.utils import loss_per_learning_rate


@pytest.fixture(name="learning_rates")
def fixture_learning_rates() -> NDArray:
    """Learning rates."""
    return np.linspace(0.0, 1.0, 100)


@jaxtyped(typechecker=typechecker)
def test_loss_per_learning_rate_invalid_model(
    model: nn.Module,
    mse: Criterion,
    sgd_minimize: Optimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    learning_rates: NDArray,
) -> None:
    """Test `loss_per_learning_rate()` with invalid `model` argument."""
    model.train()
    with pytest.raises(ValueError):
        loss_per_learning_rate(model, mse, sgd_minimize, x, y, learning_rates)


@jaxtyped(typechecker=typechecker)
def test_loss_per_learning_rate_output(
    model: nn.Module,
    mse: Criterion,
    sgd_minimize: Optimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    learning_rates: NDArray,
) -> None:
    """
    Test `loss_per_learning_rate()` output.

    This test is only valid for vanilla SGD.
    """
    # Save model state
    model_state_dict = copy.deepcopy(model.state_dict())

    # Compute gradients
    sgd_minimize.zero_grad()
    mse(model(x), y).backward()

    # Check outputs
    losses = loss_per_learning_rate(model, mse, sgd_minimize, x, y, learning_rates)
    with torch.no_grad():
        for learning_rate, loss in zip(learning_rates, losses):
            for group in sgd_minimize.param_groups:
                for param in group["params"]:
                    param.add_(param.grad, alpha=-learning_rate)
            assert mse(model(x), y) == loss, "Unexpected loss value"
            model.load_state_dict(model_state_dict)


@jaxtyped(typechecker=typechecker)
def test_loss_per_learning_rate_output_size(
    model: nn.Module,
    mse: Criterion,
    sgd_minimize: Optimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    learning_rates: NDArray,
) -> None:
    """Test `loss_per_learning_rate()` output size."""
    # Compute gradients
    sgd_minimize.zero_grad()
    mse(model(x), y).backward()

    # Check output size
    losses = loss_per_learning_rate(model, mse, sgd_minimize, x, y, learning_rates)
    assert len(losses) == len(learning_rates), "Error in output size"


@jaxtyped(typechecker=typechecker)
def test_loss_per_learning_rate_side_effects(
    model: nn.Module,
    mse: Criterion,
    sgd_minimize: Optimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    learning_rates: NDArray,
) -> None:
    """Test `loss_per_learning_rate()` for side effects."""
    # Alias for brevity
    sgd = sgd_minimize

    # Save model state and optimizer state
    model_state_dict = copy.deepcopy(model.state_dict())
    sgd_state_dict = copy.deepcopy(sgd.state_dict())

    # Compute gradients
    sgd.zero_grad()
    mse(model(x), y).backward()

    # Check for side effects
    loss_per_learning_rate(model, mse, sgd, x, y, learning_rates)

    err_str = "Unexpected change (keys)"
    assert model_state_dict.keys() == model.state_dict().keys(), err_str
    assert sgd_state_dict.keys() == sgd.state_dict().keys(), err_str

    err_str = "Unexpected change (values)"
    for key in model_state_dict:
        assert torch.equal(model_state_dict[key], model.state_dict()[key]), err_str
    for key in sgd_state_dict:
        value_1 = sgd_state_dict[key]
        value_2 = sgd.state_dict()[key]
        if isinstance(value_1, Tensor):
            assert torch.equal(value_1, value_2), err_str
        else:
            assert value_1 == value_2, err_str
