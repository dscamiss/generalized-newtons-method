"""Tests for gen_optimizer.py."""

# flake8: noqa=D401
# mypy: disable-error-code=no-untyped-def

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.generalized_newtons_method import GenOptimizer
from src.generalized_newtons_method.types import Criterion, Optimizer

_GEN_SGD_OPTIMIZERS = ["gen_sgd_minimize", "gen_sgd_maximize"]


def test_creation(gen_sgd_minimize: Optimizer) -> None:
    """Test wrapper class creation."""
    assert isinstance(gen_sgd_minimize, GenOptimizer), "Invalid class type"
    assert len(type(gen_sgd_minimize).__bases__) == 2, "Invalid inheritance pattern"
    assert type(gen_sgd_minimize).__bases__[0] == GenOptimizer, "Invalid superclass type"
    assert type(gen_sgd_minimize).__bases__[1] == torch.optim.SGD, "Invalid superclass type"


def test_get_update_invalid(model: nn.Module, gen_sgd_minimize: Optimizer) -> None:
    """Test `get_param_update()` with invalid optimizer state."""
    gen_sgd_minimize.reset()  # Ensure invalid optimizer state
    for param in model.parameters():
        with pytest.raises(ValueError):
            gen_sgd_minimize.get_param_update(param)


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_sgd", _GEN_SGD_OPTIMIZERS)
def test_get_update_valid(
    model: nn.Module,
    _sgd: str,
    mse: Criterion,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    request,
) -> None:
    """
    Test `get_param_update()` with valid optimizer state.

    Note: This test is only valid for vanilla SGD.
    """
    # Get test fixture
    sgd = request.getfixturevalue(_sgd)

    # Compute parameter gradients
    sgd.zero_grad()
    mse(model(x), y).backward()

    # Compute parameter updates
    sgd.compute_param_updates()

    # Check for errors in parameter update values
    err_str = "Error in parameter update value"
    for group in sgd.param_groups:
        for param in group["params"]:
            if "maximize" not in group or not group["maximize"]:
                expected_param_update = param.grad
            else:
                expected_param_update = -1.0 * param.grad
            actual_param_update = sgd.get_param_update(param)
            assert torch.allclose(expected_param_update, actual_param_update), err_str


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_sgd", _GEN_SGD_OPTIMIZERS)
def test_computation_graph_after_compute_param_updates(
    model: nn.Module,
    _sgd: str,
    mse: Criterion,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    request,
) -> None:
    """Check computation graph after invoking `compute_param_updates()`."""
    # Get test fixture
    sgd = request.getfixturevalue(_sgd)

    # Compute parameter gradients
    sgd.zero_grad()
    mse(model(x), y).backward()

    # Compute parameter updates
    sgd.compute_param_updates()

    # Check for errors in `requires_grad` property
    err_str = "Unexpected requires_grad value"
    for group in sgd.param_groups:
        for param in group["params"]:
            param_update = sgd.get_param_update(param)
            assert not param_update.requires_grad, err_str
