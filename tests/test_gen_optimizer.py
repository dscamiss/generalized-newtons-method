"""Tests for gen_optimizer.py."""

# flake8: noqa=D401
# mypy: disable-error-code=no-untyped-def

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.generalized_newtons_method import GenOptimizer, make_gen_optimizer
from src.generalized_newtons_method.types import Criterion

_OPTIMIZERS = ["optimizer_minimize", "optimizer_maximize"]


@pytest.fixture(name="optimizer_minimize")
@jaxtyped(typechecker=typechecker)
def fixture_optimizer_minimize(model: nn.Module) -> GenOptimizer:
    """Wrapped vanilla SGD optimizer (minimizing)."""
    return make_gen_optimizer(torch.optim.SGD, model.parameters())


@pytest.fixture(name="optimizer_maximize")
@jaxtyped(typechecker=typechecker)
def fixture_optimizer_maximize(model: nn.Module) -> GenOptimizer:
    """Wrapped vanilla SGD optimizer (maximizing)."""
    return make_gen_optimizer(torch.optim.SGD, model.parameters(), maximize=True)


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_optimizer", _OPTIMIZERS)
def test_creation(_optimizer: str, request) -> None:
    """Test wrapper class creation."""
    optimizer = request.getfixturevalue(_optimizer)
    assert isinstance(optimizer, GenOptimizer), "Invalid class type"
    assert len(type(optimizer).__bases__) == 2, "Invalid inheritance pattern"
    assert type(optimizer).__bases__[0] == GenOptimizer, "Invalid superclass type"
    assert type(optimizer).__bases__[1] == torch.optim.SGD, "Invalid superclass type"


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_optimizer", _OPTIMIZERS)
def test_get_update_invalid(model: nn.Module, _optimizer: str, request) -> None:
    """Test `get_param_update()` with invalid optimizer state."""
    optimizer = request.getfixturevalue(_optimizer)
    optimizer.reset()  # Ensure uninitialized state
    for param in model.parameters():
        with pytest.raises(ValueError):
            optimizer.get_param_update(param)


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_optimizer", _OPTIMIZERS)
def test_get_update_valid(
    model: nn.Module,
    _optimizer: str,
    criterion: Criterion,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    request,
) -> None:
    """
    Test `get_param_update()` with valid optimizer state.

    Note: This test is only valid for the vanilla SGD optimizer.
    """
    optimizer = request.getfixturevalue(_optimizer)

    # Compute parameter gradients
    optimizer.zero_grad()
    criterion(model(x), y).backward()

    # Compute parameter updates
    optimizer.compute_param_updates()

    # Check for mismatches between expected and actual parameter updates
    error_str = "Mismatch between expected and actual updates"
    for group in optimizer.param_groups:
        for param in group["params"]:
            if "maximize" not in group or not group["maximize"]:
                expected_param_update = param.grad
            else:
                expected_param_update = -1.0 * param.grad
            actual_param_update = optimizer.get_param_update(param)
            assert torch.allclose(expected_param_update, actual_param_update), error_str


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("_optimizer", _OPTIMIZERS)
def test_computation_graph_sanity(
    model: nn.Module,
    _optimizer: str,
    criterion: Criterion,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    request,
) -> None:
    """Check sanity of computation graph after invoking `step()`."""
    optimizer = request.getfixturevalue(_optimizer)

    # Compute parameter gradients
    optimizer.zero_grad()
    criterion(model(x), y).backward()

    # Compute parameter updates
    optimizer.compute_param_updates()

    # Check `requires_grad` for al parameter updates
    for group in optimizer.param_groups:
        for param in group["params"]:
            param_update = optimizer.get_param_update(param)
            assert not param_update.requires_grad, "Unexpected requires_grad value"
