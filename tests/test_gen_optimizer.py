"""Tests for gen_optimizer.py."""

# flake8: noqa=D401

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.generalized_newtons_method.gen_optimizer import gen_optimizer
from src.generalized_newtons_method.types import CustomCriterionType, OptimizerType


@pytest.fixture(name="gen_optimizer_sgd_minimize")
@jaxtyped(typechecker=typechecker)
def fixture_gen_optimizer_sgd_minimize(model: nn.Module) -> OptimizerType:
    """Wrapped vanilla SGD learning rate scheduler (minimizing)."""
    base_optimizer_class = torch.optim.SGD
    return gen_optimizer(base_optimizer_class, model.parameters())


@pytest.fixture(name="gen_optimizer_sgd_maximize")
@jaxtyped(typechecker=typechecker)
def fixture_gen_optimizer_sgd_maximize(model: nn.Module) -> OptimizerType:
    """Wrapped vanilla SGD learning rate scheduler (maximizing)."""
    base_optimizer_class = torch.optim.SGD
    return gen_optimizer(base_optimizer_class, model.parameters(), maximize=True)


@jaxtyped(typechecker=typechecker)
def test_creation(gen_optimizer_sgd_minimize: OptimizerType) -> None:
    """Test wrapper creation."""
    assert len(type(gen_optimizer_sgd_minimize).__bases__) == 1, "Invalid inheritance pattern"
    assert type(gen_optimizer_sgd_minimize).__base__ == torch.optim.SGD, "Invalid superclass"


@jaxtyped(typechecker=typechecker)
def test_get_update_invalid(model: nn.Module, gen_optimizer_sgd_minimize: OptimizerType) -> None:
    """Test `get_update()` with invalid optimizer state."""
    gen_optimizer_sgd_minimize.reset_param_cache()
    param = next(model.parameters())
    with pytest.raises(ValueError):
        gen_optimizer_sgd_minimize.get_param_update(param)


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize(
    "gen_optimizer_sgd_", ["gen_optimizer_sgd_minimize", "gen_optimizer_sgd_maximize"]
)
def test_get_update_valid(
    model: nn.Module,
    gen_optimizer_sgd_: str,
    criterion: CustomCriterionType,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    request,
) -> None:
    """Test `get_update()` with valid optimizer state."""
    gen_optimizer_sgd = request.getfixturevalue(gen_optimizer_sgd_)

    # Compute parameter gradients
    gen_optimizer_sgd.zero_grad()
    criterion(model(x), y).backward()

    # Run a single optimizer step (this mutates parameters)
    gen_optimizer_sgd.step()

    # Check for mismatches between expected and actual parameter updates
    # - For vanilla SGD, the expected parameter updates are the corresponding
    #   parameter gradients; the sign depends on the optimization objective.
    # - If the base optimizer is minimizing (resp., maximizing) then the
    #   the expected update is the gradient (resp., negative gradient).
    error_str = f"Mismatch between expected and actual updates ({gen_optimizer_sgd_})"
    for group in gen_optimizer_sgd.param_groups:
        for param in group["params"]:
            if "maximize" not in group or not group["maximize"]:
                expected_param_update = param.grad
            else:
                expected_param_update = -1.0 * param.grad
            actual_param_update = gen_optimizer_sgd.get_param_update(param)
            assert torch.allclose(expected_param_update, actual_param_update), error_str


@jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize(
    "gen_optimizer_sgd_", ["gen_optimizer_sgd_minimize", "gen_optimizer_sgd_maximize"]
)
def test_graph_sanity_after_step(
    model: nn.Module,
    gen_optimizer_sgd_: str,
    criterion: CustomCriterionType,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
    request,
) -> None:
    """Check sanity of computation graph after invoking `step()`."""
    gen_optimizer_sgd = request.getfixturevalue(gen_optimizer_sgd_)

    # Compute parameter gradients
    gen_optimizer_sgd.zero_grad()
    criterion(model(x), y).backward()

    # Run a single optimizer step (this mutates parameters)
    gen_optimizer_sgd.step()

    # Check `requires_grad` for all parameter updates
    error_str = "Unexpected requires_grad value"
    for group in gen_optimizer_sgd.param_groups:
        for param in group["params"]:
            param_update = gen_optimizer_sgd.get_param_update(param)
            assert not param_update.requires_grad, error_str
