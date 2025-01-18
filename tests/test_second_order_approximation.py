"""Tests for second_order_approximation.py."""

# flake8: noqa=D401

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.gen import GenOptimizer, make_gen_optimizer
from src.gen.types import Criterion
from src.gen.utils import second_order_approximation_coeffs


@pytest.fixture(name="optimizer")
@jaxtyped(typechecker=typechecker)
def fixture_optimizer(model: nn.Module) -> GenOptimizer:
    """Wrapped vanilla SGD optimizer."""
    return make_gen_optimizer(torch.optim.SGD, model.parameters())


@jaxtyped(typechecker=typechecker)
def test_zeroth_order_approximation_coeff(
    model: nn.Module,
    criterion: Criterion,
    optimizer: GenOptimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
) -> None:
    """Test zeroth-order approximation coefficient."""
    # Compute gradients
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # Compute parameter updates
    optimizer.compute_param_updates()

    # Compute coefficients with loss not specified
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(UserWarning):
        coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y)
    assert len(coeffs) == 3, "Unexpected number of coefficients"
    assert coeffs[0] == loss, "Mismatch between actual and expected loss values"

    # Compute coefficients with loss specified
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(UserWarning):
        coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y, loss)
    assert len(coeffs) == 3, "Unexpected number of coefficients"
    assert coeffs[0] == loss, "Mismatch between actual and expected loss values"


@jaxtyped(typechecker=typechecker)
def test_first_order_approximation_coeff(
    model: nn.Module,
    criterion: Criterion,
    optimizer: GenOptimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
) -> None:
    """
    Test first-order approximation coefficient.

    Note: This test is only valid for the vanilla SGD optimizer (minimizing).
    """
    # Compute gradients
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # Compute parameter updates
    optimizer.compute_param_updates()

    # Compute coefficients with loss not specified
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(UserWarning):
        coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y)

    with torch.no_grad():
        expected_coeff = torch.as_tensor(0.0)
        for param in model.parameters():
            expected_coeff += torch.sum(param.grad * param.grad)

    err_str = "Mismatch between actual and expected coefficients"
    assert torch.allclose(-expected_coeff, coeffs[1]), err_str


@jaxtyped(typechecker=typechecker)
def test_alpha_star_computation(
    model: nn.Module,
    criterion: Criterion,
    optimizer: GenOptimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
) -> None:
    """
    Test second-order approximation by computing alpha_*.

    This test derives alpha_* using second-order approximation coefficients,
    and compares it to the theoretical value for this particular case.

    A derivation of the theoretical value of alpha_* is here:
        https://dscamiss.github.io/blog/posts/generalized_newtons_method/
    """
    # Compute gradients
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # Compute parameter updates
    optimizer.compute_param_updates()

    # Get coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(UserWarning):
        coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y, loss)

    # Make alpha_* numerator and denominator terms
    num, den = -coeffs[1], 2.0 * coeffs[2]

    # Sanity check on numerator and denominator terms
    assert num != 0.0, "Unexpected numerator term (0)"
    assert den > 0.0, f"Unexpected denominator term ({den})"

    # Compute actual alpha_* value
    alpha_star = num / den

    # Compute theoretical alpha_* value
    alpha_star_expected = 1.0 / (1.0 + (x[0].norm() ** 2.0))

    # Compare alpha_* values
    err_msg = "Mismatch between actual and expected alpha_* values"
    assert torch.allclose(alpha_star, alpha_star_expected), err_msg
