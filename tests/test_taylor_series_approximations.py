"""Tests for taylor_series_approximations.py."""

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.generalized_newtons_method.types import CustomCriterionType, OptimizerType
from src.generalized_newtons_method.utils import second_order_approximation_coeffs


@pytest.fixture(name="optimizer")
@jaxtyped(typechecker=typechecker)
def fixture_sgd_optimizer(model: nn.Module) -> OptimizerType:
    """Vanilla SGD optimizer."""
    return torch.optim.SGD(model.parameters())


@jaxtyped(typechecker=typechecker)
def test_second_order_approximation_coeffs(
    model: nn.Module,
    criterion: CustomCriterionType,
    optimizer: OptimizerType,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
) -> None:
    """Test second-order approximation.

    Define "alpha_*" to be the learning rate which minimizes the second-order
    Taylor series approximation of the loss-per-learning-rate function.  This
    test computes alpha_* using `second_order_approximation_coeffs()` and
    compares it to the theoretical value of alpha_*.

    A derivation of the theoretical value of alpha_* is here:
        https://dscamiss.github.io/blog/posts/generalized_newtons_method/
    """
    # Get coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(UserWarning):
        coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y)

    # Sanity check on coefficients
    assert len(coeffs) == 3, "Unexpected number of coefficients"

    # Make alpha_* numerator and denominator terms
    num, den = -coeffs[1], 2.0 * coeffs[2]

    # Sanity check on numerator and denominator terms
    assert num != 0.0, "Unexpected numerator term num = {num}"
    assert den > 0.0, "Unexpected denominator term den = {den}"

    # Compute actual alpha_* value
    alpha_star = num / den

    # Compute theoretical alpha_* value
    alpha_star_expected = 1.0 / (1.0 + (x[0].norm() ** 2.0))

    # Compare alpha_* values
    err_msg = "Mismatch between actual and expected alpha_* values"
    assert torch.allclose(alpha_star, alpha_star_expected), err_msg
