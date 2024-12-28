"""Tests for taylor_series_approximations.py."""

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from examples.common import set_seed
from examples.fully_connected.fully_connected import FullyConnected
from learning_rate_utils import second_order_approximation_coeffs
from learning_rate_utils.types import CustomCriterionType


@pytest.fixture(scope="session", autouse=True)
@jaxtyped(typechecker=typechecker)
def setup_session() -> None:
    """Set up for tests."""
    set_seed(11)
    torch.set_default_dtype(torch.float64)


@pytest.fixture(name="input_dim")
@jaxtyped(typechecker=typechecker)
def fixture_input_dim() -> int:
    """Input dimension."""
    return 8


@pytest.fixture(name="output_dim")
@jaxtyped(typechecker=typechecker)
def fixture_output_dim() -> int:
    """Output size for dummy data."""
    return 4


@pytest.fixture(name="model")
@jaxtyped(typechecker=typechecker)
def fixture_model(input_dim: int, output_dim: int) -> nn.Module:
    """Fully-connected network with one layer and ReLU activation."""
    return FullyConnected(input_dim, [], output_dim)


@pytest.fixture(name="criterion")
@jaxtyped(typechecker=typechecker)
def fixture_criterion(output_dim: int) -> CustomCriterionType:  # pylint: disable=unused-argument
    """Criterion `nn.MSELoss()` with custom normalization."""

    @jaxtyped(typechecker=typechecker)
    def criterion(  # noqa
        y_hat: Float[Tensor, "b output_dim"], y: Float[Tensor, "b output_dim"]
    ) -> Float[Tensor, ""]:
        return nn.MSELoss(reduction="sum")(y_hat, y) / (2.0 * y_hat.shape[0])

    return criterion


@pytest.fixture(name="batch_size")
@jaxtyped(typechecker=typechecker)
def fixture_batch_size() -> int:
    """Batch size for dummy data."""
    return 1


@pytest.fixture(name="x")
@jaxtyped(typechecker=typechecker)
def fixture_x(batch_size: int, input_dim: int) -> Float[Tensor, "b input_dim"]:
    """Input data."""
    return torch.randn(batch_size, input_dim)


@pytest.fixture(name="y")
@jaxtyped(typechecker=typechecker)
def fixture_y(batch_size: int, output_dim: int) -> Float[Tensor, "b output_dim"]:
    """Output data (target)."""
    return torch.randn(batch_size, output_dim)


@jaxtyped(typechecker=typechecker)
def test_second_order_approximation_coeffs(
    model: nn.Module,
    criterion: CustomCriterionType,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
) -> None:
    """Test second-order approximation.

    Define "alpha_*" to be the learning rate which minimizes the second-order
    Taylor series approximation of the loss-per-learning-rate function.  This
    test computes alpha_* using `second_order_approximation_coeffs()` and
    compares it to the theoretical value of alpha_*.

    A derivation of the theoretical value of alpha_* is here:
        https://dscamiss.github.io/blog/posts/learning_rates_one_layer/
    """
    # Get coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(UserWarning):
        coeffs = second_order_approximation_coeffs(model, criterion, x, y)

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
