"""Tests for second_order_approximation.py."""

# flake8: noqa=D401
# mypy: disable-error-code=no-untyped-def

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.examples.fully_connected import FullyConnected
from src.generalized_newtons_method import GenOptimizer, make_gen_optimizer
from src.generalized_newtons_method.types import Criterion
from src.generalized_newtons_method.utils import second_order_approximation_coeffs


@pytest.fixture(name="difference")
def fixture_difference() -> Criterion:
    """Scalar difference loss criterion."""

    @jaxtyped(typechecker=typechecker)
    def difference(
        y_hat: Float[Tensor, ""], y: Float[Tensor, ""]
    ) -> Float[Tensor, ""]:  # noqa: DCO010
        return (y_hat - y).squeeze()

    return difference


def test_linear_model(difference: Criterion) -> None:
    """
    Test simple linear model.

    For this test, we take:

    - Parametric model f(w) = <w, x>,
    - Loss criterion is the scalar difference function,
    - Optimizer is vanilla SGD, and
    - Batch size is 1.


    The loss-per-learning-rate function is

        g(alpha) = loss(f(w - alpha Df(w)))
                 = loss(f(w - alpha x))
                 = <w - alpha x, x> - y
                 = [f(w) - y] - alpha |x|^2,

    so the approximation coefficients are (f(w) - y, -|x|^2, 0).
    """

    class TestLinear(nn.Module):
        """Simple linear model."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8))

        @jaxtyped(typechecker=typechecker)
        def forward(self, x: Float[Tensor, " n"]) -> Float[Tensor, ""]:
            """Run forward pass."""
            return torch.inner(self.weight, x)

    model = TestLinear().eval()

    # Make wrapped vanilla SGD optimizer
    sgd = make_gen_optimizer(torch.optim.SGD, model.parameters())

    # Set input and target output
    x = torch.randn(model.weight.shape[0])
    y = torch.randn(1).squeeze()

    # Compute network output
    y_hat = model(x)

    # Compute gradients
    sgd.zero_grad()
    loss = difference(y_hat, y)
    loss.backward()

    # Compute parameter updates
    sgd.compute_param_updates()

    # Compute approximation coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(FutureWarning):
        coeffs = second_order_approximation_coeffs(model, difference, sgd, x, y)

    # Check approximation coefficients
    expected_coeff_0 = loss
    expected_coeff_1 = -torch.inner(x, x)
    expected_coeff_2 = torch.as_tensor(0.0)

    assert torch.isclose(coeffs[0], expected_coeff_0), "Error in zeroth-order coefficient"
    assert torch.isclose(coeffs[1], expected_coeff_1), "Error in first-order coefficient"
    assert torch.isclose(coeffs[2], expected_coeff_2), "Error in second-order coefficient"


def test_nonlinear_model(difference: Criterion) -> None:
    """
    Test simple non-linear model.

    For this test, we take:

    - Parametric model f(w) = |w|^2 x for scalar x,
    - Loss criterion is the scalar identity function,
    - Optimizer is vanilla SGD, and
    - Batch size is 1.

    The loss-per-learning-rate function is

        g(alpha) = loss(f(w - alpha Df(w)))
                 = <w - alpha Df(w), w - alpha Df(w)> x - y
                 = (|w|^2 x - y) - alpha 2 <w, Df(x)> x +
                    alpha^2 <Df(w), Df(w)> x
                 = (f(x) - y) - alpha (4 |w|^2 x^2) + alpha^2 (4 |w|^2 x^3).
    """

    class TestNonlinear(nn.Module):
        """Simple non-linear model with scalar inputs."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8))

        @jaxtyped(typechecker=typechecker)
        def forward(self, x: Float[Tensor, ""]) -> Float[Tensor, ""]:
            """Run forward pass."""
            return torch.inner(self.weight, self.weight) * x

    model = TestNonlinear().eval()

    # Make vanilla SGD optimizer
    sgd = make_gen_optimizer(torch.optim.SGD, model.parameters())

    # Set input and target output
    x = torch.randn(1).squeeze()
    y = torch.randn(1).squeeze()

    # Compute network output
    y_hat = model(x)

    # Compute gradients
    sgd.zero_grad()
    loss = difference(y_hat, y)
    loss.backward()

    # Compute parameter updates
    sgd.compute_param_updates()

    # Compute approximation coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(FutureWarning):
        coeffs = second_order_approximation_coeffs(model, difference, sgd, x, y)

    # Check approximation coefficients
    expected_coeff_0 = loss
    expected_coeff_1 = -4.0 * y_hat * x
    expected_coeff_2 = 4.0 * y_hat * x * x

    assert torch.isclose(coeffs[0], expected_coeff_0), "Error in zeroth-order coefficient"
    assert torch.isclose(coeffs[1], expected_coeff_1), "Error in first-order coefficient"
    assert torch.isclose(coeffs[2], expected_coeff_2), "Error in second-order coefficient"


def test_loss_argument(
    model: nn.Module,
    mse: Criterion,
    gen_sgd_minimize: GenOptimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
) -> None:
    """Test behavior with different `loss` argument values."""
    # Alias for brevity
    sgd = gen_sgd_minimize

    # Compute gradients
    sgd.zero_grad()
    loss = mse(model(x), y)
    loss.backward()

    # Compute parameter updates
    gen_sgd_minimize.compute_param_updates()

    # Compute approximation coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(FutureWarning):
        coeffs = second_order_approximation_coeffs(model, mse, sgd, x, y, loss)
        assert coeffs[0] == loss, "Error in loss value"

    with pytest.warns(FutureWarning):
        coeffs = second_order_approximation_coeffs(model, mse, sgd, x, y)
        assert coeffs[0] == loss, "Error in loss value"


@jaxtyped(typechecker=typechecker)
def test_first_order_approximation_coeff(
    model: nn.Module,
    mse: Criterion,
    gen_sgd_minimize: GenOptimizer,
    x: Float[Tensor, "b input_dim"],
    y: Float[Tensor, "b output_dim"],
) -> None:
    """
    Test first-order approximation coefficient.

    This test is only valid for vanilla SGD (minimizing objective).
    """
    # Alias for brevity
    sgd = gen_sgd_minimize

    # Compute gradients
    sgd.zero_grad()
    loss = mse(model(x), y)
    loss.backward()

    # Compute parameter updates
    sgd.compute_param_updates()

    # Compute approximation coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(FutureWarning):
        coeffs = second_order_approximation_coeffs(model, mse, sgd, x, y)

    with torch.no_grad():
        expected_coeff = torch.as_tensor(0.0)
        for param in model.parameters():
            expected_coeff += torch.sum(param.grad * param.grad)

    err_str = "Error in first-order approximation coefficient"
    assert torch.allclose(-expected_coeff, coeffs[1]), err_str


def test_alpha_star_computation(mse: Criterion) -> None:
    """
    Test second-order approximation by computing alpha_*.

    This test derives alpha_* using the approximation coefficients, and then
    compares it to the theoretical value.

    The theoretical value is only valid under the following assumptions:

    - One-layer fully-connected network with ReLU activation,
    - Loss criterion is MSE / (2 * M), where M is the batch size
    - Optimizer is vanilla SGD (minimizing objective), and
    - Batch size is 1.

    For a derivation of the theoretical value of alpha_*, see:
        https://dscamiss.github.io/blog/posts/generalized_newtons_method/
    """
    # Model is f(x) = ReLU(Wx + b)
    model = FullyConnected(8, [], 4).eval()

    # Make wrapped vanilla SGD optimizer
    sgd = make_gen_optimizer(torch.optim.SGD, model.parameters())

    # Set input and target output
    x = torch.randn(1, model.layers[0].in_features)
    y = torch.randn(1, model.layers[0].out_features)

    # Compute gradients
    sgd.zero_grad()
    loss = mse(model(x), y)
    loss.backward()

    # Compute parameter updates
    sgd.compute_param_updates()

    # Get approximation coefficients
    # - Expected PyTorch deprecation warning for `make_functional()`
    with pytest.warns(FutureWarning):
        coeffs = second_order_approximation_coeffs(model, mse, sgd, x, y, loss)

    # Make numerator and denominator terms
    num, den = -coeffs[1], 2.0 * coeffs[2]

    # Sanity check on numerator and denominator terms
    assert num != 0.0, "Unexpected numerator term (0)"
    assert den > 0.0, f"Unexpected denominator term ({den})"

    # Compute actual alpha_*
    alpha_star = num / den

    # Compute expected alpha_* value
    alpha_star_expected = 1.0 / (1.0 + (x[0].norm() ** 2.0))

    # Compare alpha_* values
    err_msg = "Error in alpha_* value"
    assert torch.allclose(alpha_star, alpha_star_expected), err_msg
