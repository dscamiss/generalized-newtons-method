"""Example: Fully-connected neural network."""

# flake8: noqa: DCO010
# pylint: disable=import-error

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

import learning_rate_utils as lru

from common import FullyConnected, set_seed

_DEFAULT_LEARNING_RATE = 1e-3


@jaxtyped(typechecker=typechecker)
def compute_alpha_star(coeffs: tuple[Float[Tensor, ""], ...]) -> Float[Tensor, ""]:
    """Helper function to compute alpha_* from approximation coefficients.

    If alpha_* is not well-defined, then this function returns the default
    learning rate `_DEFAULT_LEARNING_RATE`.

    Args:
        coeffs: Second-order approximation coefficients.

    Returns:
        Sclar tensor with alpha_*.
    """
    num, den = -coeffs[1], 2.0 * coeffs[2]
    if den <= 0.0:
        print(f"Using default learning rate since den = {den}")
        alpha_star = torch.as_tensor(_DEFAULT_LEARNING_RATE)
    else:
        alpha_star = num / den
    return alpha_star


def run_demo():
    """Run second-order approximation demo."""
    batch_size = 16
    input_dim = 8
    hidden_layer_dims = [64, 32, 16, 8]
    output_dim = 4
    negative_slope = 0.01

    # Make dummy data
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)

    # Make fully-connected model
    model = FullyConnected(input_dim, hidden_layer_dims, output_dim, negative_slope)

    # Make MSE criterion with different normalization
    @jaxtyped(typechecker=typechecker)
    def criterion(y_hat: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, ""]:
        return nn.MSELoss()(y_hat, y) / 2.0

    # Get second-order approximation coefficients
    coeffs = lru.second_order_approximation(model, criterion, x, y)

    # Compute alpha_* using the approximation coefficients
    alpha_star = compute_alpha_star(coeffs)

    # Set up for plots
    optimizer = torch.optim.SGD(model.parameters())
    lplr_approx_coeffs = [coeff.detach().numpy() for coeff in coeffs][::-1]

    # Second-order approximation vs loss per learning rate (macro)
    learning_rates_macro = list(np.linspace(0.0, 2.0 * alpha_star, 100))
    lplr_macro = lru.loss_per_learning_rate(model, x, y, criterion, optimizer, learning_rates_macro)
    lplr_approx_macro = np.poly1d(lplr_approx_coeffs)(learning_rates_macro)

    # Second-order approximation vs loss per learning rate (detail near 0)
    learning_rates_detail = list(np.linspace(0.0, 0.1, 100))
    lplr_detail = lru.loss_per_learning_rate(
        model, x, y, criterion, optimizer, learning_rates_detail
    )
    lplr_approx_detail = np.poly1d(lplr_approx_coeffs)(learning_rates_detail)

    plt.subplot(1, 2, 1)
    plt.plot(learning_rates_macro, lplr_macro)
    plt.plot(learning_rates_macro, lplr_approx_macro)
    plt.axvline(alpha_star, color="m", linestyle="--")
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("Fully-connected example (untrained)")

    plt.subplot(1, 2, 2)
    plt.plot(learning_rates_detail, lplr_detail)
    plt.plot(learning_rates_detail, lplr_approx_detail)
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("Fully-connected example (untrained, detail near 0)")

    plt.show()

    if batch_size == 1 and not hidden_layer_dims and negative_slope == 0.0:
        # Compare to expected (theoretical) value of alpha_* for this case
        alpha_star_expected = 1.0 / (1.0 + (x[0].norm() ** 2.0))
        print(f"alpha_* actual   = {alpha_star}")
        print(f"alpha_* expected = {alpha_star_expected}")
    else:
        print(f"alpha_* = {alpha_star}")


if __name__ == "__main__":
    set_seed(11)
    torch.set_default_dtype(torch.float64)
    run_demo()
