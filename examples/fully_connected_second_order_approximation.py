"""Example: Fully-connected neural network."""

# flake8: noqa: DCO010
# pylint: disable=import-error

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import FullyConnected, set_seed
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

import learning_rate_utils as lru

_DEFAULT_LEARNING_RATE = 1e-3

plt.rcParams["text.usetex"] = True


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

    # Make MSE criterion with different normalization vs. default
    criterion = nn.MSELoss()

    # Make standard gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters())

    # Compute macro second-order approximation
    learning_rates_macro = np.linspace(0.0, 5.0, 20)
    lplr_macro = lru.loss_per_learning_rate(model, criterion, optimizer, x, y, learning_rates_macro)
    lplr_approx_macro = lru.second_order_approximation(model, criterion, x, y, learning_rates_macro)

    # Compute detailed second-order approximation near zero
    learning_rates_detail = np.linspace(0.0, 0.1, 20)
    lplr_detail = lru.loss_per_learning_rate(
        model, criterion, optimizer, x, y, learning_rates_detail
    )
    lplr_approx_detail = lru.second_order_approximation(
        model, criterion, x, y, learning_rates_detail
    )

    # Make plots of macro and detailed second-order approximations
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(learning_rates_macro, lplr_macro)
    ax1.plot(learning_rates_macro, lplr_approx_macro, "--", color="lime")
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("loss")
    ax1.set_title("Macro")

    ax2.plot(learning_rates_detail, lplr_detail)
    ax2.plot(learning_rates_detail, lplr_approx_detail, "--", color="lime")
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("loss")
    ax2.set_title("Detail near 0")

    fig.tight_layout()
    fig.suptitle("Fully-connected example (untrained)")

    plt.show()


if __name__ == "__main__":
    set_seed(11)
    torch.set_default_dtype(torch.float64)
    run_demo()
