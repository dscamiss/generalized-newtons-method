"""Demo second-order approximation to loss-per-learning-rate."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import learning_rate_utils as lru
from examples.common import set_seed
from examples.fully_connected.fully_connected import FullyConnected


def run_demo():
    """Run demo for a fully-connected neural network."""
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

    # Make MSE criterion
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

    ax1.plot(learning_rates_macro, lplr_macro, label="loss per learning rate")
    ax1.plot(
        learning_rates_macro, lplr_approx_macro, "--", color="lime", label="second-order approx."
    )
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("loss")
    ax1.set_title("Macro")
    ax1.legend()

    ax2.plot(learning_rates_detail, lplr_detail, label="loss per learning rate")
    ax2.plot(
        learning_rates_detail, lplr_approx_detail, "--", color="lime", label="second-order approx."
    )
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("loss")
    ax2.set_title("Detail near 0")
    ax2.legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle("Loss per learning rate (fully-connected, untrained)")

    plt.show()


if __name__ == "__main__":
    set_seed(11)
    torch.set_default_dtype(torch.float64)
    plt.rcParams["text.usetex"] = True
    run_demo()
