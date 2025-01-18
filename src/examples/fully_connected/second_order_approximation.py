"""Demo second-order approximation to loss-per-learning-rate."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.examples.common import set_seed
from src.examples.fully_connected import FullyConnected
from src.gen import make_gen_optimizer
from src.gen.utils import loss_per_learning_rate, second_order_approximation


def run_demo() -> None:
    """Run demo for a fully-connected neural network."""
    batch_size = 16
    input_dim = 8
    hidden_layer_dims = [64, 32, 16, 8]
    output_dim = 4
    negative_slope = 0.01

    # Make dummy data
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim) ** 2.0

    # Make fully-connected model
    # - Note: Model applies ReLU activation at final layer
    model = FullyConnected(input_dim, hidden_layer_dims, output_dim, negative_slope)

    # Make MSE criterion
    criterion = nn.MSELoss()

    # Make vanilla SGD optimizer
    optimizer = make_gen_optimizer(torch.optim.SGD, model.parameters())

    # Ensure model is in evaluation mode (disables dropout etc.)
    model.eval()

    # Compute gradients
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # Compute parameter updates
    optimizer.compute_param_updates()

    # Compute macro second-order approximation
    learning_rates_macro = np.linspace(0.0, 5.0, 100)
    losses_macro = loss_per_learning_rate(model, criterion, optimizer, x, y, learning_rates_macro)
    losses_approx_macro = second_order_approximation(
        model, criterion, optimizer, x, y, learning_rates_macro, loss
    )

    # Compute detailed second-order approximation near zero
    learning_rates_detail = np.linspace(0.0, 0.1, 100)
    losses_detail = loss_per_learning_rate(model, criterion, optimizer, x, y, learning_rates_detail)
    losses_approx_detail = second_order_approximation(
        model, criterion, optimizer, x, y, learning_rates_detail, loss
    )

    # Make plots of macro and detailed second-order approximations
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(learning_rates_macro, losses_macro, label="loss per learning rate")
    ax1.plot(
        learning_rates_macro, losses_approx_macro, "--", color="lime", label="second-order approx."
    )
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("loss")
    ax1.set_title("Macro")
    ax1.legend()

    ax2.plot(learning_rates_detail, losses_detail, label="loss per learning rate")
    ax2.plot(
        learning_rates_detail,
        losses_approx_detail,
        "--",
        color="lime",
        label="second-order approx.",
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
