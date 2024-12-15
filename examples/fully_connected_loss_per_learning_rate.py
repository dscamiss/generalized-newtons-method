"""Example: Fully-connected neural network."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import FullyConnected, set_seed  # pylint: disable=import-error
from torch import nn

from learning_rate_utils import loss_per_learning_rate

plt.rcParams["text.usetex"] = True


def run_demo() -> None:
    """Run loss per learning rate demo."""
    learning_rates = list(np.linspace(0.0, 1.0, 100))
    losses = np.ndarray((len(learning_rates), 10))

    # Example: Untrained model
    # - Input/output data varies in each loop iteration
    # - Model parameters are fixed

    batch_size = 32
    input_dim = 4
    hidden_layer_dims = [32]
    output_dim = 8

    model = FullyConnected(input_dim, hidden_layer_dims, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters())

    for i in range(losses.shape[-1]):
        x = 4.0 * torch.randn(batch_size, input_dim)  # Constant factor for larger errors
        y = torch.randn(batch_size, output_dim)
        losses[:, i] = loss_per_learning_rate(model, x, y, criterion, optimizer, learning_rates)

    plt.figure()
    plt.plot(learning_rates, losses)
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("Fully-connected example (untrained)")
    plt.show()


if __name__ == "__main__":
    set_seed(11)
    run_demo()
