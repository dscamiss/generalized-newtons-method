"""Example: Fully-connected neural network (FCNN)."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import set_seed  # pylint: disable=import-error
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from learning_rate_utils import loss_per_learning_rate

plt.rcParams["text.usetex"] = True


class FullyConnected(nn.Module):
    """Fully-connected neural network.

    Args:
        input_dim: Input dimension.
        hidden_layer_dims: Hidden layer dimensions.
        output_dim: Output dimension.
    """

    def __init__(  # noqa: DCO010
        self, input_dim: int, hidden_layer_dims: list[int], output_dim: int
    ):
        super().__init__()

        layers = []

        if not hidden_layer_dims:
            # Edge case: No hidden layers
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LeakyReLU())
        else:
            # Generic case: At least one hidden layer
            layers = []
            layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
            layers.append(nn.LeakyReLU())

            for i in range(1, len(hidden_layer_dims)):
                layers.append(nn.Linear(hidden_layer_dims[i - 1], hidden_layer_dims[i]))
                layers.append(nn.LeakyReLU())

            layers.append(nn.Linear(hidden_layer_dims[-1], output_dim))
            layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*layers)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b input_dim"]) -> Float[Tensor, "b output_dim"]:
        """Compute network output.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.layers(x)


def run_demo() -> None:
    """Run loss per learning rate demo."""
    set_seed()

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
    run_demo()
