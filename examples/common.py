"""Common code used in examples."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class FullyConnected(nn.Module):
    """Fully-connected neural network.

    Args:
        input_dim: Input dimension.
        hidden_layer_dims: Hidden layer dimensions.
        output_dim: Output dimension.
        negative_slope: Negative slope for leaky ReLU (default = 0.0).
    """

    def __init__(  # noqa: DCO010
        self,
        input_dim: int,
        hidden_layer_dims: list[int],
        output_dim: int,
        negative_slope: float = 0.0,
    ):
        super().__init__()

        layers = []

        if not hidden_layer_dims:
            # Edge case: No hidden layers
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LeakyReLU(negative_slope))
        else:
            # Generic case: At least one hidden layer
            layers = []
            layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
            layers.append(nn.LeakyReLU(negative_slope))

            for i in range(1, len(hidden_layer_dims)):
                layers.append(nn.Linear(hidden_layer_dims[i - 1], hidden_layer_dims[i]))
                layers.append(nn.LeakyReLU(negative_slope))

            layers.append(nn.Linear(hidden_layer_dims[-1], output_dim))
            layers.append(nn.LeakyReLU(negative_slope))

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


def set_seed(seed: int) -> None:
    """Set random seeds etc. to attempt reproducibility.

    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss_per_learning_rate(
    learning_rates: list[float],
    losses: list[float],
) -> None:
    """Plot loss per learning rate.

    Args:
        learning_rates: List of learning rates.
        losses: List of losses.
    """
    _, ax = plt.subplots()
    ax.plot(learning_rates, losses, "-")
    ax.set(xlabel=r"learning rate", ylabel="loss")
    plt.show()
