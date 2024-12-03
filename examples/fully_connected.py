"""Example: Loss per learning rate for a fully-connected neural network."""

import numpy as np
import torch
from common import plot_loss_per_learning_rate, set_seed  # pylint: disable=import-error
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from learning_rate_utils import loss_per_learning_rate


def make_model(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Make fully-connected neural network.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.

    Returns:
        Network model.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim, bias=True),
        nn.ReLU(),
    )


@jaxtyped(typechecker=typechecker)
def make_input_output_data(
    batch_size: int, input_dim: int, output_dim: int
) -> tuple[Float[Tensor, "{batch_size} {input_dim}"], Float[Tensor, "{batch_size} {output_dim}"]]:
    """Make random input-output data.

    Args:
        batch_size: Batch size.
        input_dim: Input dimension.
        output_dim: Output dimension.

    Returns:
        Tuple containing input-output data.
    """
    # Constant factor applied to `x` for larger errors
    x = 4.0 * torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)
    return x, y


def demo() -> None:
    """Demo: Loss per learning rate for a fully-connected neural network."""
    set_seed()

    batch_size = 32
    input_dim = 4
    hidden_dim = 32
    output_dim = 8

    model = make_model(input_dim, hidden_dim, output_dim)
    x, y = make_input_output_data(batch_size, input_dim, output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters())
    learning_rates = list(np.linspace(0.0, 1.0, 100))

    losses = loss_per_learning_rate(model, x, y, criterion, optimizer, learning_rates)
    plot_loss_per_learning_rate(learning_rates, losses)


if __name__ == "__main__":
    demo()
