"""
Compute loss-per-learning-rate function.

This is the function f() that takes a learning rate alpha and returns

    f(alpha) = loss(theta - alpha * update(...)),

where:
    - theta is the lumped model parameters,
    - update(...) is the parameter update prescribed by the optimizer, and
    - loss() is the (parameter-dependent) loss function.

Note that this definition uses a common learning rate for ALL parameters.
"""

import copy
from typing import Union

import numpy as np
import torch
from jaxtyping import Real, jaxtyped
from numpy.typing import NDArray
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from ..gen_optimizer import GenOptimizer
from ..types import Criterion, Optimizer


@jaxtyped(typechecker=typechecker)
def loss_per_learning_rate(
    model: nn.Module,
    criterion: Criterion,
    optimizer: Union[GenOptimizer, Optimizer],
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
) -> NDArray:
    """
    Compute loss-per-learning-rate function.

    Args:
        model: Network model (in evaluation mode).
        criterion: Loss criterion.
        optimizer: Optimizer for model parameters.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: Learning rate inputs.

    Returns:
        Loss values for each learning rate input.

    Raises:
        ValueError: If any arguments are invalid.

    Note:
        This function should not mutate `model` or `optimizer`.
    """
    # Sanity check on `model` argument
    if model.training:
        raise ValueError("Model is not in evaluation mode")

    # Sanity check on `learning_rates` argument
    if len(learning_rates) == 0:
        raise ValueError("learning_rates argument is empty")

    # Store one loss value for each learning rate
    losses = np.zeros(len(learning_rates))

    # Save model and optimizer states
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    for i, learning_rate in enumerate(learning_rates):
        # Update learning rate in each parameter group
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        # Update parameters
        if isinstance(optimizer, GenOptimizer):
            optimizer.step(training_step=False)
        else:
            optimizer.step()

        # Compute loss with updated parameters
        with torch.no_grad():
            losses[i] = criterion(model(x), y).item()

        # Restore model and optimizer states
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    return losses
