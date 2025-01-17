"""Compute loss-per-learning-rate function."""

import copy

import numpy as np
import torch
from jaxtyping import Real, jaxtyped
from numpy.typing import NDArray
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from ..gen_optimizer import GeNOptimizer
from ..types import CriterionType, OptimizerType


@jaxtyped(typechecker=typechecker)
def loss_per_learning_rate(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: OptimizerType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
) -> NDArray:
    """
    Compute loss-per-learning-rate function.

    Given learning rates lr_i, this function computes the loss values after
    running a single optimizer step with learning rate lr_i applied to ALL
    parameter groups.

    This function does not produce meaningful output for more complicated
    optimizer configs, where the learning rate varies by parameter group.

    Args:
        model: Network model (in evaluation mode).
        criterion: Loss criterion.
        optimizer: Optimizer for model parameters.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: List of learning rates (must be non-empty).

    Returns:
        Array of loss values for each learning rate in `learning_rates`.

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
        raise ValueError("learning_rates is empty")

    # Store one loss value for each learning rate
    losses = np.zeros(len(learning_rates))

    # Save model and optimizer states
    model_state_dict = copy.deepcopy(model.state_dict())
    optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    for i, learning_rate in enumerate(learning_rates):
        # Update learning rate in each parameter group
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        # Update parameters
        if isinstance(optimizer, GeNOptimizer):
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
