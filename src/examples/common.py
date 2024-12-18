"""Common code used in examples."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch


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
