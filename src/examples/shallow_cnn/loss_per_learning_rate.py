"""Demo loss-per-learning-rate."""

# flake8: noqa=DCO010
# pylint: disable=missing-function-docstring, not-callable

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from examples.common import set_seed
from examples.shallow_cnn import ShallowCNN
from generalized_newtons_method.utils import loss_per_learning_rate


def run_demo_untrained(train_loader: torch.utils.data.DataLoader) -> None:
    """Run demo for untrained shallow CNN.

    Args:
        train_loader: Dataloader for MNIST training data.
    """
    learning_rates = np.linspace(0.0, 0.5, 100)
    num_plots = 10
    losses = np.ndarray((len(learning_rates), num_plots))

    # Make shallow CNN model
    model = ShallowCNN()

    # Make negative log-likelihood criterion
    criterion = nn.NLLLoss()

    # Make standard gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters())

    # Model parameters are fixed; load new data for each plot
    iter_train_loader = iter(train_loader)
    for i in range(num_plots):
        x, y = next(iter_train_loader)
        losses[:, i] = loss_per_learning_rate(model, criterion, optimizer, x, y, learning_rates)

    # Plot losses
    fig, ax = plt.subplots()
    ax.plot(learning_rates, losses)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss")
    ax.set_title("Loss per learning rate (shallow CNN, untrained)")

    fig.tight_layout()

    plt.show(block=False)


def run_demo_trained(
    train_loader: torch.utils.data.DataLoader, trained_model_filename: Path
) -> None:
    """Run demo for trained shallow CNN.

    Args:
        train_loader: Dataloader for MNIST training data.
        trained_model_filename: Trained model filename.
    """
    learning_rates = np.linspace(0.0, 0.5, 100)
    num_plots = 10
    losses = np.ndarray((len(learning_rates), num_plots))

    # Make shallow CNN model
    model = ShallowCNN()

    # Load trained weights
    model.load_state_dict(torch.load(trained_model_filename, weights_only=True))

    # Make negative log-likelihood criterion
    criterion = nn.NLLLoss()

    # Make standard gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters())

    # Model parameters are fixed; load new data for each plot
    iter_train_loader = iter(train_loader)
    for i in range(num_plots):
        x, y = next(iter_train_loader)
        losses[:, i] = loss_per_learning_rate(model, criterion, optimizer, x, y, learning_rates)

    # Plot losses
    fig, ax = plt.subplots()
    ax.plot(learning_rates, losses)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss")
    ax.set_title("Loss per learning rate (shallow CNN, trained)")

    fig.tight_layout()

    plt.show()


def run_demo() -> None:
    """Run demo for a shallow CNN."""
    # Make dataloader for MNIST training data
    dataset_dir = Path(__file__).resolve().parent / "mnist_data"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    # Example 1: Untrained model
    run_demo_untrained(train_loader)

    # Example 2: Trained model
    trained_model_filename = dataset_dir / "mnist_cnn.pt"
    run_demo_trained(train_loader, trained_model_filename)


if __name__ == "__main__":
    set_seed(11)
    torch.set_default_dtype(torch.float64)
    plt.rcParams["text.usetex"] = True
    run_demo()
