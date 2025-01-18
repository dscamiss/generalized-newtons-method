"""Demo second-order approximation to loss-per-learning-rate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from src.examples.common import set_seed
from src.examples.shallow_cnn import ShallowCNN
from src.gen import make_gen_optimizer
from src.gen.utils import loss_per_learning_rate, second_order_approximation


def run_demo():
    """Run demo for a shallow CNN."""
    # Make dataloader for MNIST training data
    dataset_dir = Path(__file__).resolve().parent / "mnist_data"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    # Get test data
    x, y = next(iter(train_loader))

    # Make shallow CNN model
    model = ShallowCNN()

    # Make negative log-likelihood criterion
    criterion = nn.NLLLoss()

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
    fig.suptitle("Loss per learning rate (shallow CNN, untrained)")

    plt.show()


if __name__ == "__main__":
    set_seed(11)
    torch.set_default_dtype(torch.float64)
    plt.rcParams["text.usetex"] = True
    run_demo()
