"""Example: Shallow convolutional neural network (CNN)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from common import set_seed  # pylint: disable=import-error
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torchvision import datasets, transforms
from typeguard import typechecked as typechecker

from learning_rate_utils import loss_per_learning_rate

plt.rcParams["text.usetex"] = True


class ShallowCNN(nn.Module):
    """Shallow CNN based on the PyTorch MNIST example.

    Reference:
        https://github.com/pytorch/examples/blob/main/mnist/main.py
    """

    def __init__(self):  # noqa: DCO010
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b c n m"]) -> Float[Tensor, "b 10"]:
        """Compute CNN output.

        Args:
            x: Input tensor.

        Returns:
            Output tensor containing logits.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def run_demo() -> None:
    """Run loss per learning rate demo."""
    dataset_dir = Path(__file__).resolve().parent / "mnist_data"
    model_filename = dataset_dir / "mnist_cnn.pt"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(dataset_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

    learning_rates = np.linspace(0.0, 1.0, 50)
    losses = np.ndarray((len(learning_rates), 10))

    # Example 1: Untrained model
    # - Input/output data varies in each loop iteration
    # - Model parameters are fixed

    model = ShallowCNN()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters())

    for i in range(losses.shape[-1]):
        x, y = next(iter(train_loader))
        losses[:, i] = loss_per_learning_rate(model, criterion, optimizer, x, y, learning_rates)

    plt.figure()
    plt.plot(learning_rates, losses)
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("Shallow CNN example (untrained)")
    plt.show(block=False)

    # Example 2: Trained model
    # - Input/output data varies in each loop iteration
    # - Model parameters are fixed

    model = ShallowCNN()
    model.load_state_dict(torch.load(model_filename, weights_only=True))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters())

    for i in range(losses.shape[-1]):
        x, y = next(iter(train_loader))
        losses[:, i] = loss_per_learning_rate(model, criterion, optimizer, x, y, learning_rates)

    plt.figure()
    plt.plot(learning_rates, losses)
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("Shallow CNN example (trained)")
    plt.show()


if __name__ == "__main__":
    set_seed(11)
    run_demo()
