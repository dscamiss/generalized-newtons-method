"""Shallow CNN."""

# flake8: noqa=DCO010
# pylint: disable=missing-function-docstring, not-callable

import torch
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker


class ShallowCNN(nn.Module):
    """Shallow CNN based on the PyTorch MNIST example.

    Reference:
        https://github.com/pytorch/examples/blob/main/mnist/main.py
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "b c n m"]) -> Float[Tensor, "b 10"]:
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
