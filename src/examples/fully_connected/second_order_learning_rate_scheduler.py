"""Demo second-order learning rate scheduler."""

# flake8: noqa=DCO010
# pylint: disable=invalid-name

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from examples.common import set_seed
from examples.fully_connected import FullyConnected
from learning_rate_utils import SecondOrderLRScheduler


@dataclass
class DemoConfig:
    """
    Dataclass for demo configuration.

    Args:
        num_samples: Number of samples in dataset.
        input_dim: Input dimension.
        noise_std: Output noise standard deviation.
        batch_size: Batch size in training.
        num_epochs: Number of training epochs.
        lr_min: Minimum learning rate to use.
        lr_max: Maximum learning rate to use.
    """

    num_samples: int = 1000
    input_dim: int = 10
    noise_std: float = 0.1
    batch_size: int = 32
    num_epochs: int = 200
    lr_min: float = 0.0
    lr_max: float = 0.1


class SyntheticRegressionDataset(Dataset):
    """
    Synthetic regression dataset.

    Args:
        config: Demo configuration.
    """

    def __init__(self, config: DemoConfig) -> None:
        # Weight and bias for affine function
        self.A = torch.randn(1, config.input_dim)
        self.b = torch.randn(1)

        # Input data
        self.x = torch.randn(config.num_samples, config.input_dim)

        # Output data is affine transformation of input data, plus noise
        self.y = nn.functional.linear(self.x, self.A, self.b)  # pylint: disable=not-callable
        self.y = self.y + (config.noise_std * torch.randn_like(self.y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class Trainer:
    """
    Training loop.

    Args:
        device: Device to use for training (default = None).
        config: Demo configuration (default = None).
    """

    def __init__(self, device: Optional[str] = None, config: Optional[DemoConfig] = None) -> None:
        # Helper function to get current device name
        def get_device() -> str:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get default configuration, if necessary
        self.device = get_device() if device is None else device
        self.config = DemoConfig() if config is None else config

        # Create dataset and dataloader
        self.dataset = SyntheticRegressionDataset(self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        # Make model, loss criterion, and optimizer
        self.model = FullyConnected(self.config.input_dim, [64, 32], 1, 0.0, False).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters())

        # Make second-order learning rate scheduler
        self.scheduler = SecondOrderLRScheduler(
            self.optimizer, -1, self.model, self.criterion, self.config.lr_min, self.config.lr_max
        )

        # Metrics to track in each epoch
        self.train_losses = []
        self.learning_rates = []

    def train(self) -> tuple[list[float], list[float]]:
        """
        Run training loop.
        """
        print(f"Training on {self.device}")

        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0

            for x, y in self.dataloader:
                # Move data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Zero model parameter gradients
                self.optimizer.zero_grad()

                # Run forward pass
                y_hat = self.model(x)

                # Compute loss
                loss = self.criterion(y_hat, y)

                # Run backward pass
                loss.backward()

                # Adjust learning rate(s) in optimizer
                self.scheduler.step(x, y)

                # Adjust model parameters using new learning rate(s)
                self.optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += loss.item()

            # Record metrics for this epoch
            epoch_loss = epoch_loss / len(self.dataloader)
            last_lr = self.scheduler.get_last_lr()[0]

            self.train_losses.append(epoch_loss)
            self.learning_rates.append(last_lr)

            # Console reporting
            print(
                f"epoch {epoch + 1}: "
                f"loss: {epoch_loss:.4f}, "
                f"lr: {self.learning_rates[-1]:.6f}"
            )

        return self.train_losses, self.learning_rates

    def plot_metrics(self, train_losses: list[float], learning_rates: list[float]) -> None:
        """
        Visualize metrics.

        Args:
            train_losses: Average losses for each epoch.
            learning_rates: Learning rates after each epoch.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        # Plot average losses for each epoch
        ax1.plot(train_losses, label="loss", color="blue")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("average losses for each epoch")

        # Plot learning rates after each epoch
        ax2.plot(learning_rates, label="learning rate", color="red")
        ax2.set_xlabel("epochs")
        ax2.set_ylabel("learning rate")
        ax2.set_title("learning rates after each epoch")

        plt.tight_layout()
        plt.show()


def run_demo() -> None:
    """Run demo for a fully-connected neural network."""
    trainer = Trainer()
    train_losses, learning_rates = trainer.train()
    trainer.plot_metrics(train_losses, learning_rates)


if __name__ == "__main__":
    set_seed(11)
    run_demo()
