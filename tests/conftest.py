"""Common test fixtures."""

# flake8: noqa=D401

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.examples.common import set_seed
from src.examples.fully_connected import FullyConnected
from src.generalized_newtons_method import GenOptimizer, make_gen_optimizer
from src.generalized_newtons_method.types import Criterion, Optimizer


@pytest.fixture(scope="session", autouse=True)
def setup_session() -> None:
    """Set up for tests."""
    set_seed(11)
    torch.set_default_dtype(torch.float64)


@pytest.fixture(name="input_dim")
def fixture_input_dim() -> int:
    """Input dimension."""
    return 8


@pytest.fixture(name="output_dim")
def fixture_output_dim() -> int:
    """Output dimension."""
    return 4


@pytest.fixture(name="model")
def fixture_model(input_dim: int, output_dim: int) -> nn.Module:
    """Three-layer fully-connected network with ReLU activation."""
    return FullyConnected(input_dim, [16, 32], output_dim).eval()


@pytest.fixture(name="mse")
def fixture_mse() -> Criterion:
    """MSE loss criterion with custom normalization."""

    @jaxtyped(typechecker=typechecker)
    def mse(  # noqa: DCO010
        y_hat: Float[Tensor, "b n"], y: Float[Tensor, "b n"]
    ) -> Float[Tensor, ""]:
        return nn.MSELoss(reduction="sum")(y_hat, y) / (2.0 * y_hat.shape[0])

    return mse


@pytest.fixture(name="sgd_minimize")
def fixture_sgd_minimize(model: nn.Module) -> Optimizer:
    """Vanilla SGD optimizer (minimizing objective)."""
    return torch.optim.SGD(model.parameters())


@pytest.fixture(name="sgd_maximize")
def fixture_sgd_maximize(model: nn.Module) -> Optimizer:
    """Vanilla SGD optimizer (maximizing objective)."""
    return torch.optim.SGD(model.parameters(), maximize=True)


@pytest.fixture(name="gen_sgd_minimize")
def fixture_gen_sgd_minimize(model: nn.Module) -> GenOptimizer:
    """Wrapped vanilla SGD optimizer (minimizing objective)."""
    return make_gen_optimizer(torch.optim.SGD, model.parameters())


@pytest.fixture(name="gen_sgd_maximize")
def fixture_gen_sgd_maximize(model: nn.Module) -> GenOptimizer:
    """Wrapped vanilla SGD optimizer (maximizing objective)."""
    return make_gen_optimizer(torch.optim.SGD, model.parameters(), maximize=True)


@pytest.fixture(name="batch_size")
def fixture_batch_size() -> int:
    """Batch size."""
    return 5


@pytest.fixture(name="x")
@jaxtyped(typechecker=typechecker)
def fixture_x(batch_size: int, input_dim: int) -> Float[Tensor, "b input_dim"]:
    """Input data."""
    return torch.randn(batch_size, input_dim)


@pytest.fixture(name="y")
@jaxtyped(typechecker=typechecker)
def fixture_y(batch_size: int, output_dim: int) -> Float[Tensor, "b output_dim"]:
    """Output data (target)."""
    return torch.randn(batch_size, output_dim)
