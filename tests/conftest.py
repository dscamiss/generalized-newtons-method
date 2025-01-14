"""Common test fixtures."""

import pytest
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from src.examples.common import set_seed
from src.examples.fully_connected import FullyConnected
from src.generalized_newtons_method.types import CustomCriterionType


@pytest.fixture(scope="session", autouse=True)
@jaxtyped(typechecker=typechecker)
def setup_session() -> None:
    """Set up for tests."""
    set_seed(11)
    torch.set_default_dtype(torch.float64)


@pytest.fixture(name="input_dim")
@jaxtyped(typechecker=typechecker)
def fixture_input_dim() -> int:
    """Input dimension."""
    return 8


@pytest.fixture(name="output_dim")
@jaxtyped(typechecker=typechecker)
def fixture_output_dim() -> int:
    """Output dimension."""
    return 4


@pytest.fixture(name="model")
@jaxtyped(typechecker=typechecker)
def fixture_model(input_dim: int, output_dim: int) -> nn.Module:
    """Fully-connected network with one layer and ReLU activation."""
    return FullyConnected(input_dim, [], output_dim)


@pytest.fixture(name="criterion")
@jaxtyped(typechecker=typechecker)
def fixture_criterion(output_dim: int) -> CustomCriterionType:  # pylint: disable=unused-argument
    """Criterion `nn.MSELoss()` with custom normalization."""

    @jaxtyped(typechecker=typechecker)
    def criterion(  # noqa: DCO010
        y_hat: Float[Tensor, "b output_dim"], y: Float[Tensor, "b output_dim"]
    ) -> Float[Tensor, ""]:
        return nn.MSELoss(reduction="sum")(y_hat, y) / (2.0 * y_hat.shape[0])

    return criterion


@pytest.fixture(name="batch_size")
@jaxtyped(typechecker=typechecker)
def fixture_batch_size() -> int:
    """Batch size for dummy data."""
    return 1


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
