"""Custom type definitions."""

from collections.abc import Callable
from typing import Union

import torch
from jaxtyping import Float, Integer
from torch import Tensor

OutputDataType = Union[Float[Tensor, "b ..."], Integer[Tensor, "b ..."]]
TorchLossType = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
CustomCriterionType = Callable[[Float[Tensor, "..."], Float[Tensor, "..."]], Float[Tensor, ""]]
CriterionType = Union[TorchLossType, CustomCriterionType]
