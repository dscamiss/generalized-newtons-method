"""Custom type definitions."""

from collections.abc import Callable
from typing import Union

import torch
from jaxtyping import Real
from torch import Tensor

TorchLossType = torch.nn.modules.loss._Loss  # pylint: disable=protected-access
CustomCriterionType = Callable[[Real[Tensor, "..."], Real[Tensor, "..."]], Real[Tensor, ""]]
CriterionType = Union[TorchLossType, CustomCriterionType]
