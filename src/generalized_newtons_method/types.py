"""Custom type definitions."""

from collections.abc import Callable
from typing import Union

import torch
from jaxtyping import Real
from torch import Tensor

_CustomCriterion = Callable[[Real[Tensor, "..."], Real[Tensor, "..."]], Real[Tensor, ""]]
Criterion = Union[torch.nn.modules.loss._Loss, _CustomCriterion]  # pylint: disable=protected-access
Optimizer = torch.optim.Optimizer
