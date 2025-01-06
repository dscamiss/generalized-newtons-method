"""Exact version of generalized Newton's method."""

# pylint: disable=arguments-differ

from typing import Optional

import torch
from jaxtyping import Real, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from .types import CriterionType, OptimizerType
from .utils import second_order_approximation_coeffs


@jaxtyped(typechecker=typechecker)
def is_vanilla_sgd(optimizer: OptimizerType) -> bool:
    """Check if SGD optimizer is vanilla SGD.

    Args:
        optimizer: Optimizer.

    Returns:
        True iff `optimizer` is vanilla SGD (no momentum, dampening, etc.)
    """
    for param_group in optimizer.param_groups:
        if (
            param_group["momentum"]
            or param_group["dampening"]
            or param_group["weight_decay"]
            or param_group["nesterov"]
        ):
            return False
    return True


class ExactGeNLR(LRScheduler):
    """Exact version of generalized Newton's method.

    Args:
        optimizer: Optimizer.
        last_epoch: Number of last epoch.
        model: Network model.
        criterion: Loss criterion function.
        lr_min: Minimum learning rate to use.
        lr_max: Maximum learning rate to use.

    Raises:
        ValueError: If `optimizer` is not supported.
    """

    _DEFAULT_LR = 1e-3

    def __init__(  # noqa: DCO010
        self,
        optimizer: OptimizerType,
        last_epoch: int,
        model: nn.Module,
        criterion: CriterionType,
        lr_min: float,
        lr_max: float,
    ) -> None:
        # Sanity checks on optimizer
        if isinstance(optimizer, torch.optim.SGD):
            if not is_vanilla_sgd(optimizer):
                raise ValueError("Non-vanilla SGD is not supported")
        else:
            raise ValueError("Optimizer type is not supported")

        super().__init__(optimizer, last_epoch)

        self.model = model
        self.criterion = criterion
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_lrs = self.base_lrs.copy()

    # Pylint complains that redefinition of step() has a different signature
    @jaxtyped(typechecker=typechecker)
    def step(  # pylint: disable=arguments-renamed
        self, x: Optional[Real[Tensor, "..."]] = None, y: Optional[Real[Tensor, "..."]] = None
    ) -> list[float]:
        """Update learning rate(s) in the optimizer.

        Args:
            x: Input tensor.
            y: Output tensor (target).

        Returns:
            List of learning rates for each parameter group.
        """
        lrs = self.get_lr(x, y)

        # Update learning rates in the optimizer
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

        return lrs

    @jaxtyped(typechecker=typechecker)
    def get_lr(
        self, x: Optional[Real[Tensor, "..."]] = None, y: Optional[Real[Tensor, "..."]] = None
    ) -> list[float]:
        """Compute learning rate(s) for a particular batch.

        Args:
            x: Input tensor.
            y: Output tensor (target).

        Returns:
            List of learning rates for each parameter group.
        """
        # Handle initial step (in this case, `x` and `y` are not available)
        if x is None and y is None:
            lr = self._DEFAULT_LR
        else:
            # Get coefficients of second-order approximation to LPLR
            coeffs = second_order_approximation_coeffs(
                self.model, self.criterion, self.optimizer, x, y
            )
            coeffs = [coeff.item() for coeff in coeffs]

            if coeffs[2] <= 0.0:
                # Approximation is concave --> use default learning rate
                lr = self._DEFAULT_LR
            else:
                # Approximation is convex --> use approximate minimizer
                alpha_star = -coeffs[1] / (2.0 * coeffs[2])
                lr = min(self.lr_max, max(alpha_star, self.lr_min))

        # Update current learning rate(s)
        num_groups = len(self.optimizer.param_groups)
        self.current_lrs = [lr for _ in range(num_groups)]

        return self.current_lrs

    def get_last_lr(self) -> list[float]:  # noqa
        return self.current_lrs
