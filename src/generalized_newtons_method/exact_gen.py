"""Exact version of generalized Newton's method."""

# pylint: disable=arguments-differ

from typing import Optional

from jaxtyping import Real, jaxtyped
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from typeguard import typechecked as typechecker

from src.generalized_newtons_method.gen_optimizer import GenOptimizer
from src.generalized_newtons_method.types import Criterion
from src.generalized_newtons_method.utils import second_order_approximation_coeffs


class ExactGen(LRScheduler):
    """Exact version of generalized Newton's method.

    Args:
        optimizer: Optimizer.
        model: Network model.
        criterion: Loss criterion function.
        lr_min: Minimum learning rate to use.
        lr_max: Maximum learning rate to use.

    Raises:
        ValueError: If `optimizer` is not supported.
    """

    _FALLBACK_LR = 1e-3

    def __init__(  # noqa: DCO010
        self,
        optimizer: GenOptimizer,
        model: nn.Module,
        criterion: Criterion,
        lr_min: float,
        lr_max: float,
    ) -> None:
        # Parameter `last_epoch` unused since LR is not epoch/batch-based
        super().__init__(optimizer, -1)

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
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

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
        # Handle initial step (in this case, `x` and `y` are `None`)
        if x is None and y is None:
            lr = self._FALLBACK_LR
        else:
            # Compute parameter updates
            self.optimizer.compute_param_updates()

            # Compute coefficients of second-order approximation to loss-per-learning-rate function
            coeffs = second_order_approximation_coeffs(
                self.model, self.criterion, self.optimizer, x, y
            )
            coeffs = [coeff.item() for coeff in coeffs]

            if coeffs[2] <= 0.0:
                # Approximation is concave --> use default learning rate
                lr = self._FALLBACK_LR
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
