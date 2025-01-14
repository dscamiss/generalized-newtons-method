"""Wrapped optimizer that tracks parameter updates for GeN."""

# flake8: noqa=DCO010
# pylint: disable=C0103
# ruff: noqa: F821  <-- ruff complains about forward reference

from collections.abc import Callable
from typing import Any, Optional, Type

from jaxtyping import Real, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker

from src.generalized_newtons_method.types import OptimizerType

_StepClosureType = Optional[Callable[[], float]]


@jaxtyped(typechecker=typechecker)
def gen_optimizer(
    base_optimizer_class: Type[OptimizerType], *args, **kwargs
) -> "_OptimizerWithParamUpdateTracking":
    """
    Wrapper learning rate scheduler with parameter update tracking.
    """

    class _OptimizerWithParamUpdateTracking(base_optimizer_class):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._param_cache = {}
            self._param_updates_available = False

        @jaxtyped(typechecker=typechecker)
        def get_param_update(self, param: Real[Tensor, "..."]) -> Real[Tensor, "..."]:
            """
            Get current parameter update.

            Args:
                - param: Parameter whose update will be returned.

            Returns:
                The current parameter update for `param`.

            Raises:
                ValueError: If parameter updates are not available, i.e., if
                    `step()` has not been invoked before `get_param_update()`.
            """
            if not self._param_updates_available:
                raise ValueError("Updates are not available.")

            return self._param_cache[param]

        def _refresh_param_cache(self) -> None:
            """
            Refresh parameter cache with current parameter values.
            """
            for group in self.param_groups:
                for param in group["params"]:
                    self._param_cache[param] = param.clone().detach()

        def reset_param_cache(self) -> None:
            """
            Reset parameter cache to uninitialized state; used in test code.
            """
            self._param_cache = {}
            self._param_updates_available = False

        @jaxtyped(typechecker=typechecker)
        def step(self, closure: _StepClosureType = None) -> Optional[float]:
            """
            Run a single optimizer step.

            Args:
                - closure: Optional closure passed to `super().step()`, see
                    PyTorch docs for `torch.optim.Optimizer.step()` for more
                    (default = `None`).
            """
            self._refresh_param_cache()

            # Call base optimizer `step()`.
            super().step(closure)

            # Derive parameter updates
            for group in self.param_groups:
                for param in group["params"]:
                    prev_param = self._param_cache[param]
                    self._param_cache[param] = param.clone().detach() - prev_param
                    self._param_cache[param] /= -1.0 * group["lr"]

            self._param_updates_available = True

    return _OptimizerWithParamUpdateTracking(*args, **kwargs)
