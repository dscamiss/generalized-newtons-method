"""Wrapped optimizer for GeN."""

# flake8: noqa=DCO010

from collections.abc import Callable
from typing import Any, Optional, Type

import torch
from jaxtyping import Real, jaxtyped
from torch import Tensor
from typeguard import typechecked as typechecker

from src.generalized_newtons_method.types import Optimizer

_Closure = Optional[Callable[[], float]]


class GenOptimizer:  # pylint: disable=too-few-public-methods
    """Empty class used for wrapper class identification."""


def make_gen_optimizer(
    base_optimizer_class: Type[Optimizer], *args: Any, **kwargs: dict[str, Any]
) -> GenOptimizer:
    """
    Make wrapped optimizer for GeN that tracks parameter updates.

    Args:
        base_optimizer_class: Base optimizer class.

    Note:
      - In the function `compute_param_updates()`, we compute the current
        parameter updates by calling the base optimizer's `step()`.  The latter
        function computes the current parameter updates AND adjusts parameters.
        The adjustment then needs to be undone, to allow GeN to compute the
        current learning rate as a function of the parameter updates, then
        adjust parameters.  It looks like this inefficiency cannot be improved
        without tweaking the base optimizers.
    """

    class WrappedOptimizer(GenOptimizer, base_optimizer_class):
        """Wrapper class for base optimizer class."""

        def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
            super().__init__(*args, **kwargs)
            self._param_cache: dict[Tensor, Tensor] = {}
            self._param_update_cache: dict[Tensor, Tensor] = {}
            self._param_updates_available = False

        def _refresh_param_cache(self) -> None:
            """Refresh parameter cache with current parameter values."""
            for group in self.param_groups:
                for param in group["params"]:
                    self._param_cache[param] = param.clone().detach()

        def _refresh_param_update_cache(self) -> None:
            """Refresh parameter update cache with current parameter updates."""
            # Refresh parameter cache
            self._refresh_param_cache()

            # Call the base optimizer's `step()`; this adjusts parameters
            super().step()

            # Derive parameter updates
            for group in self.param_groups:
                for param in group["params"]:
                    prev_param = self._param_cache[param]
                    self._param_update_cache[param] = param.clone().detach() - prev_param
                    self._param_update_cache[param] /= -1.0 * group["lr"]

            # Restore parameters
            self._restore_params()

            # Flag current parameter updates are available
            self._param_updates_available = True

        def _restore_params(self) -> None:
            """Restore parameters from parameter cache."""
            with torch.no_grad():
                for group in self.param_groups:
                    for param in group["params"]:
                        param.copy_(self._param_cache[param])

        def compute_param_updates(self) -> None:
            """Compute parameter updates."""
            self._refresh_param_update_cache()

        @jaxtyped(typechecker=typechecker)
        def get_param_update(self, param: Real[Tensor, "..."]) -> Real[Tensor, "..."]:
            """
            Get current parameter update.

            Args:
                - param: Parameter whose update will be returned.

            Returns:
                Current parameter update for `param`.

            Raises:
                ValueError: If current parameter updates are not available.
            """
            if not self._param_updates_available:
                raise ValueError("Parameter updates are not available")

            return self._param_update_cache[param]

        def reset(self) -> None:
            """Reset to uninitialized state."""
            self._param_cache = {}
            self._param_update_cache = {}
            self._param_updates_available = False

        def step(self, closure: _Closure = None, training_step: bool = True) -> None:
            """
            Run optimizer for a single step.

            After invoking this function, we are in the following state:
                - Current parameter updates are not available.
                - Parameters have been updated by the base optimizer.

            Args:
                - closure: Optional closure argument (default = `None`).
                  Details in PyTorch docs for `torch.optim.Optimizer.step()`.
                - training_step: Flag indicating that this is a training step
                  (default = False).  Not used outside of examples, where it
                  is used to avoid code duplication.

            Raises:
                NotImplementedError: If `closure` is not `None`.
            """
            # Sanity check on parameters
            if closure is not None:
                raise NotImplementedError("Closure argument is not supported")

            # Run base optimizer step
            super().step()

            # Flag current parameter updates are not available
            if training_step:
                self._param_updates_available = False

    return WrappedOptimizer(*args, **kwargs)
