"""Second-order approximation of the loss-per-learning-rate function."""

# flake8: noqa=DCO010
# pylint: disable=not-callable

from typing import Optional

import numpy as np
import torch
from functorch import make_functional
from jaxtyping import Real, jaxtyped
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.autograd.functional import vhp
from typeguard import typechecked as typechecker

from ..gen_optimizer import GeNOptimizer
from ..types import CriterionType

_TensorDict = dict[str, Real[Tensor, "..."]]
_Scalar = Real[Tensor, ""]
_ScalarTwoTuple = tuple[_Scalar, _Scalar]
_ScalarThreeTuple = tuple[_Scalar, _Scalar, _Scalar]


@jaxtyped(typechecker=typechecker)
def second_order_approximation_coeffs(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: GeNOptimizer,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    loss: Optional[_Scalar] = None,
) -> _ScalarThreeTuple:
    """Compute second-order approximation coefficients.

    These are the coefficients of the second-order Taylor series approximation
    of the loss-per-learning-rate function, evaluated at zero.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for model parameters.
        x: Input tensor.
        y: Output tensor (target).
        loss: Optional loss value (default = `None`).

    Returns:
        Tuple with approximation coefficients.

    Note:
        Output entry `i` is the `i`th-order approximation coefficient.
    """

    with torch.no_grad():
        # Compute zeroth-order approximation coefficient
        coeff_0 = criterion(model(x), y) if loss is None else loss.clone()

        # Compute first-order approximation coefficient
        coeff_1 = torch.as_tensor(0.0)
        for param in model.parameters():
            coeff_1 += torch.sum(param.grad * param.grad)

        # Wrapper function for parameter-dependent loss
        # - This wrapper is compatible with `make_functional()`, which is
        #   needed for the call to `torch.autograd.functional.vhp()`.  PyTorch
        #   issues a warning about using `make_functional()`, but there seems
        #   to be no analogue of `torch.autograd.functional.vhp()` which can
        #   be used with the alternative `torch.func.functional_call()`.
        # - TODO: Investigate this further... alternatives to `vhp()`?
        def parameterized_loss(*params):
            model_func, _ = make_functional(model)
            y_hat = model_func(params, x)
            return criterion(y_hat, y)

        # Compute second-order approximation coefficient
        # - This is sum_{i} theta_i^t H theta_i, where H is the Hessian
        # - This is slow since it materializes Hessian-vector products
        coeff_2 = torch.as_tensor(0.0)
        params = tuple(model.parameters())
        param_updates = tuple(optimizer.get_param_update(param) for param in params)
        _, prod = vhp(parameterized_loss, params, param_updates)

        for i, param_update in enumerate(param_updates):
            coeff_2 += torch.sum(param_update * prod[i])

    return (coeff_0, -coeff_1, coeff_2 / 2.0)


@jaxtyped(typechecker=typechecker)
def second_order_approximation(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: GeNOptimizer,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
    loss: Optional[_Scalar] = None,
) -> NDArray:
    """Evaluate second-order approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for model parameters.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: Learning rate inputs.

    Returns:
        Second-order approximation values for each learning rate input.
    """
    coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y, loss)
    coeffs = [coeff.numpy() for coeff in coeffs][::-1]
    return np.poly1d(coeffs)(learning_rates)
