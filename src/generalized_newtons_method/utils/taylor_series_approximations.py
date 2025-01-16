"""Taylor series approximations of the loss-per-learning-rate function."""

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

from ..types import CriterionType, OptimizerType

_TensorDict = dict[str, Real[Tensor, "..."]]
_Scalar = Real[Tensor, ""]
_ScalarTwoTuple = tuple[_Scalar, _Scalar]
_ScalarThreeTuple = tuple[_Scalar, _Scalar, _Scalar]


@jaxtyped(typechecker=typechecker)
def second_order_approximation_coeffs(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: OptimizerType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    loss: Optional[_Scalar] = None,
) -> _ScalarThreeTuple:
    """Compute coefficients of second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for model parameters.
        x: Input tensor.
        y: Output tensor (target).
        loss: Optional loss value (default = None).

    Returns:
        Tuple of scalar tensors with approximation coefficients.

    Note:
        This function expects gradients to be available.
    """
    # FIXME: Only works with SGD for now

    # Wrapper function for parameter-dependent loss
    # - This version is compatible with `make_functional()`, which is needed
    #   for the call to `torch.autograd.functional.vhp()`.  PyTorch issues a
    #   warning about using `make_functional()`, but there seems to be no
    #   analogue of `torch.autograd.functional.vhp()` which can be used with
    #   `torch.func.functional_call()`.

    with torch.no_grad():
        coeff_0 = criterion(model(x), y) if loss is None else loss.clone()
        coeff_1 = torch.as_tensor(0.0)
        coeff_2 = torch.as_tensor(0.0)

        # Compute first-order coefficient
        for param in model.parameters():
            coeff_1 += torch.sum(param.grad * param.grad)  # Frobenius inner product

        # Compute second-order coefficient
        def parameterized_loss(*params):
            model_func, _ = make_functional(model)
            y_hat = model_func(params, x)
            return criterion(y_hat, y)

        params = tuple(model.parameters())
        param_updates = tuple(
            param.grad for param in model.parameters()
        )  # FIXME: Vanilla SGD-specific
        _, prod = vhp(parameterized_loss, params, param_updates)

        for i, param_update in enumerate(param_updates):
            coeff_2 += torch.sum(param_update * prod[i])  # Frobenius inner product

    return (coeff_0, -coeff_1, coeff_2 / 2.0)


@jaxtyped(typechecker=typechecker)
def second_order_approximation(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: OptimizerType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
    loss: Optional[_Scalar] = None,
) -> NDArray:
    """Evaluate second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for model parameters.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: Learning rates to use for evaluation.

    Returns:
        Array with approximation values.
    """
    coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y, loss)
    coeffs = [coeff.numpy() for coeff in coeffs][::-1]
    return np.poly1d(coeffs)(learning_rates)
