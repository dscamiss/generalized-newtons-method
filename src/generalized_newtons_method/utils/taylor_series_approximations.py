"""Taylor series approximations of the loss-per-learning-rate function."""

# flake8: noqa=DCO010
# pylint: disable=not-callable

import numpy as np
import torch
from functorch import make_functional
from jaxtyping import Real, jaxtyped
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.autograd.functional import vhp
from torch.func import functional_call, grad
from typeguard import typechecked as typechecker

from ..types import CriterionType, OptimizerType

_TensorDict = dict[str, Real[Tensor, "..."]]
_Scalar = Real[Tensor, ""]
_ScalarTwoTuple = tuple[_Scalar, _Scalar]
_ScalarThreeTuple = tuple[_Scalar, _Scalar, _Scalar]


@jaxtyped(typechecker=typechecker)
def get_effective_grad_params(
    optimizer: OptimizerType, grad_params_dict: _TensorDict
) -> _TensorDict:
    """Helper function to get effective gradients of parameters.

    The "effective gradients" are the (possibly) adjusted gradients used by
    the optimizer in each parameter update.  For example, for SGD there is no
    difference between the effective gradients and actual gradients; for Adam,
    the effective gradients are based on moving averages which are updated at
    each gradient descent iteration using the actual gradients.

    Args:
        optimizer: Optimizer.
        grad_params_dict: Dictionary mapping parameter names to parameter
            gradients.

    Returns:
        Dictionary mapping parameter names to effective parameter gradients.

    Raises:
        ValueError: If `optimizer` is not supported.
    """
    if isinstance(optimizer, torch.optim.SGD):
        return grad_params_dict
    raise ValueError("Optimizer is not supported")


@jaxtyped(typechecker=typechecker)
def first_order_approximation_coeffs(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: OptimizerType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
) -> tuple[_ScalarTwoTuple, _TensorDict]:
    """Compute coefficients of first-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for each trainable parameter in `model`.
        x: Input tensor.
        y: Output tensor (target).

    Returns:
        Tuple containing:
            - Tuple of scalar tensors with approximation coefficients.
            - Dictionary with model parameter gradients.  This can be ignored,
              since it is only to avoid code duplication in the second-order
              approximation code.
    """
    # Extract parameters from `model` to pass to `torch.func.functional_call()`
    params_dict = dict(model.named_parameters())

    # Wrapper function for parameter-dependent loss
    def parameterized_loss(params_dict):
        y_hat = functional_call(model, params_dict, (x,))
        return criterion(y_hat, y)

    with torch.no_grad():
        # Polynomial coefficients
        coeff_0 = parameterized_loss(params_dict)
        coeff_1 = torch.as_tensor(0.0)

        # Compute parameter gradients
        grad_params_dict = grad(parameterized_loss)(params_dict)

        # Compute effective parameter gradients
        effective_grad_params_dict = get_effective_grad_params(optimizer, grad_params_dict)

        # Compute first-order coefficient
        for param_name in grad_params_dict:
            grad_param = grad_params_dict[param_name]
            effective_grad_param = effective_grad_params_dict[param_name]
            coeff_1 += torch.sum(grad_param * effective_grad_param)  # Frobenius inner product

    return (coeff_0, -coeff_1), effective_grad_params_dict


@jaxtyped(typechecker=typechecker)
def first_order_approximation(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: OptimizerType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
) -> NDArray:
    """Evaluate first-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for each trainable parameter in `model`.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: Learning rates to use for evaluation.

    Returns:
        Array with approximation values.
    """
    coeffs, _ = first_order_approximation_coeffs(model, criterion, optimizer, x, y)
    coeffs = [coeff.detach().numpy() for coeff in coeffs][::-1]
    return np.poly1d(coeffs)(learning_rates)


@jaxtyped(typechecker=typechecker)
def second_order_approximation_coeffs(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: OptimizerType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
) -> _ScalarThreeTuple:
    """Compute coefficients of second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for each trainable parameter in `model`.
        x: Input tensor.
        y: Output tensor (target).

    Returns:
        Tuple of scalar tensors with approximation coefficients.
    """

    # Wrapper function for parameter-dependent loss
    # - This version is compatible with `make_functional()`, which is needed
    #   for the call to `torch.autograd.functional.vhp()`.  PyTorch issues a
    #   warning about using `make_functional()`, but there seems to be no
    #   analogue of `torch.autograd.functional.vhp()` which can be used with
    #   `torch.func.functional_call()`.
    def parameterized_loss(*params):
        model_func, _ = make_functional(model)
        y_hat = model_func(params, x)
        return criterion(y_hat, y)

    with torch.no_grad():
        coeffs, effective_grad_params_dict = first_order_approximation_coeffs(
            model, criterion, optimizer, x, y
        )
        coeff_2 = torch.as_tensor(0.0)

        # Compute second-order coefficient
        params = tuple(model.parameters())
        effective_grad_params = tuple(effective_grad_params_dict.values())
        _, prod = vhp(parameterized_loss, params, effective_grad_params)

        for i, effective_grad_param in enumerate(effective_grad_params):
            coeff_2 += torch.sum(effective_grad_param * prod[i])  # Frobenius inner product

    # Note: Minus was already applied to first-order coefficient
    return (coeffs[0], coeffs[1], coeff_2 / 2.0)


@jaxtyped(typechecker=typechecker)
def second_order_approximation(
    model: nn.Module,
    criterion: CriterionType,
    optimizer: OptimizerType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
) -> NDArray:
    """Evaluate second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        optimizer: Optimizer for each trainable parameter in `model`.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: Learning rates to use for evaluation.

    Returns:
        Array with approximation values.
    """
    coeffs = second_order_approximation_coeffs(model, criterion, optimizer, x, y)
    coeffs = [coeff.detach().numpy() for coeff in coeffs][::-1]
    return np.poly1d(coeffs)(learning_rates)
