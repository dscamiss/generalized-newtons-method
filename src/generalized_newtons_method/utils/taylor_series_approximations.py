"""Taylor series approximations of the loss-per-learning-rate function.

Note:
    The approximations here are derived under the assumption that we are using
    *standard gradient descent* as our optimizer, with no modifications such
    as momentum, dampening, weight decay, etc.

    To be precise, we assume that the optimizer updates the model parameters
    theta at each step of gradient descent by

        theta := theta - alpha Df(theta),

    where alpha is the (fixed) learning rate, f is the loss function, and
    Df(theta) is the gradient of f, evaluated at theta.
"""

# flake8: noqa=DCO010
# pylint: disable=not-callable

import numpy as np
import torch
from functorch import make_functional
from jaxtyping import Real, jaxtyped
from numpy.typing import NDArray
from torch import Tensor, linalg, nn
from torch.autograd.functional import vhp
from torch.func import functional_call, grad
from typeguard import typechecked as typechecker

from generalized_newtons_method.types import CriterionType

_TensorDict = dict[str, Real[Tensor, "..."]]
_Scalar = Real[Tensor, ""]
_ScalarTwoTuple = tuple[_Scalar, _Scalar]
_ScalarThreeTuple = tuple[_Scalar, _Scalar, _Scalar]


@jaxtyped(typechecker=typechecker)
def norm_of_tensor_dict(tensor_dict: _TensorDict, p: float = 2.0) -> _Scalar:
    """Helper function to sum the norms of each tensor in a dictionary.

    Args:
        tensor_dict: Dictionary containing only tensors.
        ord: Order of the norm (default = 2.0).

    Returns:
        Scalar tensor with 2-norm.
    """
    tensors = tensor_dict.values()
    return sum(linalg.vector_norm(tensor, p) ** 2.0 for tensor in tensors)


@jaxtyped(typechecker=typechecker)
def first_order_approximation_coeffs(
    model: nn.Module, criterion: CriterionType, x: Real[Tensor, "..."], y: Real[Tensor, "..."]
) -> tuple[_ScalarTwoTuple, _TensorDict]:
    """Compute coefficients of first-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
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

        # Compute first-order coefficient
        coeff_1 = norm_of_tensor_dict(grad_params_dict)

    return (coeff_0, -coeff_1), grad_params_dict


@jaxtyped(typechecker=typechecker)
def first_order_approximation(
    model: nn.Module,
    criterion: CriterionType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
) -> NDArray:
    """Evaluate first-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: Learning rates to use for evaluation.

    Returns:
        Array with approximation values.
    """
    coeffs, _ = first_order_approximation_coeffs(model, criterion, x, y)
    coeffs = [coeff.detach().numpy() for coeff in coeffs][::-1]
    return np.poly1d(coeffs)(learning_rates)


@jaxtyped(typechecker=typechecker)
def second_order_approximation_coeffs(
    model: nn.Module, criterion: CriterionType, x: Real[Tensor, "..."], y: Real[Tensor, "..."]
) -> _ScalarThreeTuple:
    """Compute coefficients of second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
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
        coeffs, grad_params_dict = first_order_approximation_coeffs(model, criterion, x, y)
        coeff_2 = torch.as_tensor(0.0)

        # Compute second-order coefficient
        params = tuple(model.parameters())
        grad_params = tuple(grad_params_dict.values())
        _, prod = vhp(parameterized_loss, params, grad_params)

        for i, grad_param in enumerate(grad_params):
            coeff_2 += torch.dot(grad_param.flatten(), prod[i].flatten())

    # Note: Minus was already applied to first-order coefficient
    return (coeffs[0], coeffs[1], coeff_2 / 2.0)


@jaxtyped(typechecker=typechecker)
def second_order_approximation(
    model: nn.Module,
    criterion: CriterionType,
    x: Real[Tensor, "..."],
    y: Real[Tensor, "..."],
    learning_rates: NDArray,
) -> NDArray:
    """Evaluate second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        x: Input tensor.
        y: Output tensor (target).
        learning_rates: Learning rates to use for evaluation.

    Returns:
        Array with approximation values.
    """
    coeffs = second_order_approximation_coeffs(model, criterion, x, y)
    coeffs = [coeff.detach().numpy() for coeff in coeffs][::-1]
    return np.poly1d(coeffs)(learning_rates)
