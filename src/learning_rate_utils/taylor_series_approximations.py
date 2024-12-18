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
from jaxtyping import Real, jaxtyped
from numpy.typing import NDArray
from torch import Tensor, linalg, nn
from typeguard import typechecked as typechecker

from learning_rate_utils.types import CriterionType


@jaxtyped(typechecker=typechecker)
def norm_of_tensor_dict(
    tensor_dict: dict[str, Real[Tensor, "..."]], p: float = 2.0
) -> Real[Tensor, ""]:
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
) -> tuple[Real[Tensor, ""], ...]:
    """Compute coefficients of first-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        x: Input tensor.
        y: Output tensor (target).

    Returns:
        Tuple containing scalar tensors with polynomial coefficients.
    """
    # Extract parameters from `model` to pass to `torch.func.functional_call`
    params_dict = dict(model.named_parameters())

    # Wrapper function for parameter-dependent loss
    def parameterized_loss(params_dict):
        y_hat = torch.func.functional_call(model, params_dict, (x,))
        return criterion(y_hat, y)

    with torch.no_grad():
        # Polynomial coefficients
        coeff_0 = parameterized_loss(params_dict)
        coeff_1 = torch.as_tensor(0.0)

        # Compute parameter gradients
        grad_params_dict = torch.func.grad(parameterized_loss)(params_dict)

        # Compute first-order coefficient
        coeff_1 = norm_of_tensor_dict(grad_params_dict)

    return (coeff_0, -coeff_1)


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
    coeffs = first_order_approximation_coeffs(model, criterion, x, y)
    coeffs = [coeff.detach().numpy() for coeff in coeffs][::-1]
    return np.poly1d(coeffs)(learning_rates)


@jaxtyped(typechecker=typechecker)
def second_order_approximation_coeffs(
    model: nn.Module, criterion: CriterionType, x: Real[Tensor, "..."], y: Real[Tensor, "..."]
) -> tuple[Real[Tensor, ""], ...]:
    """Compute coefficients of second-order Taylor series approximation.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        x: Input tensor.
        y: Output tensor (target).

    Returns:
        Tuple containing scalar tensors with polynomial coefficients.
    """
    # Extract parameters from `model` to pass to `torch.func.functional_call`
    params_dict = dict(model.named_parameters())

    # Wrapper function for parameter-dependent loss
    def parameterized_loss(params_dict):
        y_hat = torch.func.functional_call(model, params_dict, (x,))
        return criterion(y_hat, y)

    with torch.no_grad():
        # Polynomial coefficients
        coeff_0 = parameterized_loss(params_dict)
        coeff_1 = torch.as_tensor(0.0)
        coeff_2 = torch.as_tensor(0.0)

        # Compute parameter gradients and Hessian
        grad_params_dict = torch.func.grad(parameterized_loss)(params_dict)
        hess_params_dict = torch.func.hessian(parameterized_loss)(params_dict)

        # Compute first-order coefficient
        coeff_1 = norm_of_tensor_dict(grad_params_dict)

        # Compute second-order coefficient
        # - TODO: Can we make the Hessian-vector product more efficient?  For
        #         example using `torch.autograd.functional.hvp` or similar...
        for grad_param_name_1, grad_param_1 in grad_params_dict.items():
            grad_param_1 = grad_param_1.flatten()
            for grad_param_name_2, grad_param_2 in grad_params_dict.items():
                grad_param_2 = grad_param_2.flatten()
                hess_block = hess_params_dict[grad_param_name_1][grad_param_name_2]
                hess_block = hess_block.reshape(grad_param_1.numel(), grad_param_2.numel())
                coeff_2 += torch.dot(grad_param_1, hess_block @ grad_param_2)

    return (coeff_0, -coeff_1, coeff_2 / 2.0)


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
