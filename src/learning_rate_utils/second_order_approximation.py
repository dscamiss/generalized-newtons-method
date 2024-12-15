"""Second-order approximation of loss per learning rate (LPLR)."""

# flake8: noqa=DCO010

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from typeguard import typechecked as typechecker

from learning_rate_utils.loss_per_learning_rate import _CriterionType


@jaxtyped(typechecker=typechecker)
def second_order_approximation(
    model: nn.Module, criterion: _CriterionType, x: Float[Tensor, "..."], y: Float[Tensor, "..."]
) -> tuple[Float[Tensor, ""], ...]:
    """Compute second-order Taylor series approximation of LPLR.

    This assumes that the optimizer is *standard gradient descent*, so that

        theta_{t+1} := theta_t - alpha * Df(theta_t),

    where

        * theta_t is the model parameter vector at step t,
        * alpha is the learning rate, and
        * Df(theta_t) is the gradient of f evaluated at theta_t.

    Args:
        model: Network model.
        criterion: Loss criterion function.
        x: Input tensor.
        y: Output tensor (expected).

    Returns:
        Tuple containing scalar tensors with polynomial coefficients.
    """
    # Extract parameters from `model` to create the functional version
    params = dict(model.named_parameters())

    # Create wrapped loss function, parameterized by `params`
    def parameterized_loss(params):
        y_hat = torch.func.functional_call(model, params, (x,))
        return criterion(y_hat, y)

    # Storage for polynomial coefficients
    coeff_0 = parameterized_loss(params)
    coeff_1 = torch.as_tensor(0.0)
    coeff_2 = torch.as_tensor(0.0)

    # Compute parameter gradients and Hessian
    grad_params = torch.func.grad(parameterized_loss)(params)
    hess_params = torch.func.hessian(parameterized_loss)(params)

    # Compute first-order and second-order coefficients
    with torch.no_grad():
        for i in grad_params.keys():
            grad_param_i = grad_params[i].flatten()
            coeff_1 += grad_param_i.norm() ** 2.0
            for j in grad_params.keys():
                grad_param_j = grad_params[j].flatten()
                hess_ij = hess_params[i][j].reshape(grad_param_i.shape[0], grad_param_j.shape[0])
                coeff_2 += grad_param_i @ hess_ij @ grad_param_j

    return (coeff_0, -coeff_1, coeff_2 / 2.0)
