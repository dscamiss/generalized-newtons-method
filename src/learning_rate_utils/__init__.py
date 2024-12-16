"""Main module."""

from learning_rate_utils.loss_per_learning_rate import loss_per_learning_rate
from learning_rate_utils.taylor_series_approximations import (
    first_order_approximation,
    first_order_approximation_coeffs,
    second_order_approximation,
    second_order_approximation_coeffs,
)

__all__ = [
    "loss_per_learning_rate",
    "first_order_approximation",
    "first_order_approximation_coeffs",
    "second_order_approximation",
    "second_order_approximation_coeffs",
]
