"""Main module with learning rate-related functions."""

from importlib.metadata import metadata

from learning_rate_utils.loss_per_learning_rate import loss_per_learning_rate

# Dynamically load metadata from pyproject.toml
meta = metadata("learning-rate-utils")

__version__ = meta["Version"]
__description__ = meta["Summary"]

__all__ = ["loss_per_learning_rate"]
