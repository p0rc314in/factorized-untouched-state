"""Product-key initial-memory seam for released Sparse Delta Memory."""

from .prior import (
    AdditiveProductMemory,
    additive_projection,
    attach_additive_product_memory,
    initial_memory_modules,
    prior_parameter_count,
)

__all__ = [
    "AdditiveProductMemory",
    "additive_projection",
    "attach_additive_product_memory",
    "initial_memory_modules",
    "prior_parameter_count",
]
