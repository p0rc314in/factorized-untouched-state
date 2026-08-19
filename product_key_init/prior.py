"""Product-key factorization of SDM's learned initial memory parameter."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch
from torch import nn
from torch.nn.utils import parametrize


INITIAL_STATE_MODES = ("zero", "product", "full")


class AdditiveProductMemory(nn.Module):
    """Expose two product-key factor tables as one full SDM memory table.

    PyTorch's parametrization protocol stores the tensors returned by
    ``right_inverse`` as the only trainable originals. The released SDM layer
    continues to read ``mixer.memory`` with its usual shape.
    """

    def __init__(self, *, heads: int, slots_per_head: int, value_width: int) -> None:
        super().__init__()
        codebook_size = math.isqrt(slots_per_head)
        if heads <= 0 or value_width <= 0:
            raise ValueError("heads and value_width must be positive")
        if codebook_size * codebook_size != slots_per_head:
            raise ValueError("slots_per_head must be a perfect square")
        self.heads = heads
        self.slots_per_head = slots_per_head
        self.codebook_size = codebook_size
        self.value_width = value_width

    def _table(self, value: torch.Tensor) -> torch.Tensor:
        expected = (self.heads * self.slots_per_head, self.value_width)
        if tuple(value.shape) != expected:
            raise ValueError(f"initial memory shape is {tuple(value.shape)}, expected {expected}")
        return value.reshape(
            self.heads,
            self.codebook_size,
            self.codebook_size,
            self.value_width,
        )

    def forward(self, row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
        expected = (self.heads, self.codebook_size, self.value_width)
        if tuple(row.shape) != expected or tuple(column.shape) != expected:
            raise ValueError("product-key factor shape changed")
        return (row[:, :, None, :] + column[:, None, :, :]).reshape(
            self.heads * self.slots_per_head,
            self.value_width,
        )

    def right_inverse(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the least-squares additive factors for a full table."""

        table = self._table(value)
        grand = table.mean(dim=(1, 2), keepdim=True)
        row = table.mean(dim=2) - 0.5 * grand.squeeze(2)
        column = table.mean(dim=1) - 0.5 * grand.squeeze(1)
        return row, column


def attach_additive_product_memory(mixer: nn.Module) -> AdditiveProductMemory:
    """Replace only a released SDM layer's memory parameter representation."""

    memory = getattr(mixer, "memory", None)
    if not isinstance(memory, nn.Parameter):
        raise ValueError("product initial state requires SDM learned memory")
    if parametrize.is_parametrized(mixer, "memory"):
        raise ValueError("SDM memory is already parametrized")
    heads = int(mixer.num_heads)
    slots_per_head = int(mixer.slots_per_head)
    value_width = int(memory.shape[-1])
    representation = AdditiveProductMemory(
        heads=heads,
        slots_per_head=slots_per_head,
        value_width=value_width,
    )
    parametrize.register_parametrization(
        mixer,
        "memory",
        representation,
        unsafe=True,
    )
    originals = mixer.parametrizations.memory
    originals.original0._sdm_memory_bank = True
    originals.original1._sdm_memory_bank = True
    mixer._initial_state_mode = "product"
    return representation


def product_factors(mixer: nn.Module) -> tuple[nn.Parameter, nn.Parameter]:
    if not parametrize.is_parametrized(mixer, "memory"):
        raise ValueError("SDM memory is not product-factorized")
    originals = mixer.parametrizations.memory
    return originals.original0, originals.original1


def initialize_product_factors(
    mixer: nn.Module,
    *,
    full_table_std: float,
) -> None:
    """Match the target full-table variance under an additive two-factor sum."""

    row, column = product_factors(mixer)
    factor_std = full_table_std / math.sqrt(2.0)
    with torch.no_grad():
        for factor in (row, column):
            nn.init.trunc_normal_(
                factor,
                mean=0.0,
                std=factor_std,
                a=-3.0 * factor_std,
                b=3.0 * factor_std,
            )


def initialize_full_memory(
    mixer: nn.Module,
    *,
    full_table_std: float,
) -> None:
    """Initialize the full prior with the distribution used by released SDM."""

    memory = getattr(mixer, "memory", None)
    if not isinstance(memory, nn.Parameter):
        raise ValueError("full initial state requires SDM learned memory")
    if parametrize.is_parametrized(mixer, "memory"):
        raise ValueError("full initial state cannot use a parametrized memory")
    with torch.no_grad():
        nn.init.trunc_normal_(
            memory,
            mean=0.0,
            std=full_table_std,
            a=-3.0 * full_table_std,
            b=3.0 * full_table_std,
        )


def additive_projection(
    memory: torch.Tensor,
    *,
    heads: int,
    slots_per_head: int,
) -> torch.Tensor:
    """Orthogonally project a full table onto the additive product-key space."""

    codebook_size = math.isqrt(slots_per_head)
    if codebook_size * codebook_size != slots_per_head:
        raise ValueError("slots_per_head must be a perfect square")
    if memory.ndim != 2 or memory.shape[0] != heads * slots_per_head:
        raise ValueError("full memory shape does not match its head geometry")
    table = memory.reshape(heads, codebook_size, codebook_size, memory.shape[-1])
    row_mean = table.mean(dim=2, keepdim=True)
    column_mean = table.mean(dim=1, keepdim=True)
    grand = table.mean(dim=(1, 2), keepdim=True)
    return (row_mean + column_mean - grand).reshape_as(memory)


def additive_variance_explained(
    memory: torch.Tensor,
    *,
    heads: int,
    slots_per_head: int,
) -> float:
    memory = memory.detach()
    projected = additive_projection(
        memory.float(),
        heads=heads,
        slots_per_head=slots_per_head,
    )
    table = memory.float().reshape(heads, slots_per_head, -1)
    centered = table - table.mean(dim=1, keepdim=True)
    denominator = centered.square().sum()
    if float(denominator) == 0.0:
        return 1.0
    residual = memory.float().sub(projected).square().sum()
    return float(1.0 - residual / denominator)


def initial_memory_modules(model: nn.Module) -> list[nn.Module]:
    return [
        module
        for module in model.modules()
        if hasattr(module, "slots_per_head")
        and hasattr(module, "num_heads")
        and hasattr(module, "memory")
    ]


def prior_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and getattr(parameter, "_sdm_memory_bank", False)
    )


def initial_memory_diagnostics(model: nn.Module) -> dict[str, object]:
    layers: list[dict[str, object]] = []
    for index, mixer in enumerate(initial_memory_modules(model)):
        memory = mixer.memory
        mode = getattr(mixer, "_initial_state_mode", "zero" if memory is None else "full")
        row: dict[str, object] = {
            "mixer_index": index,
            "mode": mode,
            "heads": int(mixer.num_heads),
            "slots_per_head": int(mixer.slots_per_head),
        }
        if memory is not None:
            detached = memory.detach().float()
            row.update(
                {
                    "logical_elements": detached.numel(),
                    "l2_norm": float(detached.norm()),
                    "rms": float(detached.square().mean().sqrt()),
                    "additive_variance_explained": additive_variance_explained(
                        detached,
                        heads=int(mixer.num_heads),
                        slots_per_head=int(mixer.slots_per_head),
                    ),
                }
            )
        layers.append(row)
    return {
        "learned_prior_parameters": prior_parameter_count(model),
        "layers": layers,
    }


@contextmanager
def temporary_initial_state(model: nn.Module, state: str) -> Iterator[None]:
    """Temporarily expose the learned, zero, or additive-projected prior."""

    if state not in ("normal", "zero", "additive_projection"):
        raise ValueError(f"unknown counterfactual initial state {state!r}")
    restorations: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for mixer in initial_memory_modules(model):
                memory = mixer.memory
                if memory is None or state == "normal":
                    continue
                if parametrize.is_parametrized(mixer, "memory"):
                    factors = product_factors(mixer)
                    for factor in factors:
                        restorations.append((factor, factor.detach().clone()))
                        if state == "zero":
                            factor.zero_()
                    # A product table already lies in the additive subspace.
                    continue
                restorations.append((memory, memory.detach().clone()))
                if state == "zero":
                    memory.zero_()
                else:
                    memory.copy_(
                        additive_projection(
                            memory,
                            heads=int(mixer.num_heads),
                            slots_per_head=int(mixer.slots_per_head),
                        )
                    )
        yield
    finally:
        with torch.no_grad():
            for target, saved in restorations:
                target.copy_(saved)


__all__ = [
    "INITIAL_STATE_MODES",
    "AdditiveProductMemory",
    "additive_projection",
    "additive_variance_explained",
    "attach_additive_product_memory",
    "initial_memory_diagnostics",
    "initial_memory_modules",
    "initialize_full_memory",
    "initialize_product_factors",
    "prior_parameter_count",
    "product_factors",
    "temporary_initial_state",
]
