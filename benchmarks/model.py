"""Matched decoder shell around the unchanged released SDM layer."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from product_key_init.prior import (
    INITIAL_STATE_MODES,
    attach_additive_product_memory,
    initialize_full_memory,
    initialize_product_factors,
)


PRIOR_INITIALIZATION_BY_STATE = {
    "zero": "none",
    "product": "product_factors",
    "full": "independent_full",
}


def resolve_prior_initialization(
    initial_state: str,
    prior_initialization: str | None,
) -> str:
    expected = PRIOR_INITIALIZATION_BY_STATE[initial_state]
    resolved = expected if prior_initialization is None else prior_initialization
    if resolved != expected:
        raise ValueError(
            f"prior initialization {resolved!r} is incompatible with "
            f"initial state {initial_state!r}"
        )
    return resolved


class ResidualMLP(nn.Module):
    def __init__(self, width: int, *, expansion: float = 4.0) -> None:
        super().__init__()
        hidden = max(1, int(round(width * expansion)))
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, hidden, bias=False)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden, width, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.input(self.norm(tokens))))


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, time, _ = tokens.shape
        query, key, value = self.qkv(self.norm(tokens)).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, time, self.heads, self.head_width).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        flash_eligible = tokens.is_cuda and tokens.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        if tokens.is_cuda:
            backend = (
                torch.nn.attention.SDPBackend.FLASH_ATTENTION
                if flash_eligible
                else torch.nn.attention.SDPBackend.MATH
            )
            with torch.nn.attention.sdpa_kernel(backend):
                attended = F.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )
        else:
            attended = F.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.width)
        return self.output(attended)


class DenseAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, *, mlp_expansion: float) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(width, heads)
        self.mlp = ResidualMLP(width, expansion=mlp_expansion)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(tokens)
        return tokens + self.mlp(tokens)


class SDMBlock(nn.Module):
    """External pre-norm/MLP shell; the mixer itself is released SDM."""

    def __init__(
        self,
        *,
        initial_state: str,
        width: int,
        memory_heads: int,
        slots: int,
        reads: int,
        writes: int,
        mlp_expansion: float,
        layer_idx: int,
    ) -> None:
        super().__init__()
        if initial_state not in INITIAL_STATE_MODES:
            raise ValueError(f"unknown initial state {initial_state!r}")
        if memory_heads <= 0 or width % memory_heads:
            raise ValueError("width must be divisible by sparse memory heads")
        from lingua.sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

        self.initial_state = initial_state
        self.compute_dtype: torch.dtype | None = None
        self.norm = nn.LayerNorm(width)
        self.mixer = SparseDeltaMemory(
            SparseDeltaMemoryArgs(
                dim=width,
                num_heads=memory_heads,
                slots_per_head=slots,
                num_reads=reads,
                num_writes=writes,
                memory_block_size=256,
                backprop_on_memory=initial_state != "zero",
                output_gate=True,
                normalize_readings=True,
                snapshot_quant="none",
            ),
            layer_id=layer_idx,
        )
        self.mixer._initial_state_mode = initial_state
        if initial_state == "product":
            attach_additive_product_memory(self.mixer)
        self.mlp = ResidualMLP(width, expansion=mlp_expansion)

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        self.mixer.to(dtype=dtype)
        self.compute_dtype = dtype

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens)
        mixer_input = (
            normalized.to(self.compute_dtype)
            if self.compute_dtype is not None
            else normalized
        )
        # SDM deliberately keeps some WY products in FP32. A blanket outer
        # autocast must not change those internal boundaries.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            mixed, _ = self.mixer(mixer_input)
        tokens = tokens + mixed
        return tokens + self.mlp(tokens)


class MatchedLanguageModel(nn.Module):
    """Seven SDM blocks followed by one dense attention block by default."""

    def __init__(
        self,
        *,
        initial_state: str,
        vocab_size: int,
        maximum_sequence_length: int,
        layout: str,
        width: int,
        heads: int,
        memory_heads: int,
        slots: int,
        reads: int,
        writes: int,
        mlp_expansion: float = 4.0,
        activation_dtype: torch.dtype = torch.bfloat16,
        prior_initialization: str | None = None,
    ) -> None:
        super().__init__()
        if initial_state not in INITIAL_STATE_MODES:
            raise ValueError(f"initial_state must be one of {INITIAL_STATE_MODES}")
        if not layout or any(kind not in "AB" for kind in layout):
            raise ValueError("layout must contain only A and B")
        if not layout.count("B"):
            raise ValueError("layout must contain at least one SDM block")
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be divisible by attention heads")
        self.initial_state = initial_state
        self.prior_initialization = resolve_prior_initialization(
            initial_state,
            prior_initialization,
        )
        self.layout = layout
        self.width = width
        self.heads = heads
        self.memory_heads = memory_heads
        self.slots = slots
        self.reads = reads
        self.writes = writes
        self.maximum_sequence_length = maximum_sequence_length
        self.activation_dtype = activation_dtype
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(maximum_sequence_length, width)
        self.layers = nn.ModuleList()
        for layer_idx, kind in enumerate(layout):
            if kind == "A":
                layer: nn.Module = DenseAttentionBlock(
                    width,
                    heads,
                    mlp_expansion=mlp_expansion,
                )
            else:
                layer = SDMBlock(
                    initial_state=initial_state,
                    width=width,
                    memory_heads=memory_heads,
                    slots=slots,
                    reads=reads,
                    writes=writes,
                    mlp_expansion=mlp_expansion,
                    layer_idx=layer_idx,
                )
            self.layers.append(layer)
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)

    def initialize_role_keyed(self, seed: int) -> None:
        base = seed * 100_000
        rng_devices = _module_cuda_devices(self)
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(base + 101)
            nn.init.normal_(self.token_embedding.weight, std=self.width**-0.5)
            torch.manual_seed(base + 102)
            nn.init.normal_(self.position_embedding.weight, std=self.width**-0.5)
            torch.manual_seed(base + 103)
            nn.init.normal_(self.output.weight, std=self.width**-0.5)
        self.final_norm.reset_parameters()
        for layer_idx, (kind, layer) in enumerate(zip(self.layout, self.layers)):
            if kind == "A":
                _reset_children(layer.mlp, base + 1_000 + layer_idx)
                _reset_children(layer.attention, base + 2_000 + layer_idx)
                continue
            _reset_children(layer.mlp, base + 1_000 + layer_idx)
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(base + 6_000 + layer_idx)
                layer.mixer.init_weights()
            if self.initial_state in ("product", "full"):
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(base + 8_000 + layer_idx)
                    initializer = (
                        initialize_product_factors
                        if self.initial_state == "product"
                        else initialize_full_memory
                    )
                    initializer(layer.mixer, full_table_std=self.width**-0.5)

    def set_mixer_compute_dtype(self, dtype: torch.dtype) -> None:
        for kind, layer in zip(self.layout, self.layers):
            if kind == "B":
                layer.set_compute_dtype(dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        time = input_ids.shape[1]
        if time > self.maximum_sequence_length:
            raise ValueError("sequence exceeds maximum_sequence_length")
        positions = torch.arange(time, device=input_ids.device)
        tokens = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions).unsqueeze(0)
        ).to(self.activation_dtype)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.output(self.final_norm(tokens))

    def recurrent_state_accounting(self) -> dict[str, int | str]:
        mixer_layers = self.layout.count("B")
        head_width = self.width // self.memory_heads
        elements_per_layer = self.memory_heads * self.slots * head_width
        return {
            "kind": "sparse_logical_table",
            "layers": mixer_layers,
            "memory_heads": self.memory_heads,
            "elements_per_layer_per_sequence": elements_per_layer,
            "elements_per_sequence": mixer_layers * elements_per_layer,
            "bytes_bf16_per_sequence": 2 * mixer_layers * elements_per_layer,
            "bytes_fp32_per_sequence": 4 * mixer_layers * elements_per_layer,
        }

    def arm_identity(self) -> dict[str, Any]:
        codebook_size = int(round(self.slots**0.5))
        return {
            "mechanism": "sparse_delta_memory",
            "host_implementation": "released_sparse_delta_memory",
            "initial_state": self.initial_state,
            "prior_initialization_kind": self.prior_initialization,
            "initial_state_construction": {
                "zero": "none",
                "product": "additive_two_product_key_factor_tables",
                "full": "full_learned_table",
            }[self.initial_state],
            "width": self.width,
            "attention_heads": self.heads,
            "memory_heads": self.memory_heads,
            "slots_per_head": self.slots,
            "product_key_codebook_size": codebook_size,
            "value_width_per_memory_head": self.width // self.memory_heads,
            "reads": self.reads,
            "writes": self.writes,
            "topology": "independent_per_recurrent_layer",
            "controller": "released_tied_scalar_decay_and_delta_gate",
            "canonical_sdm_forward_and_kernels": True,
            "prior_initialization": {
                "semantic_role_seed_offset": 8_000,
                "full_table_target_std": self.width**-0.5,
                "truncation_standard_deviations": 3,
            },
        }


def _reset_children(module: nn.Module, seed: int) -> None:
    with torch.random.fork_rng(devices=_module_cuda_devices(module)):
        torch.manual_seed(seed)
        for child in module.modules():
            if child is module:
                continue
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()


def _module_cuda_devices(module: nn.Module) -> list[int]:
    return sorted(
        {
            int(parameter.device.index)
            for parameter in module.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )


def parameter_accounting(model: MatchedLanguageModel) -> dict[str, Any]:
    prior_ids = {
        id(parameter)
        for parameter in model.parameters()
        if getattr(parameter, "_sdm_memory_bank", False)
    }
    active = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in prior_ids
    )
    learned_prior = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in prior_ids
    )
    return {
        "active_parameters_excluding_learned_initial_state": active,
        "input_embedding_parameters": model.token_embedding.weight.numel(),
        "position_embedding_parameters": model.position_embedding.weight.numel(),
        "output_embedding_parameters": model.output.weight.numel(),
        "learned_initial_state_parameters": learned_prior,
        "total_trainable_parameters": active + learned_prior,
        "learned_initial_state_names": [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in prior_ids
        ],
        "logical_recurrent_state": model.recurrent_state_accounting(),
    }


def nonprior_parameter_sha256(model: MatchedLanguageModel) -> str:
    """Fingerprint every parameter that must begin identically in all arms."""

    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not getattr(parameter, "_sdm_memory_bank", False)
    }
    digest = hashlib.sha256()
    for name in sorted(selected):
        tensor = selected[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "MatchedLanguageModel",
    "SDMBlock",
    "nonprior_parameter_sha256",
    "parameter_accounting",
    "resolve_prior_initialization",
]
