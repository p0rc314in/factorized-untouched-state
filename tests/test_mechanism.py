from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch
from torch import nn

from product_key_init.prior import (
    AdditiveProductMemory,
    additive_projection,
    attach_additive_product_memory,
    prior_parameter_count,
    product_factors,
    temporary_initial_state,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeMixer(nn.Module):
    def __init__(self, *, heads: int = 2, slots: int = 16, width: int = 6) -> None:
        super().__init__()
        self.num_heads = heads
        self.slots_per_head = slots
        self.args = SimpleNamespace(backprop_on_memory=True)
        self.memory = nn.Parameter(torch.randn(heads * slots, width))
        self.memory._sdm_memory_bank = True


@dataclass
class FakeArgs:
    dim: int
    num_heads: int
    slots_per_head: int
    num_reads: int
    num_writes: int
    memory_block_size: int
    backprop_on_memory: bool
    output_gate: bool
    normalize_readings: bool
    snapshot_quant: str


class FakeSparseDeltaMemory(nn.Module):
    def __init__(self, args: FakeArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.num_heads = args.num_heads
        self.slots_per_head = args.slots_per_head
        self.projection = nn.Linear(args.dim, args.dim)
        self.memory = (
            nn.Parameter(
                torch.zeros(
                    args.num_heads * args.slots_per_head,
                    args.dim // args.num_heads,
                )
            )
            if args.backprop_on_memory
            else None
        )
        if self.memory is not None:
            self.memory._sdm_memory_bank = True

    def init_weights(self) -> None:
        self.projection.reset_parameters()
        if self.memory is not None:
            nn.init.trunc_normal_(self.memory, std=self.args.dim**-0.5)

    def forward(self, values: torch.Tensor):
        offset = 0.0 if self.memory is None else self.memory.mean()
        return self.projection(values) + offset, None


def install_fake_lingua() -> None:
    lingua = ModuleType("lingua")
    sparse = ModuleType("lingua.sparse_delta_memory")
    sparse.SparseDeltaMemory = FakeSparseDeltaMemory
    sparse.SparseDeltaMemoryArgs = FakeArgs
    lingua.sparse_delta_memory = sparse
    sys.modules["lingua"] = lingua
    sys.modules["lingua.sparse_delta_memory"] = sparse


class PriorTests(unittest.TestCase):
    def test_right_inverse_is_additive_projection(self) -> None:
        representation = AdditiveProductMemory(
            heads=2,
            slots_per_head=16,
            value_width=3,
        )
        full = torch.randn(32, 3)
        row, column = representation.right_inverse(full)
        reconstructed = representation(row, column)
        projected = additive_projection(full, heads=2, slots_per_head=16)
        torch.testing.assert_close(reconstructed, projected)

    def test_product_owns_only_factor_parameters(self) -> None:
        mixer = FakeMixer()
        attach_additive_product_memory(mixer)
        row, column = product_factors(mixer)
        self.assertEqual(tuple(row.shape), (2, 4, 6))
        self.assertEqual(tuple(column.shape), (2, 4, 6))
        self.assertEqual(prior_parameter_count(mixer), 96)
        mixer.memory.square().sum().backward()
        self.assertIsNotNone(row.grad)
        self.assertIsNotNone(column.grad)

    def test_counterfactual_restores_parameters_exactly(self) -> None:
        full = FakeMixer(heads=1, slots=16, width=4)
        product = FakeMixer(heads=1, slots=16, width=4)
        attach_additive_product_memory(product)
        model = nn.ModuleList([full, product])
        full_before = full.memory.detach().clone()
        factors_before = [factor.detach().clone() for factor in product_factors(product)]
        with temporary_initial_state(model, "zero"):
            self.assertEqual(torch.count_nonzero(full.memory), 0)
            self.assertEqual(torch.count_nonzero(product.memory), 0)
        torch.testing.assert_close(full.memory, full_before, rtol=0, atol=0)
        for observed, expected in zip(
            product_factors(product),
            factors_before,
            strict=True,
        ):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)


class ModelTests(unittest.TestCase):
    def test_model_does_not_subclass_or_copy_released_forward(self) -> None:
        tree = ast.parse((ROOT / "benchmarks/model.py").read_text())
        subclasses = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id == "SparseDeltaMemory"
                for base in node.bases
            )
        ]
        self.assertEqual(subclasses, [])
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SparseDeltaMemory"
        ]
        self.assertEqual(len(calls), 1)

    def test_arms_match_nonprior_initialization_and_counts(self) -> None:
        install_fake_lingua()
        from benchmarks.model import (
            MatchedLanguageModel,
            nonprior_parameter_sha256,
            parameter_accounting,
        )

        models = {}
        for mode in ("zero", "product", "full"):
            model = MatchedLanguageModel(
                initial_state=mode,
                vocab_size=31,
                maximum_sequence_length=16,
                layout="BBA",
                width=8,
                heads=2,
                memory_heads=1,
                slots=16,
                reads=16,
                writes=16,
                mlp_expansion=2,
                activation_dtype=torch.float32,
            )
            model.initialize_role_keyed(0)
            models[mode] = model
        hashes = {nonprior_parameter_sha256(model) for model in models.values()}
        self.assertEqual(len(hashes), 1)
        counts = {
            mode: parameter_accounting(model)["learned_initial_state_parameters"]
            for mode, model in models.items()
        }
        self.assertEqual(counts, {"zero": 0, "product": 128, "full": 256})
        for model in models.values():
            accounting = parameter_accounting(model)
            self.assertEqual(accounting["input_embedding_parameters"], 31 * 8)
            self.assertEqual(accounting["position_embedding_parameters"], 16 * 8)
            self.assertEqual(accounting["output_embedding_parameters"], 31 * 8)

    def test_canonical_checkpoint_schedule_is_observational(self) -> None:
        from benchmarks.wikitext103_coverage import OPTIMIZER_STEPS_PER_PASS

        self.assertEqual(
            [
                (OPTIMIZER_STEPS_PER_PASS + 1) // 2,
                OPTIMIZER_STEPS_PER_PASS,
                2 * OPTIMIZER_STEPS_PER_PASS,
                3 * OPTIMIZER_STEPS_PER_PASS,
            ],
            [3_601, 7_201, 14_402, 21_603],
        )


if __name__ == "__main__":
    unittest.main()
