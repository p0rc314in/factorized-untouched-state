#!/usr/bin/env python3
"""Train one matched SDM initial-state arm on canonical WikiText-103 coverage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib import metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.recovery import (
    load_recovery,
    recovery_record,
    restore_recovery,
    restore_rng,
)
from benchmarks.model import (
    MatchedLanguageModel,
    nonprior_parameter_sha256,
    parameter_accounting,
    resolve_prior_initialization,
)
from benchmarks.wikitext103_coverage import (
    CanonicalWikiTextData,
    CONTEXT_LENGTH,
    EFFECTIVE_BATCH_RECORDS,
    EVALUATION_STRIDE,
    OPTIMIZER_STEPS_PER_PASS,
    PROTOCOL_ID,
    STREAM_SEED,
    TRAIN_RECORDS_PER_PASS,
    TRAIN_TARGETS_PER_PASS,
    VOCAB_SIZE,
    canonical_json_sha256,
    sha256_file,
)
from product_key_init.prior import (
    initial_memory_diagnostics,
    temporary_initial_state,
)
from scripts.prepare_sdm_runtime import (
    RUNTIME_MANIFEST_NAME,
    verify_loader_only_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_IDENTITY = json.loads((ROOT / "PROJECT_IDENTITY.json").read_text())
PROJECT_MODEL_IDENTITY = PROJECT_IDENTITY["model_identity"]
PROJECT_EXPERIMENT_IDENTITY = PROJECT_IDENTITY["experiment"]
LAYOUT = "BBBBBBBA"
WIDTH = 128
ATTENTION_HEADS = 4
MEMORY_HEADS = 1
SLOTS = 1_024
READS = 16
WRITES = 16
MLP_EXPANSION = 4.0
EXPECTED_PARAMETERS = {
    "zero": {
        "active_parameters_excluding_learned_initial_state": 14_712_348,
        "learned_initial_state_parameters": 0,
        "total_trainable_parameters": 14_712_348,
    },
    "product": {
        "active_parameters_excluding_learned_initial_state": 14_712_348,
        "learned_initial_state_parameters": 57_344,
        "total_trainable_parameters": 14_769_692,
    },
    "full": {
        "active_parameters_excluding_learned_initial_state": 14_712_348,
        "learned_initial_state_parameters": 917_504,
        "total_trainable_parameters": 15_629_852,
    },
}
RECOVERY_KIND = "factorized_sdm_canonical_wikitext_recovery"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verified_dependency_environment() -> dict[str, Any]:
    from lingua.sparse_delta_memory import SparseDeltaMemory

    base_root = (ROOT / "third_party/released_sdm").resolve()
    source_manifest_path = base_root / "SOURCE.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("modified") is not False:
        raise RuntimeError("released SDM source manifest reports modifications")
    expected_files = source_manifest.get("files", {})
    observed_files = {
        relative: sha256_file(base_root / relative)
        for relative in sorted(expected_files)
    }
    if observed_files != expected_files:
        raise RuntimeError("vendored released SDM source does not match SOURCE.json")

    configured_root = os.environ.get("SDM_SOURCE_ROOT")
    configured_patch = os.environ.get("SDM_LOADER_PATCH_SHA256")
    if configured_root is None or configured_patch is None:
        raise RuntimeError("loader-patched SDM runtime provenance is not configured")
    sdm_root = Path(configured_root).resolve()
    verify_loader_only_runtime(sdm_root)
    runtime_manifest_path = sdm_root / RUNTIME_MANIFEST_NAME
    runtime_manifest = json.loads(runtime_manifest_path.read_text())
    if configured_patch != runtime_manifest["loader_patch_sha256"]:
        raise RuntimeError("configured SDM loader patch hash changed")
    sdm_file = Path(inspect.getfile(SparseDeltaMemory)).resolve()
    if sdm_root not in sdm_file.parents:
        raise RuntimeError(f"SDM resolved outside the patched runtime: {sdm_file}")
    return {
        "sdm_source_file": str(sdm_file),
        "sdm_source_root": str(sdm_root),
        "sdm_source_revision": source_manifest["revision"],
        "sdm_source_manifest_sha256": sha256_file(source_manifest_path),
        "sdm_loader_patch_sha256": configured_patch,
        "sdm_runtime_manifest_sha256": sha256_file(runtime_manifest_path),
        "sdm_model_and_kernel_mechanics_changed": False,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array)
    host = torch.from_numpy(contiguous).pin_memory()
    return host.to(device, non_blocking=True)


class FP32MasterAdamW:
    """AdamW with explicit FP32 master parameters and FP32 moment state."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
    ) -> None:
        self.model_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        self.master_parameters = [
            nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
            for parameter in self.model_parameters
        ]
        decay = []
        no_decay = []
        for model_parameter, master_parameter in zip(
            self.model_parameters,
            self.master_parameters,
            strict=True,
        ):
            target = no_decay if getattr(model_parameter, "_no_weight_decay", False) else decay
            target.append(master_parameter)
        groups: list[dict[str, Any]] = [
            {"params": decay, "weight_decay": weight_decay}
        ]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=lr,
            betas=betas,
            fused=True,
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.model_parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters,
            self.master_parameters,
            strict=True,
        ):
            master_parameter.grad = (
                None
                if model_parameter.grad is None
                else model_parameter.grad.detach().float()
            )
        self.optimizer.step()
        for model_parameter, master_parameter in zip(
            self.model_parameters,
            self.master_parameters,
            strict=True,
        ):
            model_parameter.copy_(master_parameter.to(dtype=model_parameter.dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "optimizer": self.optimizer.state_dict(),
            "master_parameters": [
                parameter.detach().clone() for parameter in self.master_parameters
            ],
        }

    @torch.no_grad()
    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != 1:
            raise ValueError("unexpected FP32 master optimizer schema")
        values = payload.get("master_parameters")
        if not isinstance(values, list) or len(values) != len(self.master_parameters):
            raise ValueError("FP32 master optimizer parameter count changed")
        for destination, source in zip(self.master_parameters, values, strict=True):
            if destination.shape != source.shape:
                raise ValueError("FP32 master optimizer parameter shape changed")
            destination.copy_(source.to(device=destination.device, dtype=torch.float32))
        self.optimizer.load_state_dict(payload["optimizer"])
        for model_parameter, master_parameter in zip(
            self.model_parameters,
            self.master_parameters,
            strict=True,
        ):
            model_parameter.copy_(master_parameter.to(dtype=model_parameter.dtype))

    def state_precision(self) -> dict[str, Any]:
        dtypes = sorted(
            {
                str(value.dtype).removeprefix("torch.")
                for state in self.optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor) and value.numel() > 1
            }
        )
        return {
            "master_parameter_dtype": "float32",
            "moment_dtypes": dtypes,
            "all_moments_fp32": dtypes == ["float32"],
        }


@dataclass(frozen=True)
class TrainConfig:
    campaign: str
    campaign_design_sha256: str
    arm: str
    variant_id: str
    prior_initialization: str
    experiment_scope: str
    protocol_id: str
    passes: int
    total_steps: int
    schedule_steps: int
    batch_size: int
    micro_batch_size: int
    sequence_length: int
    evaluation_stride: int
    vocab_size: int
    layout: str
    width: int
    attention_heads: int
    memory_heads: int
    slots: int
    reads: int
    writes: int
    mlp_expansion: float
    peak_learning_rate: float
    minimum_lr_ratio: float
    warmup_steps: int
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    gradient_clip: float
    seed: int
    stream_seed: int
    initialization_device: str
    activation_dtype: str
    optimizer_state_dtype: str


def learning_rate(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return config.peak_learning_rate * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(
        1,
        config.schedule_steps - config.warmup_steps,
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    factor = config.minimum_lr_ratio + (1.0 - config.minimum_lr_ratio) * cosine
    return config.peak_learning_rate * factor


def uniform_logit_probe() -> dict[str, Any]:
    observed = float(
        F.cross_entropy(
            torch.zeros((1, VOCAB_SIZE), dtype=torch.float32),
            torch.zeros(1, dtype=torch.long),
        )
    )
    expected = math.log(VOCAB_SIZE)
    difference = abs(observed - expected)
    if difference > 1e-6:
        raise AssertionError(f"uniform-logit probe changed by {difference}")
    return {
        "schema_version": 1,
        "status": "pass",
        "vocabulary_size": VOCAB_SIZE,
        "expected_nll": expected,
        "observed_nll": observed,
        "absolute_difference": difference,
    }


@torch.no_grad()
def verify_prefix_causality(
    model: MatchedLanguageModel,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed + 91_003)
    tokens = torch.randint(
        VOCAB_SIZE,
        (2, 16),
        generator=generator,
        device=device,
    )
    cut = 7
    changed = tokens.clone()
    changed[:, cut + 1 :] = torch.randint(
        VOCAB_SIZE,
        changed[:, cut + 1 :].shape,
        generator=generator,
        device=device,
    )
    model.eval()
    with torch.autocast("cuda", dtype=model.activation_dtype):
        baseline = model(tokens)
        counterfactual = model(changed)
    difference = float(
        (baseline[:, : cut + 1].float() - counterfactual[:, : cut + 1].float())
        .abs()
        .max()
    )
    if difference != 0.0:
        raise AssertionError(f"prefix causality failed: {difference}")
    return {
        "schema_version": 1,
        "status": "causal",
        "maximum_prefix_difference": difference,
        "cut_position": cut,
    }


def validate_sdm_instantiation(
    model: MatchedLanguageModel,
    accounting: dict[str, Any],
) -> dict[str, Any]:
    from lingua.sparse_delta_memory import SparseDeltaMemory
    from torch.nn.utils import parametrize

    mixers = [
        layer.mixer
        for kind, layer in zip(model.layout, model.layers)
        if kind == "B"
    ]
    if not mixers or any(not isinstance(mixer, SparseDeltaMemory) for mixer in mixers):
        raise RuntimeError("model must instantiate released SparseDeltaMemory")
    if any(type(mixer).forward is not SparseDeltaMemory.forward for mixer in mixers):
        raise RuntimeError("released SDM forward was replaced")
    expected_prior = EXPECTED_PARAMETERS[model.initial_state][
        "learned_initial_state_parameters"
    ]
    observed_prior = int(accounting["learned_initial_state_parameters"])
    if observed_prior != expected_prior:
        raise RuntimeError(
            f"learned initial state has {observed_prior} parameters, "
            f"expected {expected_prior}"
        )
    rows = []
    for layer_index, mixer in enumerate(mixers):
        is_product = parametrize.is_parametrized(mixer, "memory")
        if model.initial_state == "zero":
            valid = mixer.memory is None and not mixer.args.backprop_on_memory
        elif model.initial_state == "full":
            valid = (
                isinstance(mixer.memory, torch.nn.Parameter)
                and mixer.args.backprop_on_memory
                and not is_product
            )
        else:
            valid = (
                mixer.memory is not None
                and mixer.args.backprop_on_memory
                and is_product
            )
        if not valid:
            raise RuntimeError(
                f"initial-state representation mismatch in mixer {layer_index}"
            )
        rows.append(
            {
                "mixer_index": layer_index,
                "class_module": type(mixer).__module__,
                "class_name": type(mixer).__name__,
                "released_sdm_instance": True,
                "direct_released_type": type(mixer) is SparseDeltaMemory,
                "memory_parametrized": is_product,
                "backprop_on_memory": bool(mixer.args.backprop_on_memory),
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "released_sdm_forward_and_kernels_unchanged": True,
        "experimental_seam": "torch_parametrization_of_memory_only",
        "initial_state": model.initial_state,
        "prior_initialization": model.prior_initialization,
        "learned_initial_state_parameters": observed_prior,
        "mixers": rows,
    }


@torch.no_grad()
def evaluate(
    model: MatchedLanguageModel,
    data: CanonicalWikiTextData,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    targets = 0
    scored_bytes = 0
    windows = 0
    started = time.perf_counter()
    for batch in data.evaluation_batches(split):
        values = to_device(batch.tokens.astype(np.int64, copy=False), device)
        mask = to_device(batch.target_mask, device)
        inputs, labels = values[:, :-1], values[:, 1:]
        with torch.autocast("cuda", dtype=model.activation_dtype):
            logits = model(inputs)
            losses = F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                reduction="none",
            ).view_as(labels)
        selected = losses[mask]
        loss_sum += float(selected.float().sum())
        correct += int(logits.argmax(dim=-1).eq(labels)[mask].sum())
        targets += batch.target_count
        scored_bytes += batch.scored_bytes
        windows += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    declared = data.manifest["evaluation"]["splits"][split]
    if targets != int(declared["scored_targets"]):
        raise ValueError(f"{split} evaluation target coverage changed")
    if scored_bytes != int(declared["scored_bytes"]):
        raise ValueError(f"{split} evaluation byte coverage changed")
    nll = loss_sum / targets
    return {
        "split": split,
        "protocol_id": PROTOCOL_ID,
        "context_length": CONTEXT_LENGTH,
        "stride": EVALUATION_STRIDE,
        "windows": windows,
        "total_nll": loss_sum,
        "nll": nll,
        "perplexity": math.exp(nll),
        "bits_per_utf8_byte": loss_sum / (math.log(2.0) * scored_bytes),
        "next_token_accuracy": correct / targets,
        "correct_targets": correct,
        "scored_targets": targets,
        "scored_bytes": scored_bytes,
        "seconds": elapsed,
        "scored_targets_per_second": targets / elapsed,
    }


def final_prior_diagnostics(
    model: MatchedLanguageModel,
    data: CanonicalWikiTextData,
    device: torch.device,
    *,
    normal_validation: dict[str, Any],
    initial: dict[str, Any],
) -> dict[str, Any]:
    states = ["normal"]
    if model.initial_state in {"product", "full"}:
        states.append("zero")
    if model.initial_state == "full":
        states.append("additive_projection")
    evaluations: dict[str, Any] = {"normal": normal_validation}
    for state in states[1:]:
        with temporary_initial_state(model, state):
            evaluations[state] = evaluate(model, data, "validation", device)
    baseline = evaluations["normal"]["nll"]
    return {
        "schema_version": 1,
        "initial_state": model.initial_state,
        "initial": initial,
        "final": initial_memory_diagnostics(model),
        "validation": evaluations,
        "validation_nll_penalty": {
            state: row["nll"] - baseline
            for state, row in evaluations.items()
            if state != "normal"
        },
    }


def fixed_budget_record(
    metrics: list[dict[str, Any]],
    *,
    passes: int,
) -> dict[str, Any]:
    by_pass = {
        int(row["pass"]): row
        for row in metrics
        if row.get("checkpoint_kind") == "complete_pass"
    }
    if sorted(by_pass) != list(range(1, passes + 1)):
        raise ValueError("budget record lacks a complete validation pass series")
    return {
        "schema_version": 1,
        "status": "fixed_predeclared_budget_complete",
        "budget_passes": passes,
        "budget_selected_before_model_results": True,
        "validation_checkpoints_are_reporting_only": True,
        "test_evaluation_allowed": True,
        "schedule_action": "retain_fixed_predeclared_budget",
        "pass_validation_nll": {
            str(index): by_pass[index]["validation"]["nll"]
            for index in range(1, passes + 1)
        },
    }


def write_recovery(
    *,
    output: Path,
    model: MatchedLanguageModel,
    optimizer: FP32MasterAdamW,
    config: TrainConfig,
    step: int,
    manifest_sha256: str,
    metrics: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    checkpoint = output / "recovery_checkpoint.pt"
    atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "kind": RECOVERY_KIND,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "manifest_sha256": manifest_sha256,
            "metrics": metrics,
            "training_curve": curve,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        },
    )
    row = {
        "schema_version": 1,
        "status": "recoverable",
        "artifact": checkpoint.name,
        "step": step,
        "base_complete": complete,
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }
    atomic_json(output / "RECOVERY.json", row)
    return row


def validate_resolved_config(args: argparse.Namespace, data: CanonicalWikiTextData) -> TrainConfig:
    if args.arm not in {"zero", "product", "full"}:
        raise ValueError("canonical comparison contains only zero, product, and full")
    if args.seed != 0:
        raise ValueError("this reproduction contains the recorded seed-0 comparison")
    if args.experiment_scope != "single_seed":
        raise ValueError("the resolved experiment scope must be single_seed")
    expected = {
        "layout": LAYOUT,
        "width": WIDTH,
        "attention_heads": ATTENTION_HEADS,
        "memory_heads": MEMORY_HEADS,
        "slots": SLOTS,
        "reads": READS,
        "writes": WRITES,
        "batch_size": EFFECTIVE_BATCH_RECORDS,
        "stream_seed": STREAM_SEED,
    }
    for name, wanted in expected.items():
        if getattr(args, name) != wanted:
            raise ValueError(f"resolved {name}={getattr(args, name)!r}; expected {wanted!r}")
    identity_values = {
        "protocol_id": PROTOCOL_ID,
        "layout": LAYOUT,
        "width": WIDTH,
        "attention_heads": ATTENTION_HEADS,
        "memory_heads": MEMORY_HEADS,
        "slots_per_head": SLOTS,
        "reads": READS,
        "writes": WRITES,
        "seed": 0,
        "passes": data.passes,
        "optimizer_steps": data.training_steps,
    }
    for name, wanted in identity_values.items():
        if PROJECT_EXPERIMENT_IDENTITY.get(name) != wanted:
            raise ValueError(f"project identity disagrees at {name}")
    if args.writes != args.reads:
        raise ValueError("balanced sparse access requires W = R")
    if args.warmup_steps != round(data.training_steps * 0.025):
        raise ValueError("warmup is not 2.5 percent of the frozen schedule")
    if args.passes != data.passes:
        raise ValueError("resolved pass budget disagrees with the prepared stream")
    if args.schedule_steps != data.training_steps:
        raise ValueError("cosine schedule must end at the frozen training budget")
    if args.micro_batch_size <= 0 or args.batch_size % args.micro_batch_size:
        raise ValueError("microbatch size must divide the effective batch")
    if args.minimum_lr_ratio != 0.1:
        raise ValueError("canonical cosine schedule must end at 0.1 times peak LR")
    if (args.adam_beta1, args.adam_beta2) != (0.9, 0.95):
        raise ValueError("canonical AdamW beta values changed")
    if args.gradient_clip != 1.0:
        raise ValueError("canonical gradient clipping changed")
    if args.initialization_device != "cuda":
        raise ValueError("canonical role-keyed initialization must run on CUDA")
    return TrainConfig(
        campaign=args.campaign,
        campaign_design_sha256=args.campaign_design_sha256,
        arm=args.arm,
        variant_id=args.variant_id,
        prior_initialization=resolve_prior_initialization(
            args.arm,
            args.prior_initialization,
        ),
        experiment_scope=args.experiment_scope,
        protocol_id=PROTOCOL_ID,
        passes=args.passes,
        total_steps=data.training_steps,
        schedule_steps=args.schedule_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        sequence_length=CONTEXT_LENGTH,
        evaluation_stride=EVALUATION_STRIDE,
        vocab_size=VOCAB_SIZE,
        layout=args.layout,
        width=args.width,
        attention_heads=args.attention_heads,
        memory_heads=args.memory_heads,
        slots=args.slots,
        reads=args.reads,
        writes=args.writes,
        mlp_expansion=args.mlp_expansion,
        peak_learning_rate=args.peak_learning_rate,
        minimum_lr_ratio=args.minimum_lr_ratio,
        warmup_steps=args.warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        stream_seed=args.stream_seed,
        initialization_device=args.initialization_device,
        activation_dtype=args.activation_dtype,
        optimizer_state_dtype="float32",
    )


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    data = CanonicalWikiTextData(args.manifest, passes=args.passes)
    config = validate_resolved_config(args, data)
    dependency_environment = verified_dependency_environment()
    set_seed(config.seed)
    device = torch.device("cuda")
    activation_dtype = {"bfloat16": torch.bfloat16}[config.activation_dtype]
    model = MatchedLanguageModel(
        initial_state=config.arm,
        vocab_size=config.vocab_size,
        maximum_sequence_length=config.sequence_length,
        layout=config.layout,
        width=config.width,
        heads=config.attention_heads,
        memory_heads=config.memory_heads,
        slots=config.slots,
        reads=config.reads,
        writes=config.writes,
        mlp_expansion=config.mlp_expansion,
        activation_dtype=activation_dtype,
        prior_initialization=config.prior_initialization,
    )
    model = model.to(device)
    model.initialize_role_keyed(config.seed)
    matched_sha = nonprior_parameter_sha256(model)
    initial_prior = initial_memory_diagnostics(model)
    model.set_mixer_compute_dtype(activation_dtype)
    accounting = parameter_accounting(model)
    for key, expected in EXPECTED_PARAMETERS[config.arm].items():
        if accounting[key] != expected:
            raise ValueError(f"{config.arm} parameter count changed at {key}")
    arm_identity = model.arm_identity()
    sdm_validation = validate_sdm_instantiation(model, accounting)
    if arm_identity.get("reads") != READS or arm_identity.get("writes") != WRITES:
        raise ValueError("instantiated model access geometry changed")
    if arm_identity.get("memory_heads") != MEMORY_HEADS:
        raise ValueError("instantiated model memory-head geometry changed")
    optimizer = FP32MasterAdamW(
        model,
        lr=config.peak_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256_file(data.manifest_path)
    if bool(args.resume_checkpoint) != bool(args.resume_checkpoint_sha256):
        raise ValueError("resume checkpoint and SHA-256 must be supplied together")
    resume_payload: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None
    start_step = 0
    if args.resume_checkpoint is not None:
        resume_payload = load_recovery(
            args.resume_checkpoint,
            expected_sha256=args.resume_checkpoint_sha256,
            expected_kind=RECOVERY_KIND,
            expected_config=asdict(config),
            expected_manifest_sha256=manifest_sha,
            sha256_file=sha256_file,
        )
        restore_recovery(resume_payload, model=model, optimizer=optimizer)
        start_step = int(resume_payload["step"])
        resume = recovery_record(resume_payload)

    environment = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": metadata.version("triton"),
        "gpu": torch.cuda.get_device_name(device),
        **dependency_environment,
    }
    configuration = {
        **asdict(config),
        "project_model_identity": PROJECT_MODEL_IDENTITY,
        "model_identity": arm_identity,
        "parameter_accounting": accounting,
        "nonprior_initialization_sha256": matched_sha,
        "initial_prior_diagnostics": initial_prior,
        "manifest_sha256": manifest_sha,
        "data_payload_sha256": data.manifest["remote_payload"]["sha256"],
        "checkpoint_steps": data.manifest["training"]["checkpoint_steps"],
        "precision": {
            "activations": "bfloat16",
            "optimizer_master_parameters": "float32",
            "optimizer_moments": "float32",
        },
        "resume": resume,
        "environment": environment,
        "experiment_scope": config.experiment_scope,
    }
    atomic_json(output / "config.json", configuration)
    atomic_json(output / "UNIFORM_LOGIT.json", uniform_logit_probe())
    atomic_json(
        output / "CAUSALITY.json",
        verify_prefix_causality(model, device=device, seed=config.seed),
    )
    atomic_json(output / "SDM_VALIDATION.json", sdm_validation)
    if resume_payload is not None:
        restore_rng(resume_payload)

    coverage = data.coverage_report()
    coverage["consumed_optimizer_steps"] = start_step
    coverage["consumed_target_presentations"] = data.target_presentations_through_step(
        start_step
    )
    coverage["complete"] = start_step == config.total_steps
    atomic_json(output / "COVERAGE.json", coverage)
    print(
        json.dumps(
            {
                "marker": "STARTED",
                "campaign": config.campaign,
                "arm": config.arm,
                "variant_id": config.variant_id,
                "steps": config.total_steps,
                "target_presentations": data.coverage["total_target_presentations"],
                "reads": config.reads,
                "writes": config.writes,
                "resumed_from_step": start_step,
                "environment": environment,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    metrics: list[dict[str, Any]] = (
        list(resume_payload["metrics"]) if resume_payload is not None else []
    )
    curve: list[dict[str, Any]] = (
        list(resume_payload["training_curve"])
        if resume_payload is not None
        else []
    )
    recoveries: list[dict[str, Any]] = []
    checkpoints = set(int(step) for step in data.manifest["training"]["checkpoint_steps"])
    recovery_steps = set(
        range(
            args.recovery_checkpoint_interval,
            config.total_steps + 1,
            args.recovery_checkpoint_interval,
        )
    ) | {config.total_steps}
    metric_path = output / "metrics.jsonl"
    curve_path = output / "training_curve.jsonl"
    started = time.perf_counter()
    interval_started = started
    interval_targets = 0
    optimizer_seconds = 0.0
    evaluation_seconds = 0.0
    recovery_seconds = 0.0
    segment_targets = 0
    pending: list[tuple[int, float, torch.Tensor, int]] = []
    torch.cuda.reset_peak_memory_stats(device)

    with metric_path.open("w", encoding="utf-8") as metric_handle, curve_path.open(
        "w", encoding="utf-8"
    ) as curve_handle:
        for row in metrics:
            metric_handle.write(json.dumps(row, sort_keys=True) + "\n")
        metric_handle.flush()
        for row in curve:
            curve_handle.write(json.dumps(row, sort_keys=True) + "\n")
        curve_handle.flush()
        for step_index in range(start_step, config.total_steps):
            step = step_index + 1
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = data.train_batch(step_index)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            snapshot = None
            if step == start_step + 1:
                snapshot = next(model.parameters()).detach().clone()
            for micro_start in range(0, config.batch_size, config.micro_batch_size):
                micro_stop = micro_start + config.micro_batch_size
                values = to_device(
                    batch.tokens[micro_start:micro_stop].astype(np.int64, copy=False),
                    device,
                )
                mask = to_device(batch.target_mask[micro_start:micro_stop], device)
                inputs, labels = values[:, :-1], values[:, 1:]
                with torch.autocast("cuda", dtype=model.activation_dtype):
                    logits = model(inputs)
                    losses = F.cross_entropy(
                        logits.flatten(0, 1),
                        labels.flatten(),
                        reduction="none",
                    ).view_as(labels)
                    selected_sum = losses[mask].sum()
                    scaled = selected_sum / batch.target_count
                scaled.backward()
                loss_sum += selected_sum.detach()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            mean_loss = loss_sum / batch.target_count
            interval_targets += batch.target_count
            segment_targets += batch.target_count
            pending.append((step, lr, mean_loss, batch.target_count))

            if step == start_step + 1:
                torch.cuda.synchronize(device)
                if snapshot is None:
                    raise AssertionError("first-step parameter snapshot is absent")
                update = float((next(model.parameters()).detach() - snapshot).abs().max())
                precision = optimizer.state_precision()
                health = {
                    "schema_version": 1,
                    "status": "healthy",
                    "campaign": config.campaign,
                    "arm": config.arm,
                    "variant_id": config.variant_id,
                    "initial_state": config.arm,
                    "prior_initialization": config.prior_initialization,
                    "seed": config.seed,
                    "experiment_scope": config.experiment_scope,
                    "protocol_id": config.protocol_id,
                    "campaign_design_sha256": config.campaign_design_sha256,
                    "manifest_sha256": manifest_sha,
                    "data_payload_sha256": data.manifest["remote_payload"]["sha256"],
                    "nonprior_initialization_sha256": matched_sha,
                    "initialization_device": config.initialization_device,
                    "step": step,
                    "task_nll": float(mean_loss),
                    "gradient_norm": float(gradient_norm),
                    "maximum_parameter_update": update,
                    "optimizer_precision": precision,
                    "gpu": environment["gpu"],
                    "reads": config.reads,
                    "writes": config.writes,
                }
                if not all(
                    math.isfinite(float(health[key]))
                    for key in ("task_nll", "gradient_norm", "maximum_parameter_update")
                ) or update <= 0:
                    raise FloatingPointError(f"invalid first optimizer step: {health}")
                if not precision["all_moments_fp32"]:
                    raise FloatingPointError("optimizer moments are not FP32")
                atomic_json(output / "HEALTHY.json", health)

            should_flush = (
                step % args.heartbeat_steps == 0
                or step in checkpoints
                or step in recovery_steps
            )
            if should_flush:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval_elapsed = now - interval_started
                optimizer_seconds += interval_elapsed
                loss_values = torch.stack([row[2].float() for row in pending]).cpu().tolist()
                for pending_row, loss_value in zip(pending, loss_values, strict=True):
                    row = {
                        "step": pending_row[0],
                        "learning_rate": pending_row[1],
                        "train_nll": loss_value,
                        "scored_targets": pending_row[3],
                        "target_presentations": data.target_presentations_through_step(
                            pending_row[0]
                        ),
                    }
                    curve.append(row)
                    curve_handle.write(json.dumps(row, sort_keys=True) + "\n")
                curve_handle.flush()
                pending.clear()
                elapsed = now - started
                completed_targets = data.target_presentations_through_step(step)
                heartbeat = {
                    "schema_version": 1,
                    "marker": "HEARTBEAT",
                    "campaign": config.campaign,
                    "arm": config.arm,
                    "variant_id": config.variant_id,
                    "step": step,
                    "steps": config.total_steps,
                    "completed_passes": step / OPTIMIZER_STEPS_PER_PASS,
                    "target_presentations": completed_targets,
                    "total_target_presentations": data.coverage[
                        "total_target_presentations"
                    ],
                    "latest_train_nll": curve[-1]["train_nll"],
                    "interval_targets_per_second": interval_targets
                    / max(interval_elapsed, 1e-9),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed
                    / max(step - start_step, 1)
                    * (config.total_steps - step),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reads": config.reads,
                    "writes": config.writes,
                }
                atomic_json(output / "HEARTBEAT.json", heartbeat)
                print(json.dumps(heartbeat, sort_keys=True), flush=True)
                interval_targets = 0
                interval_started = time.perf_counter()

            if step in checkpoints:
                evaluation_started = time.perf_counter()
                validation = evaluate(model, data, "validation", device)
                evaluation_seconds += time.perf_counter() - evaluation_started
                if step == (OPTIMIZER_STEPS_PER_PASS + 1) // 2:
                    kind = "half_pass"
                    pass_value: float | int = step / OPTIMIZER_STEPS_PER_PASS
                else:
                    kind = "complete_pass"
                    pass_value = step // OPTIMIZER_STEPS_PER_PASS
                metric = {
                    "step": step,
                    "checkpoint_kind": kind,
                    "pass": pass_value,
                    "target_presentations": data.target_presentations_through_step(step),
                    "train_nll": curve[-1]["train_nll"],
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": lr,
                    "elapsed_seconds": time.perf_counter() - started,
                    "validation": validation,
                }
                metrics.append(metric)
                metric_handle.write(json.dumps(metric, sort_keys=True) + "\n")
                metric_handle.flush()
                print(json.dumps({"marker": "EVAL", **metric}, sort_keys=True), flush=True)
                interval_started = time.perf_counter()

            if step in recovery_steps:
                torch.cuda.synchronize(device)
                recovery_started = time.perf_counter()
                recovery = write_recovery(
                    output=output,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    manifest_sha256=manifest_sha,
                    metrics=metrics,
                    curve=curve,
                    complete=step == config.total_steps,
                )
                recovery_elapsed = time.perf_counter() - recovery_started
                recovery_seconds += recovery_elapsed
                recovery["write_and_hash_seconds"] = recovery_elapsed
                recoveries.append(recovery)
                atomic_json(
                    output / "RECOVERY_AUDIT.json",
                    {
                        "schema_version": 1,
                        "interval_steps": args.recovery_checkpoint_interval,
                        "checkpoints": recoveries,
                        "latest": recovery,
                    },
                )
                interval_started = time.perf_counter()

    budget = fixed_budget_record(metrics, passes=config.passes)
    atomic_json(output / "BUDGET.json", budget)
    complete_pass_metrics = {
        int(row["pass"]): row
        for row in metrics
        if row["checkpoint_kind"] == "complete_pass"
    }
    terminal_validation = complete_pass_metrics[config.passes]["validation"]
    terminal_test = None
    terminal_started = time.perf_counter()
    if budget["test_evaluation_allowed"]:
        terminal_test = evaluate(model, data, "test", device)
    evaluation_seconds += time.perf_counter() - terminal_started
    terminal_evaluation = {
        "schema_version": 1,
        "budget_frozen_before_test": budget["test_evaluation_allowed"],
        "checkpoint_rule": "terminal_frozen_budget_checkpoint",
        "validation": terminal_validation,
        "test": terminal_test,
        "test_status": "evaluated" if terminal_test is not None else "not_evaluated",
        "test_not_evaluated_reason": (
            None
            if terminal_test is not None
            else "fixed-budget terminal evaluation was not completed"
        ),
    }
    atomic_json(output / "TERMINAL_EVALUATION.json", terminal_evaluation)
    prior_diagnostics = final_prior_diagnostics(
        model,
        data,
        device,
        normal_validation=terminal_validation,
        initial=initial_prior,
    )
    atomic_json(output / "PRIOR_DIAGNOSTICS.json", prior_diagnostics)

    coverage = data.coverage_report()
    coverage["consumed_optimizer_steps"] = config.total_steps
    coverage["consumed_target_presentations"] = data.target_presentations_through_step(
        config.total_steps
    )
    coverage["complete"] = True
    coverage["all_passes_exactly_once_without_replacement"] = True
    atomic_json(output / "COVERAGE.json", coverage)
    total_seconds = time.perf_counter() - started
    result = {
        "schema": "factorized-sdm-canonical-wikitext103-language-v1",
        "status": "complete",
        "campaign": config.campaign,
        "arm": config.arm,
        "variant_id": config.variant_id,
        "seed": config.seed,
        "experiment_scope": config.experiment_scope,
        "protocol_id": PROTOCOL_ID,
        "project_model_identity": PROJECT_MODEL_IDENTITY,
        "model_identity": arm_identity,
        "config": asdict(config),
        "manifest_sha256": manifest_sha,
        "data_payload_sha256": data.manifest["remote_payload"]["sha256"],
        "campaign_design_sha256": config.campaign_design_sha256,
        "nonprior_initialization_sha256": matched_sha,
        "parameter_accounting": accounting,
        "prior_diagnostics": prior_diagnostics,
        "coverage": coverage,
        "budget": budget,
        "validation_by_pass": {
            str(index): complete_pass_metrics[index]["validation"]
            for index in range(1, config.passes + 1)
        },
        "half_pass_validation": next(
            row["validation"] for row in metrics if row["checkpoint_kind"] == "half_pass"
        ),
        "terminal_validation": terminal_validation,
        "terminal_test": terminal_test,
        "optimizer": {
            "name": "AdamW",
            "beta1": config.adam_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
            "gradient_clip": config.gradient_clip,
            "state_precision": optimizer.state_precision(),
            "schedule": "2.5_percent_linear_warmup_then_cosine_to_0.1_peak",
        },
        "precision": {
            "activations": "bfloat16",
            "optimizer_state": "float32",
        },
        "environment": environment,
        "resume": resume,
        "optimizer_seconds": optimizer_seconds,
        "evaluation_seconds": evaluation_seconds,
        "recovery_checkpoint_seconds": recovery_seconds,
        "total_seconds": total_seconds,
        "optimizer_measurement_targets": segment_targets,
        "optimizer_targets_per_second": segment_targets / optimizer_seconds,
        "peak_allocated_device_memory_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "metrics": metrics,
    }
    atomic_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "marker": "COMPLETE",
                "campaign": config.campaign,
                "arm": config.arm,
                "variant_id": config.variant_id,
                "budget": budget["status"],
                "validation_nll": terminal_validation["nll"],
                "test_nll": None if terminal_test is None else terminal_test["nll"],
                "target_presentations": coverage["consumed_target_presentations"],
                "optimizer_targets_per_second": result["optimizer_targets_per_second"],
                "peak_allocated_device_memory_bytes": result[
                    "peak_allocated_device_memory_bytes"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-design-sha256", required=True)
    parser.add_argument("--arm", choices=("zero", "product", "full"), required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument(
        "--prior-initialization",
        choices=("none", "product_factors", "independent_full"),
        required=True,
    )
    parser.add_argument(
        "--experiment-scope",
        choices=("single_seed",),
        required=True,
    )
    parser.add_argument("--passes", type=int, required=True)
    parser.add_argument("--schedule-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--attention-heads", type=int, required=True)
    parser.add_argument("--memory-heads", type=int, required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--reads", type=int, required=True)
    parser.add_argument("--writes", type=int, required=True)
    parser.add_argument("--mlp-expansion", type=float, required=True)
    parser.add_argument("--peak-learning-rate", type=float, required=True)
    parser.add_argument("--minimum-lr-ratio", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--adam-beta1", type=float, required=True)
    parser.add_argument("--adam-beta2", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--gradient-clip", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stream-seed", type=int, required=True)
    parser.add_argument("--initialization-device", choices=("cuda",), required=True)
    parser.add_argument("--activation-dtype", choices=("bfloat16",), required=True)
    parser.add_argument("--heartbeat-steps", type=int, default=25)
    parser.add_argument("--recovery-checkpoint-interval", type=int, default=2_400)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint-sha256")
    args = parser.parse_args()
    for name in (
        "passes",
        "schedule_steps",
        "batch_size",
        "micro_batch_size",
        "width",
        "attention_heads",
        "memory_heads",
        "slots",
        "reads",
        "writes",
        "warmup_steps",
        "heartbeat_steps",
        "recovery_checkpoint_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if len(args.campaign_design_sha256) != 64:
        parser.error("campaign-design-sha256 must be a SHA-256 digest")
    return args


if __name__ == "__main__":
    train(parse_args())
