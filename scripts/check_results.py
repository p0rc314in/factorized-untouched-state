#!/usr/bin/env python3
"""Validate fresh canonical runs and extract compact measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODES = ("zero", "product", "full")
EXPECTED_PARAMETERS = {"zero": 0, "product": 57_344, "full": 917_504}
EXPECTED_CAMPAIGN = "factorized-sdm-init-canonical-wt103-seed0-v6"
EXPECTED_MANIFEST = "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"
EXPECTED_PAYLOAD = "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e"
EXPECTED_LOADER_PATCH = (
    "15751efa855485d3c2eabbaa8b0c686f94d11ee03d64366dc8d2cd6e7d36f574"
)
EXPECTED_STEPS = {0.5: 3_601, 1.0: 7_201, 2.0: 14_402, 3.0: 21_603}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def arm_result(output_root: Path, mode: str) -> dict[str, Any]:
    arm_root = output_root / "seed-0" / mode
    result_path = arm_root / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"missing result: {result_path}")
    result = load_json(result_path)
    config = result["config"]
    expected_config = {
        "campaign": EXPECTED_CAMPAIGN,
        "arm": mode,
        "passes": 3,
        "total_steps": 21_603,
        "schedule_steps": 21_603,
        "batch_size": 8,
        "micro_batch_size": 1,
        "sequence_length": 2_048,
        "evaluation_stride": 512,
        "vocab_size": 50_257,
        "layout": "BBBBBBBA",
        "width": 128,
        "attention_heads": 4,
        "memory_heads": 1,
        "slots": 1_024,
        "reads": 16,
        "writes": 16,
        "seed": 0,
        "stream_seed": 20_260_818,
        "initialization_device": "cuda",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(f"seed 0 {mode}: {key} changed")
    if result.get("campaign") != EXPECTED_CAMPAIGN:
        raise RuntimeError(f"seed 0 {mode}: campaign identity changed")
    if config["reads"] != config["writes"]:
        raise RuntimeError(f"seed 0 {mode}: sparse access is not balanced")
    if result["manifest_sha256"] != EXPECTED_MANIFEST:
        raise RuntimeError(f"seed 0 {mode}: data manifest changed")
    if result["data_payload_sha256"] != EXPECTED_PAYLOAD:
        raise RuntimeError(f"seed 0 {mode}: data payload changed")
    if result["protocol_id"] != "wikitext103-gpt2-causal-t2048-coverage-v1":
        raise RuntimeError(f"seed 0 {mode}: evaluation protocol changed")
    environment = result["environment"]
    if (
        environment.get("sdm_source_revision")
        != "183e7df809131b80ad4393741029d0f20fc3640b"
        or environment.get("sdm_loader_patch_sha256") != EXPECTED_LOADER_PATCH
        or environment.get("sdm_model_and_kernel_mechanics_changed") is not False
    ):
        raise RuntimeError(f"seed 0 {mode}: SDM runtime provenance changed")

    accounting = result["parameter_accounting"]
    if accounting["learned_initial_state_parameters"] != EXPECTED_PARAMETERS[mode]:
        raise RuntimeError(f"seed 0 {mode}: initial-state parameter count changed")
    if (
        accounting["input_embedding_parameters"] != 6_432_896
        or accounting["position_embedding_parameters"] != 262_144
        or accounting["output_embedding_parameters"] != 6_432_896
    ):
        raise RuntimeError(f"seed 0 {mode}: embedding parameter count changed")
    if accounting["logical_recurrent_state"]["elements_per_sequence"] != 917_504:
        raise RuntimeError(f"seed 0 {mode}: logical recurrent state changed")

    causality = load_json(arm_root / "CAUSALITY.json")
    if causality.get("maximum_prefix_difference") != 0.0:
        raise RuntimeError(f"seed 0 {mode}: prefix causality failed")
    coverage = result["coverage"]
    if (
        coverage.get("complete") is not True
        or coverage.get("all_passes_exactly_once_without_replacement") is not True
        or coverage.get("consumed_optimizer_steps") != 21_603
        or coverage.get("consumed_target_presentations") != 353_941_347
    ):
        raise RuntimeError(f"seed 0 {mode}: canonical coverage is incomplete")
    budget = result["budget"]
    if (
        budget.get("budget_passes") != 3
        or budget.get("budget_selected_before_model_results") is not True
        or budget.get("validation_checkpoints_are_reporting_only") is not True
    ):
        raise RuntimeError(f"seed 0 {mode}: fixed-budget protocol changed")

    for field in ("terminal_validation", "terminal_test"):
        nll = float(result[field]["nll"])
        if not math.isfinite(nll) or not 4.3 <= nll <= 4.8:
            raise RuntimeError(f"seed 0 {mode}: {field} NLL outside reproduction range")
    return result


def validation_trajectory(result: dict[str, Any]) -> dict[float, float]:
    observed_steps = [int(row["step"]) for row in result["metrics"]]
    if observed_steps != list(EXPECTED_STEPS.values()):
        raise RuntimeError("validation checkpoint steps changed")
    trajectory = {
        0.5: float(result["half_pass_validation"]["nll"]),
        **{
            float(pass_index): float(values["nll"])
            for pass_index, values in result["validation_by_pass"].items()
        },
    }
    if set(trajectory) != set(EXPECTED_STEPS):
        raise RuntimeError("validation checkpoint schedule changed")
    return trajectory


def extract(output_root: Path) -> dict[str, Any]:
    results = {mode: arm_result(output_root, mode) for mode in MODES}
    hashes = {
        result["nonprior_initialization_sha256"]
        for result in results.values()
    }
    if len(hashes) != 1:
        raise RuntimeError("non-initial-state parameters are not matched across arms")
    accelerators = {result["environment"]["gpu"] for result in results.values()}
    if len(accelerators) != 1:
        raise RuntimeError("comparison arms used different accelerators")

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        result = results[mode]
        prior = result["prior_diagnostics"]
        row: dict[str, Any] = {
            "seed": 0,
            "initial_state": mode,
            "learned_initial_state_parameters": result["parameter_accounting"][
                "learned_initial_state_parameters"
            ],
            "validation_nll": result["terminal_validation"]["nll"],
            "test_nll": result["terminal_test"]["nll"],
            "validation_token_accuracy": result["terminal_validation"][
                "next_token_accuracy"
            ],
            "test_token_accuracy": result["terminal_test"]["next_token_accuracy"],
            "optimizer_targets_per_second": result["optimizer_targets_per_second"],
            "peak_allocated_device_memory_bytes": result[
                "peak_allocated_device_memory_bytes"
            ],
        }
        if mode != "zero":
            row["validation_zeroing_penalty"] = prior["validation_nll_penalty"]["zero"]
        if mode == "full":
            row["validation_additive_projection_penalty"] = prior[
                "validation_nll_penalty"
            ]["additive_projection"]
            row["mean_additive_variance_explained"] = sum(
                float(layer["additive_variance_explained"])
                for layer in prior["final"]["layers"]
            ) / len(prior["final"]["layers"])
        rows.append(row)

    trajectories = {
        mode: validation_trajectory(result)
        for mode, result in results.items()
    }
    checkpoints = [
        {
            "passes": pass_count,
            "step": step,
            "validation_nll": {
                mode: trajectories[mode][pass_count]
                for mode in MODES
            },
        }
        for pass_count, step in EXPECTED_STEPS.items()
    ]
    terminal = {
        split: {
            mode: float(results[mode][f"terminal_{split}"]["nll"])
            for mode in MODES
        }
        for split in ("validation", "test")
    }
    for split, values in terminal.items():
        if not (
            values["product"] < values["zero"]
            and values["full"] < values["zero"]
        ):
            raise RuntimeError(
                f"{split}: learned initial states did not improve over zero"
            )
        if abs(values["product"] - values["full"]) >= abs(
            values["product"] - values["zero"]
        ):
            raise RuntimeError(
                f"{split}: factorized state was not closer to full than zero"
            )
    return {
        "schema_version": 1,
        "arms": rows,
        "terminal_nll": terminal,
        "differences": {
            "validation": {
                "product_minus_zero": terminal["validation"]["product"]
                - terminal["validation"]["zero"],
                "product_minus_full": terminal["validation"]["product"]
                - terminal["validation"]["full"],
            },
            "test": {
                "product_minus_zero": terminal["test"]["product"]
                - terminal["test"]["zero"],
                "product_minus_full": terminal["test"]["product"]
                - terminal["test"]["full"],
            },
        },
        "validation_trajectory": checkpoints,
        "invariants": {
            "accelerator": accelerators.pop(),
            "manifest_sha256": EXPECTED_MANIFEST,
            "data_payload_sha256": EXPECTED_PAYLOAD,
            "nonprior_initialization_sha256": hashes.pop(),
            "balanced_reads_and_writes": 16,
            "logical_recurrent_state_elements_per_sequence": 917_504,
            "passes": 3,
            "optimizer_steps_per_arm": 21_603,
            "target_presentations_per_arm": 353_941_347,
            "validation_scored_targets_per_arm": 247_416,
            "test_scored_targets_per_arm": 283_426,
        },
    }


def write_outputs(output_root: Path, measurements: dict[str, Any]) -> None:
    measurements_path = output_root / "measurements.json"
    measurements_path.write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n"
    )
    rows = measurements["arms"]
    fields = sorted({key for row in rows for key in row})
    with (output_root / "results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": "passed",
                "measurements": str(measurements_path),
                "sha256": hashlib.sha256(measurements_path.read_bytes()).hexdigest(),
                "test_nll": measurements["terminal_nll"]["test"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/reproduction",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    write_outputs(output_root, extract(output_root))


if __name__ == "__main__":
    main()
