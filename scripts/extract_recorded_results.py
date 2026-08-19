#!/usr/bin/env python3
"""Extract the compact result table from the recorded arm records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODES = ("zero", "product", "full")
PROTOCOL_ID = "wikitext103-gpt2-causal-t2048-coverage-v1"
EXPECTED_STEPS = {0.5: 3_601, 1.0: 7_201, 2.0: 14_402, 3.0: 21_603}
RECORDED_CAMPAIGN = "factorized-sdm-init-canonical-wt103-seed0-v6"
RECORDED_DESIGN_SHA256 = (
    "cd67e9964c830a2583688b8eb331a89724c2b31cccd18cdf0a12a0804b18c0d5"
)
RECORDED_SOURCE_SHA256 = (
    "3cb3a8f3ce0b3fbbd7a00f06f52571922982d147a3f810249162beb5a39a0f12"
)
DERIVED_ACCOUNTING_FIELDS = (
    "input_embedding_parameters",
    "output_embedding_parameters",
    "position_embedding_parameters",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_recorded_result(record: dict[str, Any]) -> dict[str, Any]:
    """Validate record identity and add deterministic parameter counts."""

    normalized = json.loads(json.dumps(record))
    if normalized.get("campaign_design_sha256") != RECORDED_DESIGN_SHA256:
        raise ValueError("unexpected recorded campaign design")
    config = normalized["config"]
    if config.get("campaign_design_sha256") != RECORDED_DESIGN_SHA256:
        raise ValueError("unexpected recorded config design")

    accounting = normalized["parameter_accounting"]
    vocabulary = int(config["vocab_size"])
    width = int(config["width"])
    sequence_length = int(config["sequence_length"])
    accounting["input_embedding_parameters"] = vocabulary * width
    accounting["position_embedding_parameters"] = sequence_length * width
    accounting["output_embedding_parameters"] = vocabulary * width
    return normalized


def load_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        path = ROOT / "data" / "recorded-results" / f"{mode}.json"
        raw_record = json.loads(path.read_text())
        if (
            raw_record.get("arm") != mode
            or raw_record.get("seed") != 0
            or raw_record.get("protocol_id") != PROTOCOL_ID
            or raw_record.get("status") != "decision_bearing_complete"
            or raw_record.get("campaign") != RECORDED_CAMPAIGN
        ):
            raise ValueError(f"unexpected recorded result identity: {mode}")
        record = normalize_recorded_result(raw_record)
        record["_path"] = path
        records[mode] = record
    return records


def execution(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "accelerator": record["environment"]["gpu"],
        "total_seconds": record["total_seconds"],
        "optimizer_seconds": record["optimizer_seconds"],
        "evaluation_seconds": record["evaluation_seconds"],
        "recovery_checkpoint_seconds": record["recovery_checkpoint_seconds"],
        "optimizer_targets_per_second": record["optimizer_targets_per_second"],
        "peak_allocated_device_memory_bytes": record[
            "peak_allocated_device_memory_bytes"
        ],
    }


def extract() -> dict[str, Any]:
    records = load_records()
    first = records["zero"]
    config = first["config"]
    hashes = {
        record["nonprior_initialization_sha256"] for record in records.values()
    }
    if len(hashes) != 1:
        raise ValueError("non-initial-state initialization differs across arms")

    arms: list[dict[str, Any]] = []
    for mode in MODES:
        record = records[mode]
        accounting = record["parameter_accounting"]
        path = record.pop("_path")
        arms.append(
            {
                "seed": 0,
                "initial_state": mode,
                "source_record": path.relative_to(ROOT).as_posix(),
                "source_record_sha256": sha256(path),
                "parameters": {
                    "active_excluding_learned_initial_state": accounting[
                        "active_parameters_excluding_learned_initial_state"
                    ],
                    "input_embedding": accounting["input_embedding_parameters"],
                    "position_embedding": accounting[
                        "position_embedding_parameters"
                    ],
                    "output_embedding": accounting["output_embedding_parameters"],
                    "learned_initial_state": accounting[
                        "learned_initial_state_parameters"
                    ],
                    "total_trainable": accounting["total_trainable_parameters"],
                    "logical_recurrent_state": accounting[
                        "logical_recurrent_state"
                    ],
                },
                "validation": record["terminal_validation"],
                "test": record["terminal_test"],
                "execution": execution(record),
                "hashes": {
                    "data_manifest": record["manifest_sha256"],
                    "data_payload": record["data_payload_sha256"],
                    "nonprior_initialization": record[
                        "nonprior_initialization_sha256"
                    ],
                },
            }
        )

    terminal = {
        split: {
            mode: float(records[mode][f"terminal_{split}"]["nll"])
            for mode in MODES
        }
        for split in ("validation", "test")
    }
    aggregate_seconds = sum(float(records[mode]["total_seconds"]) for mode in MODES)
    slowest_arm_seconds = max(
        float(records[mode]["total_seconds"]) for mode in MODES
    )
    checkpoints = []
    for pass_count, step in EXPECTED_STEPS.items():
        checkpoints.append(
            {
                "passes": pass_count,
                "step": step,
                "validation_nll": {
                    mode: float(
                        records[mode]["half_pass_validation"]["nll"]
                        if pass_count == 0.5
                        else records[mode]["validation_by_pass"][
                            str(int(pass_count))
                        ]["nll"]
                    )
                    for mode in MODES
                },
            }
        )

    return {
        "schema_version": 3,
        "record_provenance": {
            "source": "recorded_result_records",
            "recorded_source_payload_sha256": RECORDED_SOURCE_SHA256,
            "derived_accounting_fields": list(DERIVED_ACCOUNTING_FIELDS),
        },
        "experiment": {
            "task": "wikitext103_gpt2_causal_language_modeling",
            "protocol_id": PROTOCOL_ID,
            "seed": 0,
            "passes": config["passes"],
            "optimizer_steps_per_arm": config["total_steps"],
            "target_presentations_per_arm": first["coverage"][
                "consumed_target_presentations"
            ],
            "validation_scored_targets_per_arm": first["terminal_validation"][
                "scored_targets"
            ],
            "test_scored_targets_per_arm": first["terminal_test"]["scored_targets"],
            "layout": config["layout"],
            "width": config["width"],
            "attention_heads": config["attention_heads"],
            "memory_heads": config["memory_heads"],
            "slots_per_head": config["slots"],
            "product_key_factors": 2,
            "product_key_codebook_size": records["product"]["model_identity"][
                "product_key_codebook_size"
            ],
            "value_width": records["product"]["model_identity"][
                "value_width_per_memory_head"
            ],
            "reads": config["reads"],
            "writes": config["writes"],
        },
        "execution_summary": {
            "aggregate_gpu_seconds": aggregate_seconds,
            "aggregate_gpu_hours": aggregate_seconds / 3_600,
            "slowest_arm_seconds": slowest_arm_seconds,
            "slowest_arm_hours": slowest_arm_seconds / 3_600,
        },
        "arms": arms,
        "terminal_nll": terminal,
        "differences": {
            split: {
                "product_minus_zero": terminal[split]["product"]
                - terminal[split]["zero"],
                "product_minus_full": terminal[split]["product"]
                - terminal[split]["full"],
            }
            for split in ("validation", "test")
        },
        "validation_trajectory": checkpoints,
        "interpretation": "single_canonical_seed0_full_coverage_comparison",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "results.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.write_text(json.dumps(extract(), indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
