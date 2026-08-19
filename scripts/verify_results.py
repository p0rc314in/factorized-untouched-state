#!/usr/bin/env python3
"""Verify committed measurements, formulas, figures, and provenance."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_recorded_results import (  # noqa: E402
    DERIVED_ACCOUNTING_FIELDS,
    RECORDED_DESIGN_SHA256,
    RECORDED_SOURCE_SHA256,
    extract,
    normalize_recorded_result,
)
from scripts.prepare_sdm_runtime import prepare_runtime  # noqa: E402


MODES = ("zero", "product", "full")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{left} != {right}")


def indexed_arms(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["initial_state"]: row for row in payload["arms"]}
    if set(rows) != set(MODES):
        raise AssertionError("committed seed-0 result matrix is incomplete")
    if any(int(row["seed"]) != 0 for row in rows.values()):
        raise AssertionError("committed result contains an unexpected seed")
    return rows


def png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise AssertionError(f"not a PNG: {path}")
    return struct.unpack(">II", payload[16:24])


def verify_metric(row: dict[str, Any]) -> None:
    total_nll = float(row["total_nll"])
    targets = int(row["scored_targets"])
    scored_bytes = int(row["scored_bytes"])
    correct = int(row["correct_targets"])
    close(float(row["nll"]), total_nll / targets)
    close(float(row["perplexity"]), math.exp(float(row["nll"])))
    close(
        float(row["bits_per_utf8_byte"]),
        total_nll / (math.log(2.0) * scored_bytes),
    )
    close(float(row["next_token_accuracy"]), correct / targets)


def verify_recorded_results(
    results: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    rows = indexed_arms(results)
    records = provenance["campaign"]["recorded_results"]
    if set(records) != set(MODES):
        raise AssertionError("recorded-result provenance is incomplete")

    expected_parameters = {"zero": 0, "product": 57_344, "full": 917_504}
    accelerators = set()
    nonprior_hashes = set()
    for mode in MODES:
        declaration = records[mode]
        path = ROOT / declaration["path"]
        observed_hash = sha256(path)
        if observed_hash != declaration["sha256"]:
            raise AssertionError(f"recorded result changed: {mode}")
        if (
            rows[mode]["source_record"] != declaration["path"]
            or rows[mode]["source_record_sha256"] != observed_hash
        ):
            raise AssertionError(f"compact result is not bound to {mode}")

        raw_record = json.loads(path.read_text())
        if (
            raw_record.get("status") != "decision_bearing_complete"
            or raw_record.get("campaign") != provenance["campaign"]["id"]
            or raw_record.get("arm") != mode
            or raw_record.get("seed") != 0
            or raw_record.get("protocol_id")
            != results["experiment"]["protocol_id"]
            or raw_record.get("campaign_design_sha256") != RECORDED_DESIGN_SHA256
        ):
            raise AssertionError(f"recorded result identity changed: {mode}")
        record = normalize_recorded_result(raw_record)
        coverage = record["coverage"]
        if (
            coverage.get("complete") is not True
            or coverage.get("all_passes_exactly_once_without_replacement") is not True
            or coverage.get("replacement_sampling") is not False
            or coverage.get("consumed_optimizer_steps") != 21_603
            or coverage.get("consumed_target_presentations") != 353_941_347
        ):
            raise AssertionError(f"recorded coverage changed: {mode}")
        budget = record["budget"]
        if (
            budget.get("budget_passes") != 3
            or budget.get("budget_selected_before_model_results") is not True
            or budget.get("validation_checkpoints_are_reporting_only") is not True
            or budget.get("test_evaluation_allowed") is not True
        ):
            raise AssertionError(f"recorded fixed budget changed: {mode}")

        accounting = record["parameter_accounting"]
        if (
            accounting["learned_initial_state_parameters"]
            != expected_parameters[mode]
            or accounting["input_embedding_parameters"] != 6_432_896
            or accounting["position_embedding_parameters"] != 262_144
            or accounting["output_embedding_parameters"] != 6_432_896
            or accounting["logical_recurrent_state"]["elements_per_sequence"]
            != 917_504
        ):
            raise AssertionError(f"recorded parameter accounting changed: {mode}")
        for split in ("validation", "test"):
            verify_metric(record[f"terminal_{split}"])
        accelerators.add(record["environment"]["gpu"])
        nonprior_hashes.add(record["nonprior_initialization_sha256"])
        if (
            record["environment"].get("sdm_source_revision")
            != provenance["implementation"]["upstream_revision"]
            or record["environment"].get("sdm_loader_patch_sha256")
            != provenance["implementation"]["loader_patch"]["sha256"]
            or record["environment"].get("sdm_model_and_kernel_mechanics_changed")
            is not False
        ):
            raise AssertionError(f"recorded SDM runtime provenance changed: {mode}")

    if accelerators != {"NVIDIA RTX A6000"}:
        raise AssertionError("recorded arms used different accelerators")
    if nonprior_hashes != {provenance["campaign"]["nonprior_initialization_sha256"]}:
        raise AssertionError("recorded non-prior initialization is not matched")

    extracted = extract()
    if results != extracted:
        raise AssertionError("data/results.json is not the exact recorded-result extract")


def verify_recorded_source(provenance: dict[str, Any]) -> None:
    declaration = provenance["campaign"]["recorded_source"]
    source_root = ROOT / declaration["path"]
    manifest_path = ROOT / declaration["manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        sha256(manifest_path) != declaration["manifest_sha256"]
        or manifest.get("source_payload_sha256") != RECORDED_SOURCE_SHA256
        or declaration["source_payload_sha256"] != RECORDED_SOURCE_SHA256
        or manifest.get("files") != declaration["members"]
    ):
        raise AssertionError("recorded training source manifest changed")
    observed = {
        relative: sha256(source_root / relative)
        for relative in declaration["members"]
    }
    if observed != declaration["members"]:
        raise AssertionError("recorded training source inventory changed")
    actual_files = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and path.name != "SOURCE.json"
    }
    if actual_files != set(declaration["members"]):
        raise AssertionError("recorded training source file set changed")
    trainer = (source_root / "benchmarks/train_canonical_wikitext_cuda.py").read_text()
    model = (source_root / "benchmarks/model.py").read_text()
    for expected in (
        '"status": "decision_bearing_complete"',
        '"flash_linear_attention": metadata.version("flash-linear-attention")',
        '"campaign_design_sha256": config.campaign_design_sha256',
    ):
        if expected not in trainer:
            raise AssertionError(f"recorded trainer output contract changed: {expected}")
    if '"input_embedding_parameters"' in model:
        raise AssertionError("recorded model unexpectedly emitted derived accounting")


def main() -> None:
    results_path = ROOT / "data/results.json"
    results = json.loads(results_path.read_text())
    if results.get("schema_version") != 3:
        raise AssertionError("unexpected committed-result schema")
    rows = indexed_arms(results)
    experiment = results["experiment"]
    factors = int(experiment["product_key_factors"])
    codebook = int(experiment["product_key_codebook_size"])
    slots = int(experiment["slots_per_head"])
    width = int(experiment["value_width"])
    layers = experiment["layout"].count("B")
    if factors != 2 or codebook**2 != slots:
        raise AssertionError("released SDM product-key shape changed")
    if experiment["reads"] != experiment["writes"]:
        raise AssertionError("released experiment no longer has balanced access")
    if (
        experiment["seed"] != 0
        or experiment["passes"] != 3
        or experiment["optimizer_steps_per_arm"] != 21_603
    ):
        raise AssertionError("canonical seed-0 budget changed")

    expected_parameters = {
        "zero": 0,
        "product": layers * 2 * math.isqrt(slots) * width,
        "full": layers * slots * width,
    }
    observed_parameters = {
        mode: rows[mode]["parameters"]["learned_initial_state"] for mode in MODES
    }
    if observed_parameters != expected_parameters:
        raise AssertionError("initial-state parameter formula changed")
    if {
        rows[mode]["parameters"]["logical_recurrent_state"]["elements_per_sequence"]
        for mode in MODES
    } != {917_504}:
        raise AssertionError("logical recurrent state differs across arms")

    terminal = results["terminal_nll"]
    for split in ("validation", "test"):
        for mode in MODES:
            close(float(rows[mode][split]["nll"]), float(terminal[split][mode]))
        differences = results["differences"][split]
        close(
            float(differences["product_minus_zero"]),
            float(terminal[split]["product"]) - float(terminal[split]["zero"]),
        )
        close(
            float(differences["product_minus_full"]),
            float(terminal[split]["product"]) - float(terminal[split]["full"]),
        )

    trajectory = results["validation_trajectory"]
    expected_checkpoints = [(0.5, 3_601), (1.0, 7_201), (2.0, 14_402), (3.0, 21_603)]
    observed_checkpoints = [
        (float(row["passes"]), int(row["step"])) for row in trajectory
    ]
    if observed_checkpoints != expected_checkpoints:
        raise AssertionError("validation trajectory changed")
    for mode in MODES:
        close(
            float(trajectory[-1]["validation_nll"][mode]),
            float(terminal["validation"][mode]),
        )

    provenance = json.loads((ROOT / "provenance.json").read_text())
    if provenance.get("schema_version") != 3:
        raise AssertionError("unexpected provenance schema")
    expected_extraction = {
        "script": "scripts/extract_recorded_results.py",
        "derived_fields": list(DERIVED_ACCOUNTING_FIELDS),
    }
    if provenance["campaign"].get("result_extraction") != expected_extraction:
        raise AssertionError("recorded-result extraction declaration changed")
    expected_result_provenance = {
        "source": "recorded_result_records",
        "recorded_source_payload_sha256": RECORDED_SOURCE_SHA256,
        "derived_accounting_fields": list(DERIVED_ACCOUNTING_FIELDS),
    }
    if results.get("record_provenance") != expected_result_provenance:
        raise AssertionError("compact result provenance changed")
    verify_recorded_source(provenance)
    verify_recorded_results(results, provenance)

    source_root = ROOT / "third_party/released_sdm"
    source_manifest_path = source_root / "SOURCE.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    observed_source = {
        relative: sha256(source_root / relative)
        for relative in sorted(source_manifest["files"])
    }
    if observed_source != source_manifest["files"] or source_manifest["modified"] is not False:
        raise AssertionError("vendored released SDM source changed")
    if (
        provenance["implementation"]["vendored_source_manifest_sha256"]
        != sha256(source_manifest_path)
    ):
        raise AssertionError("source manifest provenance changed")
    patch_declaration = provenance["implementation"]["loader_patch"]
    patch_path = ROOT / patch_declaration["path"]
    patch_manifest_path = ROOT / patch_declaration["manifest"]
    patch_manifest = json.loads(patch_manifest_path.read_text())
    if (
        sha256(patch_path) != patch_declaration["sha256"]
        or patch_manifest["patch_sha256"] != patch_declaration["sha256"]
        or patch_manifest["base_revision"]
        != provenance["implementation"]["upstream_revision"]
    ):
        raise AssertionError("loader-only SDM patch provenance changed")
    with tempfile.TemporaryDirectory(prefix="verify-sdm-runtime-") as temporary:
        prepare_runtime(Path(temporary) / "sdm_patched")
    if (
        provenance["data"]["manifest_sha256"]
        != "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"
        or provenance["data"]["payload_sha256"]
        != "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e"
    ):
        raise AssertionError("canonical data provenance changed")

    figure = ROOT / "figures/validation-nll-by-pass.png"
    figure_dimensions = tuple(
        int(value) for value in provenance["figures"]["validation_nll_by_pass_dimensions"]
    )
    if figure_dimensions != (1_720, 840) or png_dimensions(figure) != figure_dimensions:
        raise AssertionError("canonical result figure dimensions changed")
    if sha256(figure) != provenance["figures"]["validation_nll_by_pass_sha256"]:
        raise AssertionError("canonical result figure hash changed")

    social_preview = ROOT / "figures/social-preview.png"
    social_dimensions = tuple(
        int(value) for value in provenance["figures"]["social_preview_dimensions"]
    )
    if social_dimensions != (1_280, 640) or png_dimensions(social_preview) != social_dimensions:
        raise AssertionError("social preview dimensions changed")
    if sha256(social_preview) != provenance["figures"]["social_preview_sha256"]:
        raise AssertionError("social preview hash changed")

    for relative, expected in provenance["released_sources"].items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise AssertionError(f"released source changed: {relative}")

    print(
        json.dumps(
            {
                "status": "passed",
                "results_sha256": sha256(results_path),
                "recorded_result_sha256": {
                    mode: rows[mode]["source_record_sha256"] for mode in MODES
                },
                "structural_parameter_law": "N d_v -> 2 sqrt(N) d_v",
                "tested_instance": {"slots": slots, "value_width": width},
                "terminal_test_nll": terminal["test"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
