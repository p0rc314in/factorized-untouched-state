#!/usr/bin/env python3
"""Run the three canonical seed-0 SDM initial-state arms."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.prepare_sdm_runtime import prepare_runtime


CAMPAIGN = "factorized-sdm-init-canonical-wt103-seed0-v6"
LOADER_PATCH_SHA256 = (
    "15751efa855485d3c2eabbaa8b0c686f94d11ee03d64366dc8d2cd6e7d36f574"
)
MANIFEST_SHA256 = "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"
PAYLOAD_SHA256 = "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e"
INVENTORY_SHA256 = "efa0a0b857d184dccdecf6adafda5d646ec45bae91f01a15cd1165f9fe9ad6b3"
ARMS = (
    ("zero", "none", 0),
    ("product", "product_factors", 57_344),
    ("full", "independent_full", 917_504),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_command(
    manifest: Path,
    output_root: Path,
    arm: tuple[str, str, int],
) -> list[str]:
    mode, initialization, _parameters = arm
    design_sha256 = sha256(ROOT / "PROJECT_IDENTITY.json")
    return [
        sys.executable,
        "-m",
        "benchmarks.train_canonical_wikitext_cuda",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_root / "seed-0" / mode),
        "--campaign",
        CAMPAIGN,
        "--campaign-design-sha256",
        design_sha256,
        "--arm",
        mode,
        "--variant-id",
        f"sdm_{mode}",
        "--prior-initialization",
        initialization,
        "--experiment-scope",
        "single_seed",
        "--passes",
        "3",
        "--schedule-steps",
        "21603",
        "--batch-size",
        "8",
        "--micro-batch-size",
        "1",
        "--layout",
        "BBBBBBBA",
        "--width",
        "128",
        "--attention-heads",
        "4",
        "--memory-heads",
        "1",
        "--slots",
        "1024",
        "--reads",
        "16",
        "--writes",
        "16",
        "--mlp-expansion",
        "4",
        "--peak-learning-rate",
        "0.0003",
        "--minimum-lr-ratio",
        "0.1",
        "--warmup-steps",
        "540",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--weight-decay",
        "0.01",
        "--gradient-clip",
        "1.0",
        "--seed",
        "0",
        "--stream-seed",
        "20260818",
        "--initialization-device",
        "cuda",
        "--activation-dtype",
        "bfloat16",
        "--heartbeat-steps",
        "50",
        "--recovery-checkpoint-interval",
        "2400",
    ]


def parse_gpus(value: str | None) -> list[str]:
    if value:
        result = [row.strip() for row in value.split(",") if row.strip()]
    else:
        result = [str(index) for index in range(torch.cuda.device_count())]
    if not result:
        raise SystemExit("a CUDA GPU is required; pass --gpus with visible device IDs")
    return result


def run_parallel(
    commands: list[tuple[str, list[str]]],
    gpus: list[str],
    *,
    extension_root: Path,
    runtime_root: Path,
) -> None:
    pending = deque(commands)
    active: dict[str, tuple[str, subprocess.Popen[bytes]]] = {}
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            name, command = pending.popleft()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["TORCH_EXTENSIONS_DIR"] = str(extension_root)
            environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (str(runtime_root), environment.get("PYTHONPATH"))
                if part
            )
            environment["SDM_SOURCE_ROOT"] = str(runtime_root)
            environment["SDM_LOADER_PATCH_SHA256"] = LOADER_PATCH_SHA256
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            print(f"START {name} on GPU {gpu}", flush=True)
            active[gpu] = (
                name,
                subprocess.Popen(command, cwd=ROOT, env=environment),
            )
        time.sleep(1)
        for gpu, (name, process) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[gpu]
            if code:
                for _, other in active.values():
                    other.terminate()
                raise SystemExit(f"{name} failed with exit code {code}")
            output = Path(process.args[process.args.index("--output-dir") + 1])
            (output / "recovery_checkpoint.pt").unlink(missing_ok=True)
            print(f"COMPLETE {name}", flush=True)


def verified_manifest(data_root: Path) -> Path:
    manifest = data_root / "wikitext103_gpt2_causal_t2048_coverage_v1" / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"missing prepared data: {manifest}")
    observed = sha256(manifest)
    if observed != MANIFEST_SHA256:
        raise SystemExit(
            f"prepared manifest SHA-256 is {observed}; expected {MANIFEST_SHA256}"
        )
    payload = json.loads(manifest.read_text())
    if payload["remote_payload"]["sha256"] != PAYLOAD_SHA256:
        raise SystemExit("prepared data payload identity changed")
    attestation_path = manifest.parent / "DOUBLE_BUILD.json"
    if not attestation_path.is_file():
        raise SystemExit(f"missing deterministic-build attestation: {attestation_path}")
    attestation = json.loads(attestation_path.read_text())
    expected_attestation = {
        "status": "byte_identical_double_build",
        "manifest_sha256": MANIFEST_SHA256,
        "remote_payload_sha256": PAYLOAD_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "passes": 3,
        "stream_seed": 20_260_818,
    }
    for key, expected in expected_attestation.items():
        if attestation.get(key) != expected:
            raise SystemExit(f"deterministic-build attestation changed at {key}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "runs/data")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/reproduction",
    )
    parser.add_argument("--gpus", help="comma-separated physical GPU IDs")
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    manifest = verified_manifest(args.data_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extension_root = output_root / "torch-extensions"
    extension_root.mkdir(parents=True, exist_ok=True)
    runtime_root = prepare_runtime(output_root / "runtime/sdm_patched")

    build_environment = os.environ.copy()
    build_environment["CUDA_VISIBLE_DEVICES"] = gpus[0]
    build_environment["TORCH_EXTENSIONS_DIR"] = str(extension_root)
    build_environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(runtime_root), build_environment.get("PYTHONPATH"))
        if part
    )
    build_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_released_sdm_extensions.py")],
        cwd=ROOT,
        env=build_environment,
        check=True,
    )

    print(
        "One seed by three initial-state representations; every arm uses "
        "released SDM with one memory head and R=W=16.",
        flush=True,
    )
    commands = [
        (
            f"seed-0/{mode}",
            arm_command(manifest, output_root, arm),
        )
        for arm in ARMS
        for mode, _initialization, _parameters in (arm,)
    ]
    run_parallel(
        commands,
        gpus,
        extension_root=extension_root,
        runtime_root=runtime_root,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_results.py"),
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
