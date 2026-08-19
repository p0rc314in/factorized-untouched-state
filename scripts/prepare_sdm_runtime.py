#!/usr/bin/env python3
"""Materialize the exact loader-patched SDM runtime used by the experiment."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "third_party/released_sdm"
PATCH_MANIFEST_PATH = ROOT / "third_party/patches/sdm-current-prebuilt-loader.json"
RUNTIME_MANIFEST_NAME = "RUNTIME_SOURCE.json"
EXPECTED_PATCHED_PATHS = {
    "lingua/sparse_delta_memory/cuda/_prebuilt.py",
    "lingua/sparse_delta_memory/cuda/sparse_ip_cuda.py",
    "lingua/sparse_delta_memory/cuda/warp_cooperative_gather_cuda.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if (
            path.is_file()
            and path.name != RUNTIME_MANIFEST_NAME
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    }


def verified_base_manifest() -> dict[str, Any]:
    manifest_path = BASE_ROOT / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text())
    observed = {
        relative: sha256(BASE_ROOT / relative)
        for relative in sorted(manifest["files"])
    }
    if manifest.get("modified") is not False or observed != manifest["files"]:
        raise RuntimeError("vendored SDM base does not match SOURCE.json")
    return manifest


def verified_patch_manifest() -> tuple[dict[str, Any], Path]:
    manifest = json.loads(PATCH_MANIFEST_PATH.read_text())
    patch = PATCH_MANIFEST_PATH.with_name(manifest["patch"])
    if sha256(patch) != manifest["patch_sha256"]:
        raise RuntimeError("SDM loader patch hash changed")
    if manifest["base_revision"] != verified_base_manifest()["revision"]:
        raise RuntimeError("SDM loader patch targets a different base revision")
    targets = {
        right
        for left, right in re.findall(
            r"^diff --git a/(.+) b/(.+)$",
            patch.read_text(),
            flags=re.MULTILINE,
        )
        if left == right
    }
    if targets != EXPECTED_PATCHED_PATHS:
        raise RuntimeError(f"unexpected SDM loader patch targets: {sorted(targets)}")
    return manifest, patch


def string_constant(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"missing string constant {name} in {path}")


def verify_loader_only_runtime(runtime_root: Path) -> None:
    manifest, _patch = verified_patch_manifest()
    runtime_manifest_path = runtime_root / RUNTIME_MANIFEST_NAME
    runtime_manifest = json.loads(runtime_manifest_path.read_text())
    expected_header = {
        "schema_version": 1,
        "base_revision": manifest["base_revision"],
        "base_source_manifest_sha256": sha256(BASE_ROOT / "SOURCE.json"),
        "loader_patch_sha256": manifest["patch_sha256"],
        "changed_paths": sorted(EXPECTED_PATCHED_PATHS),
        "model_and_kernel_mechanics_changed": False,
    }
    for key, expected in expected_header.items():
        if runtime_manifest.get(key) != expected:
            raise RuntimeError(f"patched SDM runtime metadata changed at {key}")
    if file_inventory(runtime_root) != runtime_manifest.get("files"):
        raise RuntimeError("patched SDM runtime file inventory changed")

    base_files = {
        path.relative_to(BASE_ROOT).as_posix(): path
        for path in BASE_ROOT.rglob("*")
        if path.is_file()
    }
    runtime_files = {
        path.relative_to(runtime_root).as_posix(): path
        for path in runtime_root.rglob("*")
        if (
            path.is_file()
            and path.name != RUNTIME_MANIFEST_NAME
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    }
    if set(runtime_files) - set(base_files) != {
        "lingua/sparse_delta_memory/cuda/_prebuilt.py"
    }:
        raise RuntimeError("loader patch introduced an unexpected file")
    for relative, base_path in base_files.items():
        runtime_path = runtime_files.get(relative)
        if runtime_path is None:
            raise RuntimeError(f"loader patch removed {relative}")
        if (
            relative not in EXPECTED_PATCHED_PATHS
            and sha256(runtime_path) != sha256(base_path)
        ):
            raise RuntimeError(f"loader patch unexpectedly changed {relative}")

    for relative in (
        "lingua/sparse_delta_memory/cuda/sparse_ip_cuda.py",
        "lingua/sparse_delta_memory/cuda/warp_cooperative_gather_cuda.py",
    ):
        for constant in ("_CPP_SOURCE", "_CUDA_SOURCE"):
            if string_constant(BASE_ROOT / relative, constant) != string_constant(
                runtime_root / relative,
                constant,
            ):
                raise RuntimeError(f"loader patch changed {relative}:{constant}")


def prepare_runtime(output: Path) -> Path:
    """Create or verify one checksum-bound loader-patched source tree."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        verify_loader_only_runtime(output)
        return output

    base_manifest = verified_base_manifest()
    patch_manifest, patch = verified_patch_manifest()
    with tempfile.TemporaryDirectory(
        prefix="sdm-runtime-",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary) / "sdm_patched"
        shutil.copytree(BASE_ROOT, staging)
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=staging, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=staging, check=True)
        runtime_manifest = {
            "schema_version": 1,
            "base_revision": base_manifest["revision"],
            "base_source_manifest_sha256": sha256(BASE_ROOT / "SOURCE.json"),
            "loader_patch_sha256": patch_manifest["patch_sha256"],
            "changed_paths": sorted(EXPECTED_PATCHED_PATHS),
            "model_and_kernel_mechanics_changed": False,
            "files": file_inventory(staging),
        }
        (staging / RUNTIME_MANIFEST_NAME).write_text(
            json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_loader_only_runtime(staging)
        os.replace(staging, output)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(prepare_runtime(arguments.output))
