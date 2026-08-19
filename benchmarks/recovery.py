"""Strict checkpoint validation and restoration for interrupted scale runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class RecoveryError(ValueError):
    """A checkpoint cannot safely continue the requested experiment."""


def _ordered_steps(rows: list[dict[str, Any]], *, name: str) -> list[int]:
    steps: list[int] = []
    for row in rows:
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise RecoveryError(f"{name} contains an invalid step")
        steps.append(step)
    if steps != sorted(set(steps)):
        raise RecoveryError(f"{name} steps are not strictly increasing")
    return steps


def load_recovery(
    path: Path,
    *,
    expected_sha256: str,
    expected_kind: str,
    expected_config: dict[str, Any],
    expected_manifest_sha256: str,
    sha256_file: Any,
) -> dict[str, Any]:
    """Load a trusted local checkpoint only after its full identity matches."""

    path = path.resolve()
    if not path.is_file():
        raise RecoveryError(f"recovery checkpoint is absent: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RecoveryError(
            f"recovery SHA-256 changed: {observed} != {expected_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RecoveryError("unexpected recovery checkpoint schema")
    if payload.get("kind") != expected_kind:
        raise RecoveryError(
            f"recovery kind changed: {payload.get('kind')!r} != {expected_kind!r}"
        )
    if payload.get("config") != expected_config:
        raise RecoveryError("recovery configuration does not match this arm")
    if payload.get("manifest_sha256") != expected_manifest_sha256:
        raise RecoveryError("recovery data manifest does not match this arm")
    step = payload.get("step")
    total_steps = expected_config.get("steps", expected_config.get("total_steps"))
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or not 0 < step < total_steps
    ):
        raise RecoveryError("recovery step is outside the resumable range")
    for name in ("model", "optimizer"):
        if not isinstance(payload.get(name), dict):
            raise RecoveryError(f"recovery checkpoint lacks {name} state")
    metrics = payload.get("metrics")
    curve = payload.get("training_curve")
    if not isinstance(metrics, list) or not isinstance(curve, list):
        raise RecoveryError("recovery checkpoint lacks metric history")
    metric_steps = _ordered_steps(metrics, name="metrics")
    curve_steps = _ordered_steps(curve, name="training_curve")
    if not curve_steps or curve_steps[-1] != step:
        raise RecoveryError("training curve does not end at the recovery step")
    if metric_steps and metric_steps[-1] > step:
        raise RecoveryError("metric history extends beyond the recovery step")
    if not isinstance(payload.get("torch_rng_state"), torch.Tensor):
        raise RecoveryError("recovery checkpoint lacks the CPU RNG state")
    cuda_states = payload.get("cuda_rng_states")
    if (
        not isinstance(cuda_states, list)
        or not cuda_states
        or any(not isinstance(value, torch.Tensor) for value in cuda_states)
    ):
        raise RecoveryError("recovery checkpoint lacks CUDA RNG states")
    payload["_sha256"] = observed
    payload["_path"] = str(path)
    return payload


def restore_recovery(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Restore every state that can affect the subsequent parameter trajectory."""

    model.load_state_dict(payload.pop("model"), strict=True)
    optimizer.load_state_dict(payload.pop("optimizer"))
    restore_rng(payload)


def restore_rng(payload: dict[str, Any]) -> None:
    """Restore CPU and CUDA generators after any deterministic setup checks."""

    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(
        [value.cpu() for value in payload["cuda_rng_states"]]
    )


def recovery_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return provenance safe to store in final result artifacts."""

    return {
        "resumed": True,
        "kind": payload["kind"],
        "step": payload["step"],
        "sha256": payload["_sha256"],
    }
