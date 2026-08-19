#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
gpus="${GPUS:-}"
gpu_args=()
if [[ -n "$gpus" ]]; then
  gpu_args=(--gpus "$gpus")
fi

case "$stage" in
  prepare)
    python3 scripts/prepare_canonical_wikitext103.py \
      --output runs/data/wikitext103_gpt2_causal_t2048_coverage_v1 \
      --download-cache runs/data/.cache/huggingface-wikitext103
    ;;
  train)
    python3 scripts/run_experiments.py "${gpu_args[@]}"
    ;;
  verify-results)
    python3 scripts/verify_results.py
    ;;
  all)
    "$0" prepare
    python3 scripts/run_experiments.py "${gpu_args[@]}"
    ;;
  *)
    echo "usage: ./reproduce.sh [prepare|train|verify-results|all]" >&2
    exit 2
    ;;
esac
