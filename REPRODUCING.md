# Reproducing the experiment

The reproduction reruns the recorded seed-0 comparison: zero, factorized, and
learned initial memory. Every arm uses one SDM memory head and balanced R = W =
16 access.

## Environment and data

Linux x86-64, Python 3.12 or 3.13, a CUDA 12.8 development toolchain with a
host C++ compiler, and CUDA GPUs are required. The checked-in lockfile fixes
PyTorch 2.11, Triton 3.6, and the local data-preparation dependencies.

Prepare WikiText-103 on a CPU machine before allocating experiment GPUs:

```bash
uv sync --frozen
uv run --no-sync ./reproduce.sh prepare
```

Preparation downloads the pinned public WikiText-103 Parquet shards, performs
two byte-identical builds, and writes the immutable GPT-2 token streams and
indexes under `runs/data/`. Allow roughly 3 GB of temporary disk space. The
finished training payload is about 227 MB; the retained local tree is larger
because it also includes source and serialized-text provenance.

The generated manifest must have SHA-256:

```text
fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6
```

Its immutable training and evaluation payload must have SHA-256:

```text
f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e
```

## Training

The recorded experiment used one NVIDIA RTX A6000 per arm. To run all three
arms concurrently, use three identical GPUs:

```bash
GPUS=0,1,2 uv run --no-sync ./reproduce.sh train
```

With fewer GPUs, arms queue automatically. From a clean checkout, the complete
command performs local data preparation followed by training:

```bash
GPUS=0,1,2 uv run --no-sync ./reproduce.sh all
```

The command applies the recorded loader-only patch to the pinned released-SDM
source, builds its two CUDA extensions once, then starts the matched arms. Each
arm trains for 21,603 steps, or 353,941,347 target presentations. The recorded
process times total 11.53 RTX A6000 GPU-hours; the slowest arm took 4.06 hours.
Budget about 4.5 hours of wall time with three matching GPUs, or 12 aggregate
GPU-hours. Expected cost is 12 times the current per-GPU-hour price.

Expected terminal validation and test NLL are approximately 4.45–4.60. The
reproduction checker uses the broader 4.3–4.8 range to accommodate hardware
variation from BF16 atomic accumulation.

Fresh arm artifacts appear under:

```text
runs/reproduction/seed-0/{zero,product,full}/
```

Here `product` is the factorized arm and `full` is the fully learned arm.

The run then writes `measurements.json` and `results.csv` under
`runs/reproduction/` and checks:

- the exact data, model, access, schedule, parameter, and evaluation identities;
- bit-identical parameters outside M₀ across arms;
- complete three-pass coverage without replacement;
- prefix causality and finite optimization;
- terminal NLL inside the broad reproduction range.

The checker requires both learned initial-memory parametrizations to improve over zero and the
factorized result to be closer to full than to zero. It does not require an
ordering or fixed numerical gap between factorized and full.

## Recorded values

```bash
uv run --no-sync ./reproduce.sh verify-results
```

This verifies the three recorded result records and the preserved training
source inventory, deterministically derives `data/results.json`, and checks its
arithmetic and provenance. Three embedding parameter counts are derived from
the recorded configuration; metrics and model settings are read directly from
the records. This command does not train a model and is kept separate from
reproduction.
