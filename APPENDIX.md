# Experimental details

## Product-key-factorized SDM initial memory

SDM's product-key routing assigns each of N memory slots a pair of indices
(i, j), with √N choices on each side. The initial memory is factorized over
that same pair:

```text
M₀[i, j] = A[i] + B[j]
```

A and B are learned √N × dᵥ tables. Together they contain 2√N dᵥ scalars
instead of the Ndᵥ scalars in an independently learned initial table. The
functional dimension is (2√N − 1)dᵥ because a shared offset can move between A
and B without changing their sum.

The experiment uses N = 1,024 and dᵥ = 128 in each of seven SDM layers:

| Representation | M₀ parameters per layer | Seven-layer M₀ total | Total trainable parameters |
|---|---:|---:|---:|
| Zero | 0 | 0 | 14,712,348 |
| Factorized | 2 × 32 × 128 = 8,192 | 57,344 | 14,769,692 |
| Learned | 1,024 × 128 = 131,072 | 917,504 | 15,629,852 |

The factor tables are registered through `torch.nn.utils.parametrize`. The
released layer still receives a dense tensor with its ordinary 1,024 × 128
shape. No SDM source or forward path is changed.

Each factor is initialized with standard deviation dᵥ⁻¹ᐟ² / √2, so their sum
has the same target variance as the learned table initialized with standard
deviation dᵥ⁻¹ᐟ². Both learned arms use the same initial-memory role seed. All
other parameters use role-keyed seeds and have the same SHA-256 across arms:
`5c57ed222c1cf4dcd57e8893377a9a069f11df6e98a5d2e05be7fcc474ed295e`.
Initialization runs on CUDA before this identity is checked.

## Model

- layout `BBBBBBBA`: seven released-SDM blocks followed by one causal-attention
  block;
- width 128, four attention heads, one SDM memory head, and expansion-four MLPs;
- 1,024 logical slots per SDM layer with balanced sparse access R = W = 16;
- read-after-write recurrence, scalar SDM controller, normalized readings,
  channelwise output gate, and output projection;
- seed 0;
- BF16 activations with the released SDM FP32 internal boundaries preserved.

The only experimental mutation is the initial representation of `mixer.memory`.
Routing, recurrence, controller, forward method, and CUDA kernels are identical
across arms.

## WikiText-103 protocol

The experiment uses WikiText-103 raw v1 at dataset revision
`5fddba447aa4e75996922ea0d6b18b42f0a81cc4` and GPT-2 tokenization. Its
protocol ID is `wikitext103-gpt2-causal-t2048-coverage-v1`.

The 117,980,450-token training split is tiled into 57,608 causal records with a
2,048-token context limit. Every pass contains every record exactly once in a
seeded permutation, without replacement. With an effective batch of eight
records, each pass takes 7,201 optimizer steps and presents 117,980,449 scored
targets. Three passes therefore take 21,603 steps and present 353,941,347
targets per arm.

The three-pass budget was fixed before model results were inspected.
Validation checkpoints are reporting-only and occur without interrupting
training. The half-pass checkpoint is an experiment-defined early observation,
not a WikiText-103 boundary or standard milestone:

| Passes | Optimizer step |
|---:|---:|
| 0.5 | 3,601 |
| 1 | 7,201 |
| 2 | 14,402 |
| 3 | 21,603 |

Validation and test use the official splits, a 2,048-token context, and stride
512. Each transition is scored exactly once: 247,416 validation targets and
283,426 test targets. Test is evaluated for every arm only after the fixed
training budget is complete.

The prepared manifest has SHA-256
`fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6`;
its training and evaluation payload has SHA-256
`f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e`.

## Optimization

- AdamW with FP32 master parameters and FP32 moment state;
- peak learning rate 3 × 10⁻⁴;
- 540-step linear warmup, equal to 2.5% of the fixed schedule;
- cosine decay to 10% of the peak learning rate;
- weight decay 0.01 and betas 0.9, 0.95;
- gradient-norm limit 1.0;
- effective batch 8 and microbatch 1;
- uninterrupted training, with recovery snapshots that do not segment the run.

The recorded arms ran independently on NVIDIA RTX A6000 GPUs.

## State accounting

M₀ is shared model state. The active logical recurrent state remains one
1,024 × 128 value table in each of seven SDM layers for every sequence:
917,504 elements, or 1.75 MiB in BF16 and 3.50 MiB in FP32. Parameters,
per-sequence recurrent state, training activations, CUDA workspace, and peak
device allocation are separate quantities.

## Reproduction boundary

The reproduction regenerates the checksum-addressed data locally from the
pinned public dataset and starts from the vendored official SDM source at
revision `183e7df809131b80ad4393741029d0f20fc3640b`. It then applies the exact
recorded loader patch, SHA-256
`15751efa855485d3c2eabbaa8b0c686f94d11ee03d64366dc8d2cd6e7d36f574`, and
builds the two released CUDA extensions. The patch adds one loader file and
wraps the two upstream extension-loading calls; its verifier confirms that no
other source file or embedded C++/CUDA source changed.

The recorded run loaded prebuilt copies of those extensions. The clean
reproducer compiles them once into the local PyTorch extension cache and uses
the same patched SDM source for all three arms. Model and kernel mechanics are
unchanged.

The repository preserves the training implementation and one result record for
each arm. The reproducer uses the same three initialization paths, model,
optimizer, data order, schedule, and evaluation. The result extractor derives
only the embedding parameter counts needed by the compact result table.

Released SDM uses BF16 atomic accumulation, so reruns are checked against broad
metric ranges and configuration invariants rather than expected to be bitwise
identical. NLL differences are descriptive.
