# Third-party sources

## Sparse Delta Memory

The experiment uses the official Sparse Delta Memory layer and CUDA training
path.

- Paper: <https://arxiv.org/abs/2607.07386>
- Official repository: <https://github.com/facebookresearch/sparse-delta-memory>
- Revision: `183e7df809131b80ad4393741029d0f20fc3640b`
- Vendored tree: `third_party/released_sdm/lingua/sparse_delta_memory/`
- Upstream license: [CC BY-NC 4.0](third_party/released_sdm/LICENSE)

The files enumerated by `third_party/released_sdm/SOURCE.json` are byte-for-byte
copies of that revision. The loader-only patch used by the recorded run is
included under `third_party/patches/`; it leaves the embedded C++ and CUDA
sources unchanged and is applied only to a generated runtime tree. No model
weights or generated artifacts are redistributed.

The repository's MIT license applies only to the original experiment,
reproduction, and factorization code. The vendored Sparse Delta Memory code and
the runtime derived from it remain under CC BY-NC 4.0. Distributions built from
this repository include both license notices.

## WikiText-103

The preparation script downloads the public WikiText-103 raw-v1 Parquet
conversion from `Salesforce/wikitext` at revision
`5fddba447aa4e75996922ea0d6b18b42f0a81cc4`, then verifies every shard before
local preprocessing. Dataset files are not redistributed. The dataset card
lists CC BY-SA and GFDL licensing; users are responsible for those terms when
downloading the data.
