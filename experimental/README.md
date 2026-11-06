# Experimental code

`experimental/` holds opt-in research code. Its targets are not part of the
default native library, CLI, App or release package and do not define the
product ABI.

## Available experiments

The maintained directory map contains one laboratory:

- [transformer/](transformer/README.md): synthetic Transformer CPU/GPU/ANE
  splitting, public/private API comparisons, real LTX replay diagnostics and
  VAE quality studies. Its README lists the build and reproduction commands.

The old `video/` and `private-ane/` snapshot directories no longer exist.
Production model implementations are under `native/models/`; do not restore
vendor snapshots into the shipping source list.

## Documentation and results

Keep runnable research sources here and archive experiment narratives under
`docs/experiments/`. The [Transformer archive](../docs/experiments/transformer/README.md)
holds the M4 Max reproduction ledger and links the historical M4 Pro report.
The M4 Pro report, protocol, provenance and small evidence are archived there
as well. Report generators remain with the runnable sources and write ignored
local output without overwriting the archive.

New raw runs, tensor dumps, compiled models and generated media belong in the
ignored root `outputs/` directory. Retain small, portable evidence with the
corresponding archived narrative; do not turn a kernel observation into a
whole-model performance claim.

## Shipping boundary

- `tools/native/build.sh` uses an explicit source list and excludes this tree.
- The default native build uses public APIs. A separate research configuration
  can enable `native/backends/private/` with `TURBOCIDER_ENABLE_PRIVATE_ANE=1`;
  private API code is therefore not confined to `experimental/`.
- `tools/native/package.sh` invokes `check_release_binary.py`, which rejects
  private-ANE release artifacts.
- The legacy H3 private-ANE entry points link `h3_ane_disabled.c` and fail
  closed. The source isolation rules are checked by
  `tests/repository/test_layout.py`.

Consult the [current acceleration entry](../docs/status/acceleration.md) for
runtime choices and the [historical heterogeneous report](../docs/experiments/transformer/2026-09-08-adoption-report.md)
for earlier adoption decisions. Historical reports describe their own code
revision and hardware.
