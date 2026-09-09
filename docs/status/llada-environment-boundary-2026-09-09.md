# LLaDA production environment boundary — 2026-09-09

Removed reference conditioning, router and QueryFormer tensor injection and
the environment-controlled eager-block override from native model sources.
Explicit request dumps and the standalone development Python oracle remain.
GPU-only inference retains its previous default compiled-block policy.

## Current-build validation

- Native library, CLI, Swift App and integration-test build succeeded.
- Repository, contract and LLaDA reference tests: 73 passed, 45 subtests passed.
- Built dynamic library has none of the four retired environment-variable names.
- Real 256x256, four-step, seed-42 fox-prompt generation succeeded from
  `/private/tmp`, with all four retired environment variables set (reference
  directories intentionally nonexistent). Backend remained `mlx_cpp_metal`,
  GPU graph `compiled_single_blocks`, prompt cache miss.
- Current request wall: 10.946654792 seconds; MLX peak: 45,973,308,210 bytes.
- This run required explicit sandbox escalation for Metal GPU access; the
  initial sandboxed attempt returned `Metal GPU unavailable`.

Historical worker-removal measurements (10.803101125 and 10.844790584 seconds)
are distinct runs, not a matched before/after performance experiment for this
change. A single current run cannot establish no performance regression across
models or replace the outstanding cold/warm benchmark matrix.
