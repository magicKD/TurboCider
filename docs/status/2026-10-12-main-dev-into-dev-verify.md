# Integrating main and dev into dev-verify

Validated on 2026-10-12, using existing local dependencies and no model downloads.

## Branches and conflict resolution

After fetching origin, local and tracked tips agreed:

- Target before integration: `dev-verify` at `e484870`.
- `main` at `5dbb30a`: five commits ahead of the target; fast-forwarded first.
- `dev` at `a151dc2`: 32 commits absent from the original target, while the
  original target had 48 commits absent from dev. Merged after main.

Five files required content conflict resolution: `Makefile`,
`native/backends/mlx.cpp`, `native/backends/mlx.hpp`,
`native/models/qwen21/pipeline.cpp`, and
`tests/native/qwen21_diagnostic_options_test.cpp`.

Both sets of test targets, LoRA ownership/metadata/snapshots, and declared affine
quantization geometry are retained. Qwen's source-qualified GGUF fast defaults
coexist with native encoder/DiT admission checks and explicit overrides; the
new defaults do not remove the original BF16/native routing constraints.

Semantic review found a conflict beyond the textual markers: dev-verify's
`bind_arrays()` calls `clear()` to release old LoRA/source ownership, while dev
adds an affine geometry reset to `clear()`. Combining them naively would erase
an explicitly selected Q4/g64 or Q8/g128 layout. Rebinding now preserves the
caller-declared geometry while releasing old ownership. An explicit `clear()`
still resets it. Supplied-base LoRA slices and runtime LoRA input validation now
use the same declared logical input width rather than a fixed group size of 32.
The tiny affine fixture covers rebinding, supplied-base slices and reset behavior.

Main's ComfyUI integration, App HTTP gateway and Mach memory compatibility fix
remain present. Dev contributes bounded verified GGUF import/read-ahead,
authenticated source registration reuse, Qwen fast encoder defaults, native
FLUX saved affine models and the consolidated benchmark evidence.

## Verification

- Fresh package-mode build: native library, CLI, SwiftUI App and model-library
  helper succeeded with the installed Command Line Tools and macOS 15.2 SDK.
  All 609 source hashes in its runtime build manifest match the merged tree.
  Existing Metal `fastMathEnabled` deprecation warnings remain.
- Benchmark evidence/contracts: 194 tests passed (`make test-base-benchmark`).
- Source registration, source leases, read-ahead, private GGUF snapshots,
  packed import and Qwen options: 26 passed, one opt-in case skipped.
- Repository layout, independence and C++ boundaries: 17 passed.
- ComfyUI: 14 passed, including actual CPU image tensors and loopback HTTP.
- Fresh Swift HTTP gateway fixture: passed health/catalog, uploads, LoRA,
  planning, jobs, cancellation, artifacts, framing, port conflicts and restart.
- Against the newly built library: 142 passed, three conditional cases skipped.
  This includes six Q4/Q8 group-32/64/128 GPU tensor cases, LoRA compiled/supplied
  projection equivalence in FP32/FP16/BF16, native route and public contracts.
- README links/code fences and Git whitespace checks passed.

Total Python checks: 393 passed and four skipped; Swift gateway acceptance is
additional. Logs and binaries are local under
`outputs/main-dev-verify-20261012/` and `build/main-dev-verify-20261012/`.
Initial attempts exposed an absent Pillow import path and sandbox Metal access;
reruns used the already installed Pillow and permitted local device access.
No dependencies or models were downloaded.

This is a merge/build/component acceptance, not a fresh full-model performance
or image-quality campaign. The repository-wide `make test` and private-ANE build
were not rerun. No long inference, model conversion or App installation was done.

## README evidence

English and Chinese introductions now emphasize complete-request comparisons:
Qwen GGUF 3.40× and Z-Image 3.15× against their recorded Unsloth/Metal backends,
plus Z-Image 1.34× against mflux and 2.55× GPU-only against Unsloth. These are
existing M4 Max 64 GiB cohorts, not new measurements from this merge. Their
backend, warm timing and optional approximate ANE boundaries remain explicit;
detailed measurements stay in `docs/benchmark/PERFORMANCE_REPORT.md`.

The feature overview includes local HTTP/Unix APIs, ComfyUI/Comfy Desktop,
LoRA, model management, image editing and video generation. It also corrects
the old model table's missing Qwen row and outdated LTX image/audio limits.
