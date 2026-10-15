# Integrating dev-verify into main

Validated on 2026-10-12 after fetching both upstream branch references.

## Merge result

- Main before integration: `9a9938eafecaf6851693c1bb24c1bfe20fa021b3`.
- Dev-verify: `e48487058438ebb4033b98cb12028fc239b7dfcf`.
- Common ancestor: `dfb0ad5dfb056cdf19c4004fc3793e121bc56dfc`.
- The branches had two main-only commits and 139 dev-verify-only commits.
  A normal two-parent merge completed automatically with zero file conflicts.
- The merged production tree is exactly dev-verify plus main's
  `b3c4134` Mach memory-observation compatibility fix. That fix accepts an older
  kernel's shorter VM statistics structure when all consumed fields are present,
  retaining failure checks for unavailable or insufficient observations.
- Both histories are preserved. Model hub, component installation, LoRA
  management, bundled publisher logos and the other dev-verify changes remain
  intact. This record is the only additional integration file.

## Validation

- `make test-runtime-ane-host`: 17 host tests and 24 calibration/descriptor/
  policy tests passed, followed by the worker protocol/admission cases. The
  memory test compiled the merged `ane_memory.cpp` and successfully queried
  actual Mach memory statistics on this Mac. It also checked admission limits,
  overflow handling and retained-versus-new scratch allocation accounting.
- Repository suite: all 17 boundary, independence and layout tests passed.
- No unresolved index entries; staged whitespace validation passed.
- App, model-library helper and packaging source hashes match the preceding
  accepted dev-verify version. Its 16 App suites and eight model-library stages
  were already accepted in the same local session; this merge does not change
  those sources. See [the preceding acceptance](2026-10-12-model-brand-logos.md).

These merge checks use CPU fixtures and direct system observation, without
loading model weights, running a GPU stress experiment or exporting Core ML.
They do not constitute a fresh full native/App package build or a real-model
quality/performance campaign.

Local logs and the integration receipt are in `outputs/main-merge-20261012/`.
