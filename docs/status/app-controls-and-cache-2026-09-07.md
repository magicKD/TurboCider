# App controls and cache reuse — 2026-09-07

Branch: `dev-verify`. These changes extend the Z-Image App integration on the M4 Pro 48 GB host.

## Behavior

- New drafts and model selections use GPU only. Legacy automatic defaults migrate to GPU; explicit saved choices remain. Creation parameters and model management expose a checked GPU and an optional ANE checkbox.
- Each LoRA has an enable checkbox and editable strength, default 1.0. Disabled entries retain their path and strength, survive restart and are omitted from validation, generation and Core ML export. Older LoRA records without an `enabled` field remain enabled.
- Generation, load and warmup inspect matching compiled partitions before using ANE. Selection checks the checkpoint identity (including shared hard-link installations), shape, active LoRA paths/roles/strengths, complete artifacts, and available managed-cache OS/GPU identity. Native loading additionally verifies checkpoint and adapter SHA-256.
- A valid selected or remembered compiled partition is reused without invoking the compiler. Otherwise the App resolves the matching source manifest and uses the existing content-addressed compiler, which compiles only missing source/OS/GPU/architecture cache entries. Missing or incompatible LoRA/strength artifacts produce an actionable error; users can disable ANE to use GPU.
- Deleting a result moves an App-owned output to the macOS Trash and hides it from previews and the library, retaining task parameters. Deleting a task removes its record while keeping its output file. The last deleted task can be restored with Undo. Running tasks and external/symlink outputs are protected.

## Verification

`turbocider-studio-tests` passes the new default, legacy decoding, LoRA enable/strength persistence, disabled missing-file behavior, ANE toggles, external compiled-cache discovery, image trash/persistence, task delete/undo and external output rejection checks, along with the existing App behavior suite.

`turbocider-studio-controls-tests BASE_CONFIG LORA_CONFIG OUTPUT` exercises the actual asynchronous App job store with local Z-Image weights. Results are in `outputs/z-image-controls-20260907/`:

| Configuration | Result |
| --- | --- |
| LoRA disabled + ANE | Base cache selected; PNG identical to prior base ANE reference |
| LoRA enabled, strength 0.5 + GPU | All 180 projections applied; PNG differs from strength 1.0 GPU reference |
| LoRA enabled, strength 1.0 + ANE | LoRA cache selected; identity verified; PNG identical to prior LoRA ANE reference |
| LoRA strength 0.5 + strength 1.0 ANE cache | Rejected before generation |

The native compile request was also run against the existing LoRA source and cache: **32/32 cache hits**, with all 32 checked compiled files retaining their modification timestamps. Evidence: `compile-reuse.json`, `compile-reuse.stderr.log`, `report.json`, `jobs.json`, and `parity.json` in the same output directory. Single integration timings include load and cache setup and are not a new warm-performance benchmark.

`app-controls-config.json` is an optional App-importable configuration containing both validated base and LoRA manifests, GPU selected, and LoRA enabled at strength 1.0. It uses local absolute paths. No models were downloaded or rewritten.

Build output: `dist/TurboCider.app`, locally ad-hoc signed. The earlier unrelated LTX source-sync failures in repository-wide `make test` remain documented in the [Z-Image App support record](z-image-app-support-2026-09-07.md).

The packaged window was checked through native UI automation: disabling LoRA disables its strength input without removing the row; re-enabling it permits entering 0.5 and restoring 1.0; disabling ANE shows GPU only. Library trash buttons and task-detail deletion actions are present. Actual deletion/undo behavior was tested on generated fixtures, leaving the user's existing gallery untouched. Both validated external ANE manifests were registered through the model-center picker, and the current draft was left with GPU only and LoRA enabled at 1.0.
