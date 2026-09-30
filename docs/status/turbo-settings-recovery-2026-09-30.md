# Turbo settings recovery, 2026-09-30

## Reproduction and cause

The user's saved Qwen Image 2.1 draft had a valid enabled Viggle v0.2.1 r128
adapter at strength 1, GPU execution and a 512 × 512 canvas, but retained the
base model's 40 sampling steps. Generation was correctly rejected. However,
the composer always displayed “Turbo · 6 步” merely because it recognized the
adapter filename, and its generic error also blamed an already-correct canvas.
The recovery action was buried in settings.

The real App was inspected through its UI. Applying the existing Turbo preset
changed 40 to 6, immediately enabled submission, and preserved the user's current
prompt and reference. No user generation was submitted during that repair.

Two reproducible settings transitions could recreate the conflict: attaching or
re-enabling the adapter did not configure its sampling settings, and returning
to Qwen through the model picker restored its adapter alongside base defaults.
Several editable fields also offered values that the maintained Turbo executor
would reject.

## Changes

- Adding or enabling a recognized single Turbo adapter, or returning to its
  model, synchronizes the existing six-step preset. Prompt and references remain.
- Disabling/removing Turbo restores 40 steps only when leaving its six-step
  schedule; other models and manually chosen non-six-step schedules are preserved.
- Existing saved drafts are not silently migrated. Their badge reflects an
  invalid Turbo configuration and the exact conflicting values appear next to
  a direct **Apply Turbo settings** action.
- The Generate button and request use the same draft validation, including seed,
  supported LoRA strategy, missing files and model capability checks.
- Turbo's steps, strength, role and currently supported canvas controls cannot
  be edited into unsupported combinations. Base-model controls remain available.
- The active adapter is described as runtime loading without merging weights.
  Resolution wording identifies the App's maintained fast-mode scope explicitly.
- Duplicate adapter paths are rejected; asset undo replaces stale mask-success
  messages with the actual undo result.
- Prompt-enhancer directory and experimental editing checks now participate in
  draft validation, so known PE configuration errors appear before submission.
- Shared frame/dimension bounds are checked before LTX token arithmetic. Extreme
  integer inputs now produce validation errors instead of overflowing on the UI
  thread; normal supported LTX settings remain valid.

## Runtime audit

The current [Viggle model card](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo#rules-that-matter)
recommends scale 1, unmerged runtime LoRA, six specific raw sigma nodes, no CFG,
and disabling terminal sigma stretching. The native implementation follows
those core choices: `[1, 0.9375, 0.875, 0.75, 0.5, 0.25]` before the
resolution-dependent shift, guidance scale 1, and a final zero sigma. Alpha
equals rank for both supported r128 and r256 adapters;
its effective strength scaling is therefore 1. Runtime loading validates the
adapter identity and binds 227 projections.

The App's 512 × 512 GPU-only fast path is a maintained local acceptance boundary,
not Viggle's maximum resolution. The official examples use larger canvases and
recommend prompt enhancement. This user's quick-validation setup keeps prompt
enhancement off. Thus the acceleration settings agree, while the complete
quality/size configuration differs from the official demo. No weights were
added or downloaded.

Relevant implementation: `native/models/qwen21/scheduler.hpp`,
`native/models/qwen21/pipeline.cpp`, `native/models/qwen21_module.cpp`, and
`native/backends/mlx.cpp`. No numerical runtime changes were made in this fix.

## Verification

An isolated native App reproduced the user's 40-step conflict with a benign
teapot prompt and one retained reference. Its new inline repair enabled text
image generation, preserved the draft, and displayed the actual six-step state.
GUI checks also passed for disable/re-enable, Qwen → FLUX → Qwen, locked Turbo
controls, and invalid seed −1 → valid seed 42 with immediate button/error updates.

A real submission from the repaired GUI completed: 512 × 512 PNG, six actual
steps, 227 LoRA projections, Metal GPU text encoding and sampling, strength 1,
and runtime LoRA. The request had zero inputs despite the retained reference
thumbnail, confirming text-generation semantics. The output decoded correctly,
matched its receipt SHA-256, and visually contained the requested blue teapot.
Its 61.45-second cold-session sample overlapped compilation and is not an
isolated performance benchmark.

New behavior tests cover stale saved drafts, precise reasons, add/enable/disable/
remove transitions, duplicate paths, multiple active adapters, model round trips,
retained references excluded from text requests, GPU hints, and unrelated-model
settings. Additional regressions cover extreme LTX frame/dimension values and
PE directory/experimental opt-in validation.

- Repository `make test`: passed after the real GPU job completed.
- Final App plus all 40 helper/integration targets: compiled successfully.
  The initial sequential `make build-app` was interrupted by the final source
  edits; all 41 original build-script compiler invocations were then rerun from
  frozen source in four independent batches.
- Final `make test-app`: passed, including the new Turbo, PE and LTX bounds
  regressions, image transactions, worker lifecycle, model settings, history,
  clipboard, library, telemetry and cache checks.
- Final `turbocider-qwen21-workflow-tests`: passed.
- Final read-only review: the overflow case is rejected before arithmetic;
  PE request semantics and valid Qwen/LTX bounds are unchanged.
- `make package -o build`: passed with the final App and unchanged native engine.
  App/CLI signatures and required libraries, shaders, branding and license files
  passed inspection. All 35 Mach-O sections of the packaged UI executable match
  the final build. Both packaged `doctor` commands found the Apple M4 Pro GPU.

The live GUI/GPU run preceded the final PE/bounds guard changes and the generic
LoRA strategy label cleanup; those final differences were rebuilt and covered by
the final behavior suite. The main user App's current settings were repaired
in place; restarting `dist/TurboCider.app` loads the permanent code changes.

The first repository test attempt overlapped the real App GPU run and encountered
the intended cross-process GPU exclusion. The repository suite passed when rerun
after the App generation finished; this contention is separate from the UI defect.
Raw local evidence lives under ignored `outputs/turbo-settings-20260930/`.
