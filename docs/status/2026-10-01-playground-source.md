# Editing canvas and Playground source work

This is source work after the `d312dcf` package handoff. The delivered App and
archive remain unchanged. Model inference, UI interaction checks and behavioral
tests are still stopped at the user's request; compilation does not establish
functional acceptance.

## Editing canvas

The resolution inspector offers reference-based output sizes separately from
input-file preparation. Matching uses the retained original metadata; automatic
512/768/1024 choices preserve approximate proportions and respect native width,
height, alignment and area limits. Original-size matching shows the actual
aligned dimensions before applying. Automatic fitting does not enlarge a small
source. Changing output dimensions does not rewrite input files.

Qwen LoRA and DiT-cache paths retain their existing 512-square constraint. A
conflicting candidate remains visible with an explanation and cannot be applied;
the operation never silently disables an adapter or cache. Manual incompatible
shortcuts are disabled while the 512 button remains available to repair an old
incompatible draft. FLUX transform uses only its selected initial image.

The metadata regression source covers original versus derivative selection,
active-reference selection, alignment, small-source fitting, model bounds and
LoRA/cache conflicts. It is included as `test-editing-canvas` but is not run in
this work session.

## Playground integration

Playground has a separate sidebar entry and separate persisted drafts for outfit
change and character consistency. Reference roles establish input order. It
shares the existing job runner, cancellation, output transactions and global
history; optional `workflowID` metadata associates jobs with a template without
changing the native request. Older job records remain compatible.

Each slot accepts a single replacement. Group import fills empty roles in
person-first order and stages the complete batch before committing; an invalid
later input removes earlier staged copies. Reference preparation always starts
from the retained original. An unreadable or structurally invalid saved document
blocks overwriting and generation, with explicit reload/save recovery controls.
Model settings are copied only through an explicit sync action. Incompatible
settings show a generation blocker rather than silently changing LoRA, cache or
acceleration. Automatic post-generation upscaling is not executed here and must
be disabled before synchronizing a usable configuration.

The actual constructed prompt can be inspected. Historical thumbnails switch
only the preview; using a result as a person or continuing in Creation requires
its explicit button. The cancellation handle lives with Playground state across
page changes, and cancellation is restricted to that workflow's admitted job.
LoRA names/strengths, steps, seed and cache mode are displayed with the copied
configuration. These templates guide the model; identity fidelity has not been
measured or guaranteed by this source work.

The toolbar shows the active workspace's save state. App termination saves both
workspaces. Creation's automatic upscaler preload is restricted to the Creation
page, so switching into Playground does not start an unrelated preload.
The quit prompt also covers pre-submission work and in-flight image imports.

## Static validation in this stage

- Whole-App Swift type checking passed with the new views and job metadata.
- The final Playground state and both new regression sources passed Swift type
  checking. `test-playground` covers template persistence/parameter isolation,
  fixed roles, failed-batch cleanup, damaged-file protection and result/seed
  ownership; its fixtures do not load a native catalog or model.
- Build-script syntax and `git diff --check` passed.

No test binary, App, API service, Core ML model or inference was launched for
these checks. The new UI has not been visually inspected. The preceding package
is not rebuilt in this source stage.

## Remaining goal acceptance

| Requirement | Current evidence and remaining work |
| --- | --- |
| Reversible input resizing and multiple-reference ordering | Source committed and packaged at `d312dcf`; behavioral/UI acceptance remains pending. |
| Original/automatic output canvas | Source and metadata regression target added in this stage; tests remain unexecuted. |
| Independent common workflows | Playground source integration in this stage; persistence, import, generation, cancellation and actual UI checks remain required. |
| AI-friendly local interface | Earlier capabilities, strict validation and Python client sources exist; local installation discovery and shared App/API history remain open. |
| Runtime ANE artifact lifecycle | Earlier private-lease changes compiled and packaged; regression execution remains pending. Core ML graph artifacts still use disk. |
| Runtime W8A8/Hadamard, encoder and scheduling improvements | Not implemented or established by the FP16 lease changes. |
| Laptop speed and image-quality benefit | No new measurements; earlier hardware results cannot establish this requirement. |

The full optimization goal is not complete. Resume behavioral acceptance only
after the testing stop is lifted; keep model experiments sparse and use existing
local weights. Do not infer performance or correctness from a successful build.
