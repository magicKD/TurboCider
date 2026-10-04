# Qwen reference encoding and Playground acceptance — 2026-10-01

Follow-up: an actual exported identity-template request and the QKV lifecycle
regression also passed; see [final laptop acceptance](2026-10-01-runtime-ane-decision.md).
The App binaries and package described below remain unchanged.

This milestone separates App input-file preparation from the Qwen encoder's
reference resolution. Resizing an input file alone did not reduce the default
4,096 reference tokens: Qwen resized it again to its canonical 1024-area input.
The App now exposes **standard 1024** and an explicit **fast 512 preview** in
Creation and Playground. The default remains 1024. Both preserve aspect ratio;
the numeric choice describes approximate squared-pixel area, not a forced
square crop or the output canvas.

## Admission and persistence

The new no-environment-variable Turbo route is deliberately narrow: pure GPU,
512×512 output, one to three references, pinned Viggle v0.2.1 r128, six steps,
strength 1, inference-time LoRA, no prompt enhancement or DiT cache, and explicit
approximation. Model binding still verifies adapter SHA-256. Ordinary LoRA,
r256 and ANE keep their previous admission rules. Existing approximate base
editing remains available; the App shortcut limits it to 20–40 steps.

The setting survives draft save/reload, JSON export/import, history reuse and
Playground requests. Schema v1 uses `qwen21_reference_size`; schema v2 uses
`parameters.qwen21_reference_size`. Model discovery advertises field locations
and admission constraints. The workflow example supports `--reference-size`.
Incompatible saved combinations show a reason and an explicit standard-1024
recovery action. Selection does not silently change LoRA, steps, canvas or
acceleration settings.

## Paired local evidence

Existing local weights only; M4 Pro / 48 GiB; serial requests, no downloads.
Same 512×512 blue-teapot input, red-teapot edit prompt, seed 42, six steps,
Viggle r128 strength 1, component-staged GPU, no DiT cache or prompt enhancement.
The frozen `89c90dd` runtime ran one cold process at each encoding size; its
512 diagnostic gate selected the same numerical pipeline now exposed publicly.

| Encoding | Reference tokens | Request | Denoising | First step | Later steps | MLX peak |
|---|---:|---:|---:|---:|---:|---:|
| Standard 1024 | 4,096 | 37.28 s | 26.92 s | 13.29 s | 2.71–2.74 s | 19.99 GiB |
| Preview 512 | 1,024 | 25.16 s | 17.68 s | 5.31 s | 2.43–2.56 s | 17.06 GiB |

This pair took 32.5% less request time (1.48× ratio). It is not a repeated,
thermally isolated benchmark or a general speed guarantee. MLX allocation
excludes Core ML, OS and file caches. The first-step reduction is consistent
with substantially less reference-prefix work; it does not isolate compiler,
lazy materialization and prefix construction costs. Compilation alone is not
established as the cause of the slower first step.

Both outputs preserve the broad pot shape, composition and red recoloring;
texture differs. RGB correlation is 0.990831, cosine 0.998828, MAE 6.157/255,
RMSE 8.667/255. The existing strict MAE ≤5 gate **failed**. This is a quality
tradeoff for optional previews, not numerical equivalence. Fine facial details
and text should use standard 1024. A single teapot is not broad quality evidence.

Cross-request prefix snapshots currently require 1024 encoding and are bypassed
at 512; conditioning-cache keys still include resolution. These cold timings
do not establish repeated warm-request speedup. No snapshot implementation was
changed in this milestone.

Raw requests, receipts, phase events, images and metrics are under
`outputs/qwen-reference-quality-20261001/`. New native build identity, source
hashes and contract logs are under `outputs/qwen21-ref512-candidate-20261001/`.

## Public route and workflow checks

The rebuilt native runtime accepted and executed a three-reference edit through
the public 512 request field with **no diagnostic environment variable**. Its
receipt reports six actual steps, 227 LoRA-applied projections, 3,072 reference
tokens and 32.23 s request time. The decoded 512×512 result visually contains
all three requested subjects (blue teapot, fox and puffin) in one composition.
This is a functional smoke request; concurrent host compilation means its timing
is not a controlled performance comparison.

`tests/integration/PlaygroundRequestExport.swift` exports requests using the
actual Playground role import, template prompt and draft validation path. It
writes only a new/empty isolated state directory and does not load model weights.
The outfit export produced ordered person/clothing inputs, the selected 512
encoding and the correct schema-v2 parameters. Fixtures were generated serially
with the existing local six-step model (20.72 s and 20.57 s) and visually checked.
No real person's photograph or additional model was used.

The 12 native contract tests and focused RPC discovery/plan regression passed.
They cover v1/v2, one to three references, residency variants, missing explicit
approximation, invalid steps/strength/adapter, ordinary LoRA, legacy r256 gates,
DiT cache and ANE restrictions. A plan validates the request contract; weight
identity is still validated on binding.

## App acceptance and release

An ad-hoc signed isolated App used only `outputs/playground-e2e-20261001/`
state/library/settings. The original running user App was not replaced. The
new App and model-library helper compiled successfully. The focused Swift
reference-size regression passed, covering legacy request/draft/history
compatibility, Base/r128 eligibility, invalid combinations, explicit recovery,
persistence/import/reuse/model reset, canvas restrictions and independent
Playground submission. Final packaged CLI RPC regressions passed **3/3**.
The new focused Swift target was compiled separately; it is not yet a default
`build_app.sh` target. Exact commands and logs are in
`outputs/qwen21-reference-ui-20261001/acceptance.md`.

A real click on **Generate · Outfit** ran the actual App worker to success:
512×512, six steps, 2,048 reference tokens, 227 LoRA-applied projections,
35.04 s native request time / 35.13 s App elapsed time. Busy controls recovered,
the output and history thumbnail appeared, and the result displayed its saved
512 approximate-reference setting. Face/pose remained broadly recognizable and
clothing became yellow. However, the jacket's pockets and denim fabric were
not faithfully retained, and the plain background became textured. This passes
functional workflow execution, **not** fine garment/background fidelity. No
claim is made that 512 and 1024 are interchangeable for person workflows.

Live UI checks also verified the generation blocker after moving an edit-only
512 setting into text generation, the explicit restore-standard action enabling
the Generate button, settings-panel collapse/reopen, retained six-step/strength-1
LoRA parameters, and independent persistence of Creation=1024 and Playground=512.
The result is recorded in `outputs/playground-e2e-20261001/app-outfit-report.json`.

To keep laptop load and acceptance time bounded, this round stopped after the
paired single-reference edit, two fixture generations, one public three-reference
edit and one actual App outfit job. No new base-model long run, actual identity
template generation, native mouse multi-file drag dispatch or ANE experiment
was added. Earlier CPU provider tests cover order/cancellation, but do not replace
those live scenarios. Broad image-quality acceptance and ANE performance remain
separate work; this release makes no new claims about either.
