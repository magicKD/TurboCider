# Studio image editor UX, 2026-09-30

## Problem and design

At the minimum 980 × 700 content size, the previous editor gave much of its
width to an always-open parameter inspector. Reference images and the prompt
shared a second scroll view, so the actual editing instruction could be offscreen.
The main preview also fell back to the latest historical result, hiding the
current input even after starting a new draft or choosing a different reference.

The revised workspace gives the image the flexible area and keeps the instruction
and submit action visible. References occupy their own compact strip. Selecting
a reference changes the large preview without changing its request order; an
explicit menu handles reorder, removal, and editing just that image. A numbered
strip makes multi-image instructions easier to write.

Generated results have an explicit **Continue editing** action. This imports a
separate input copy and starts a fresh instruction while preserving the result
in history. The canvas supports fit, zoom, and pan; the adjacent selection tool
offers circles, a brush, undo, and optional independent black-and-white guidance.
Annotation coordinates remain relative to the image after zooming and panning.

Model, LoRA, seed, compute, resolution, and advanced options move to a settings
popover. Resolution presets and engine implementation text no longer consume
the editing canvas. Model names use a clean presentation name while retaining
the raw catalog value and capability checks. Standalone upscaling remains a
separate accessible operation with its own settings.

## Product references

- [OpenAI image generation](https://learn.chatgpt.com/docs/image-generation):
  expanded single-image inspection, focused feedback, small targeted revisions,
  and clearly identified reference order.
- [Midjourney Editor](https://docs.midjourney.com/hc/en-us/articles/32764383466893-Editor):
  a direct transition from generated images to editing, contextual canvas tools,
  and nearby results.
- [Adobe Firefly image editing](https://helpx.adobe.com/mena_en/firefly/web/create-mood-boards/firefly-boards/edit-images.html):
  canvas-centered editing and tools tied to the selected image.

These informed the interaction hierarchy. Qwen's annotation/mask input remains
semantic guidance, not a pixel-preserving inpainting contract. The UI explains
this in the annotation tool instead of making an unsupported preservation claim.

## Verification

### Build and regression

- `make test`: passed.
- `make build-app`: passed (App and all Swift integration targets).
- `make test-app`: passed, including image publication/cancellation, history,
  model selection, preferences, clipboard, telemetry, and cache protection.
- `make test-library`: passed using only tiny loopback HTTP fixtures; no model
  weights downloaded.
- `turbocider-qwen21-workflow-tests`: passed, including image order, annotation,
  separate-mask rendering, orientation, transparency, and the base model's
  ten-reference contract.
- Added regression coverage for result-copy isolation, fresh editing prompts,
  single-image selection, Turbo's three-input limit, asynchronous provider
  conflicts and cancellation, preserving newly typed prompts, and workspace reset.
- Preview geometry checks covered fitted square/portrait images, zoom limits,
  finite-value guards, bounded panning, resizing, and normalized coordinates.
- Final read-only review found no additional blocking issue in the edited paths.
- `make package -o build`: passed using the freshly built artifacts; no runtime
  source changed in this iteration. App and CLI signatures, bundled libraries,
  shaders, branding and licenses passed inspection. Packaged App/CLI `doctor`
  both found the Apple M4 Pro Metal GPU. All 33 Mach-O sections of the packaged
  UI executable match the compiled executable (code-signing metadata differs).
- The last text-only deltas from the live GUI are the upscaler empty-state
  sentence pointing to the new settings control and retaining annotation/mask
  suffixes when a generated UUID filename is simplified. The App was recompiled
  after these changes.

The first sandbox-only clipboard run could not access the real pasteboard; the
same tests passed with local clipboard access. This is recorded separately from
product behavior. The Command Line Tools toolchain was used without changing
system settings.

### Actual GUI checks

A separately signed **TurboCider Verify** used an isolated state directory.
The user's existing library and drafts were not used for destructive tests.
The App was operated through the native UI, with screenshots inspected at a
980 × 752 point window and a larger window.

Verified interactions:

- The current reference appears on startup; historical output is no longer
  implicitly substituted for the editing input.
- One or three references, the large preview, instruction, and submit action
  remain visible at the minimum window width.
- Fit, 150% zoom, panning, and reset work; changing images resets the viewport.
- The annotation sheet supports drawing, undo, redrawing, and saving a separate
  annotated copy. Independent-mask brush, clear, circle, save and preview also
  passed. Clearing strokes disables submission; undoing the asset change restores
  the original reference.
- Selecting a thumbnail changes the preview without reordering request inputs.
  Explicit reorder updates numbering; undo restores the original order.
- At three Turbo inputs, add and paste actions are disabled consistently.
- Recent-result selection opens the result and closes its popover. Original/result
  comparison works. Continue editing uses an independent input copy and a fresh
  instruction, retaining the generated original in history.
- Cancelling sampling returns to a retryable state and publishes no output;
  the next correctly configured request succeeds.
- Command-N and toolbar New both clear the old result, references, prompt, and completion banner,
  retaining the configured model, six-step LoRA, GPU, and seed 42.
- Settings show the clean model name, selected adapter, GPU-only execution,
  and a collapsed resolution section; opening it confirms 512 × 512.
- Standalone upscaling opens its own model/device controls and returns to the
  composition workspace. Its empty-state wording now points to “超分设置”.

One automated attempt to remove the third reference did not activate the menu
item. The actual persisted request revealed three inputs, so it was cancelled
and excluded from two-reference acceptance. Removal was repeated with an
intermediate UI observation; the next request had exactly two inputs and passed.

### Real GPU acceptance

Existing local Qwen Image 2.1 BF16 weights and the existing Viggle v0.2.1 r128
six-step LoRA were used. All outputs are 512 × 512, seed 42, component-staged,
with neural inference and reference/text encoding on Metal GPU. No extra model
weights were downloaded. Each successful receipt must confirm six actual steps,
227 LoRA projections, GPU encoding, and a published PNG whose SHA-256 matches
the file on disk.

| GUI workload | Elapsed | MLX allocator peak | Visual inspection |
| --- | ---: | ---: | --- |
| Text to image, zero references, warm conditioning cache | 19.79 s | 17.83 GB | Blue teapot and orange on a wooden tabletop |
| Continue editing a generated result, one reference | 42.05 s | 20.66 GB | Red teapot changes to blue; fox and puffin remain present |
| Two references | 65.65 s | 24.59 GB | Red teapot and fox combined on a wooden tabletop |
| Three references | 91.23 s | 28.35 GB | Fox, red teapot, and puffin appear together |

These are functional GUI samples, not an isolated speed benchmark. The editing
runs overlapped Swift compilation; OS file caches were not cleared. MLX allocator
peak excludes OS, file cache, and Core ML. Subject preservation is a visual
assessment, not a guarantee that unedited pixels remain identical.

Raw requests, receipts, file hashes, cancellation evidence, screenshots shown in
the task, and GUI observations are retained under ignored `outputs/` paths.
The first text-to-image run succeeded but included a macOS Documents permission
wait inside text encoding (158.29 s total); its total is excluded from speed
assessment. The repeat after authorization took 19.79 s (16.97 s denoising), hit the text
conditioning cache, and produced the exact same PNG SHA-256. This is a warm
sample, not a cold-start estimate.
