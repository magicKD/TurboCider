# Final desktop acceptance — 2026-10-02

This continues the `66647ab` history/API fix on the M4 Pro / 48 GiB laptop.
Testing uses the actual `dist/TurboCider.app`, the existing local Qwen2.1
checkpoint and Viggle v0.2.1 r128 adapter. No models or dependencies were
downloaded. The earlier locked-desktop limitation in the runtime and
history/API reports is superseded by this desktop acceptance.

## Drag interaction limit and final scope

The reference transfer type is now exported in the packaged Info.plist and
explicitly conforms to `public.data`. Each card has a separate drag handle,
an accessible label and a highlighted drop target, avoiding the preview button
as the only apparent dragging surface. The existing workspace/asset checks and
reorder operation remain unchanged. Launch Services confirms the type is declared.

Automated mouse drags did not reorder cards. The same drag operation also failed
to move two selected test PNGs into a folder in the foreground Finder window.
This prevents attributing the failure to TurboCider or claiming successful mouse
drag acceptance. Multi-file picker/paste import and menu reordering passed;
actual mouse multi-file drop and handle reordering remain **unverified**.
A diagnostic NSItemProvider replacement was built but withdrawn because the
control experiment did not justify changing the transfer mechanism. The release
uses the previously built and exercised minimal UI candidate, with all 40 App
source hashes matched to that build. No diagnostic logging is shipped.

The user requested immediate delivery and an end to this round before going
offline, so no further model runs or drag experiments were started. This is a
local delivery with the above acceptance limitation, not a claim of zero bugs.

## Actual App checks

- The settings inspector collapses and reopens. Text generation is enabled
  with a nonempty prompt and valid settings; references can remain in the
  draft without silently changing text generation to editing.
- Enabling the existing six-step Viggle adapter selects GPU, 512×512, six
  steps, strength 1 and inference-time LoRA. ANE and DiT cache are disabled.
  Disabling it restores base 40-step sampling; 25 steps and balanced DiT cache
  are selectable. Reenabling Turbo restores its compatible settings.
- Two Finder file URLs import together using the explicit Paste Images menu.
  The reference size sheet applies fit-512 to both: a 1024-square teapot becomes
  512-square and the 512-square fox stays unchanged. Original metadata survives.
  The output-canvas sheet reports the original 1024 dimensions and rejects an
  incompatible Turbo size with an actionable explanation; auto-512 is accepted.
- A real two-reference edit completes and publishes its PNG and history item.
  Original comparison, history reuse and reference menu ordering work. Reuse
  preserves submitted order, prompt, seed, LoRA and original-image metadata.
  Restore Original returns the exact 1024-square source bytes.
- Playground accepts two files from one Open panel, fills the first two roles,
  produces the corresponding image-numbered prompt and enables generation.
  Syncing Creation settings preserves the GPU/Turbo configuration. Input resize
  works; switching templates preserves separate drafts and disables generation
  when the chosen workflow lacks required inputs. Restart preserves drafts.
- Upscale This Image uses the completed edit, runs the existing x2 Core ML
  upscaler and publishes a 1024-square result without replacing its source.
  The result appears in the same thumbnail history; model release works.
- The actual App API starts, stops, restarts and stops again. Socket requests
  for capabilities, models, installations and service status return success.
  Status without a job ID correctly reports the missing required field.

Plain paste inside the focused prompt editor inserts text; the explicit Paste
Images action imports clipboard image/file references. These are separate
actions, both checked. Playground's real generation arithmetic is shared with
Creation; its UI submission readiness was checked without adding another model
run to the laptop workload.

## Real output evidence

The safe edit changes the teapot in image 1 to white ceramic and paints the fox
from image 2 onto it. Visual review confirms the requested teapot shape, fox
illustration and simple background. It uses seed 529022108, two 512-square
inputs, optional fast-512 reference encoding, six steps and adapter strength 1.
The receipt confirms 227 LoRA projections, GPU encoding and six denoising steps.

| Operation | Result | Relevant timing |
| --- | --- | --- |
| Two-reference GPU edit | 512×512 PNG; job `53CFCF1D-3841-42F4-8BB4-CEAD1BDE2B02` | Denoising 20.516 s; first step 7.567 s |
| Core ML x2 upscale | 1024×1024 PNG; job `9D9BC465-BBB7-4307-AA4F-C812C091F46F` | Reported job elapsed 0.616 s, excluding model preload |

The edit's wall time includes a long file-access permission wait and **must not
be used as a performance benchmark**. This acceptance adds one Qwen generation
and one upscale, not a repeated throughput campaign. Fast-512 reference encoding
remains an approximate preview option; this sample does not certify all image
quality or replace the standard-1024 recommendation for fine detail.

## Regressions, data and delivery

The prior three Swift suites and nine Python service/RPC/CLI test methods remain
applicable to their unchanged production state/service sources. The final UI
changes are rebuilt and checked through the real App. Native source identities,
final App source hashes, test-log hashes, executable sections, strict local
signature verification and archive byte equality are recorded in the final
validation and delivery JSON files.

Evidence is under `outputs/final-ui-acceptance-20261002/`, including the two
completed job records, API responses, exact original-restoration check and
source snapshot. All 105 preexisting history records remain equal to the backup;
the two successful test results are retained. The original Creation draft is
restored after testing, and the newly created test Playground draft is backed up
outside the App state directory.

Delivery uses `dist/release-<commit>/TurboCider.app` and
`dist/TurboCider-macOS-arm64-<commit>.zip`, with a matching checksum and delivery
manifest. The actual `dist/TurboCider.app` is updated to those same files and
reopened. The preceding installed version is retained for rollback.

The runtime decision is unchanged: GPU remains the App default. The staged
Runtime ANE route did not beat GPU in the matched laptop experiment, so this
release does not advertise it as faster. See the separate
[runtime lifecycle report](2026-10-02-runtime-staged-lifecycle.md) for the measured
comparison and the fixed 576 MiB compiled-graph retention.
