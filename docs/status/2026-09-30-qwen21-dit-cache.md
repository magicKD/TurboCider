# Qwen Image 2.1 DiT-cache qualification

This report covers native BF16 GPU inference on an **Apple M4 Pro, 48 GiB,
macOS 26.6**. It uses the existing local Qwen-Image-2.1 checkpoint, 512×512
outputs, the normal 1024px reference processing, component-staged residency,
dynamic text and prompt enhancement disabled. No model was downloaded.
Raw requests, results, per-step traces and comparison images are retained in
`outputs/qwen21-dit-cache-20260930/` (ignored local evidence).

## Algorithm and correctness

The existing algorithm is DBCache-style aggregate residual reuse. Every step
runs the front blocks and compares the relative L1 change of their residual
against the immediately preceding step. When the change is small enough, it
adds the last completed middle-block residual instead of evaluating those
blocks. The front comparison is updated on skipped steps, as in the reference
algorithm. A consecutive-skip bound periodically forces fresh computation.
This is a lossy sampling optimization, not a compiled-kernel cache.

The established policy uses eight front blocks, no back blocks and eight full
warmup steps. The last sampling step is always full. The implementation does
not reuse the first full-prefix step's aggregate residual for target-only
decode. A compiled closure cache and the editing prefix K/V bank solve
different problems.

Changes in this revision:

- Clear residual state on step restart, repeated/discontinuous indices, traced
  execution, changed conditions and callback failure. Cancellation/retry cannot
  consume a partially completed cached step.
- Require stored residuals to be evaluated leaves without retained input graphs.
  Residual subtraction owns its storage; an MLX `copy` alone can alias its input.
- Make cache choice a saved, per-request field, with actual skipped-step and
  saved-block receipts. Explicit `off` overrides diagnostic environment settings,
  including malformed diagnostic overrides. Omitted fields retain the legacy
  diagnostic behavior.
- Separate ordinary runtime LoRA from the pinned Viggle six-step student.
  Ordinary adapters use the base Euler schedule. Qwen's strict loader rejects
  unmatched pairs, unsupported tensors, orphan alpha, invalid rank and unused
  output rows, and clears partial bindings on failure. Other models retain their
  existing non-strict behavior.

The supplied `Qwen-Image-2.1_NSFW_Image_Edit.safetensors` contains 448 BF16 A/B
tensors, 224 projection pairs, rank 16, and no alpha tensor. Tests use strength
1.0 with benign teapot prompts. This verifies loading and execution, not the
adapter author's intended style or a recommended training-specific strength.

## Reference systems

[SGLang's Cache-DiT guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/sglang-diffusion/cache_dit.mdx)
supports per-request enable/disable with an explicit false overriding server
defaults. It is used as design reference only. Qualification below tests
TurboCider's local presets; it does not benchmark the SGLang framework.
TaylorSeer and other calibration strategies remain disabled.

The merged [Qwen 2.1 prefix-K/V fix and experiment](https://github.com/sgl-project/sglang/pull/40472)
shows why each block must keep its own prefix K/V; TurboCider's native block loop already
indexes each layer separately and does not use SGLang's wrapped block container.
An optional local F1/W4 parameter comparison was stopped at the user's request;
its partial results are excluded from recommendations and qualification tables.

[Cache-DiT's design](https://cache-dit.readthedocs.io/en/latest/user_guide/DBCACHE_DESIGN/)
describes the front/back residual construction and accuracy/speed tradeoff.
Neither a residual threshold nor a small pixel error guarantees semantic
equivalence on arbitrary prompts, reference images or adapters.

## Measurement method

`tools/validation/qwen21_dit_cache_benchmark.py` runs a matched uncached control
and the candidates in one native CLI session. It excludes a full-step warmup
per prompt/step condition. Requests share prompt, seed, geometry, reference
bytes, LoRA, schedule and execution route. Model weights are staged as in the
App; warm conditioning is distinguished from model loading and first-prefix
work. The manifest records executable, native-library, reference and adapter
digests. Per-step traces verify warmup, final-step and consecutive-skip guards,
and must agree with actual saved-block receipts.

Primary comparisons disable cross-request editing prefix snapshots to isolate
DiT reuse. Additional repeated-edit checks use the App's default prefix policy;
the two gains must not be multiplied. Results are local samples, not confidence
intervals or universal quality guarantees.
Where a repeated uncached control is present, reported speedups use the mean of
the before/after controls; both timings remain in the JSON report. This records
order-dependent runtime variation rather than selecting the fastest control.

`tools/validation/qwen21_dit_cache_report.py` compares RGB pixels with the
matched uncached output. It reports MAE/RMSE on 0–255, PSNR, correlation and SSIM
(11×11 Gaussian, sigma 1.5, population covariance, valid interior, mean over RGB).
PNG dimensions are checked from the decoded pixels, not only the request receipt.
Alpha differences are reported separately; the teapot outputs are almost opaque,
but Qwen also predicts small alpha variations. Base cached outputs differ from
their matched alpha channels by at most 2/255. No perceptual model is fetched.
RGB contact sheets require visual inspection in addition to these metrics.

## Measured results

Times are seconds per request, including decode/export but excluding initial preparation.
Speedups use the matched uncached control (mean of before/after controls when available).
All outputs and input reference files are 512×512; normal reference encoding remains 1024px.

| Model | Task / steps | Off wall | Preset | Wall | Speedup | Denoise | SSIM | PSNR dB | Reused steps |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Base | edit-25 | 82.89 | conservative | 66.47 | 1.25× | 64.56 | 0.99708 | 47.54 | 8 |
| Base | edit-25 | 82.89 | balanced | 62.47 | 1.33× | 60.51 | 0.99331 | 42.20 | 11 |
| Base | edit-25 | 82.89 | fast | 62.00 | 1.34× | 59.39 | 0.99065 | 39.97 | 13 |
| Base | generate-25 | 55.47 | conservative | 42.82 | 1.30× | 40.97 | 0.98719 | 38.51 | 8 |
| Base | generate-25 | 55.47 | balanced | 38.27 | 1.45× | 36.41 | 0.96555 | 34.30 | 11 |
| Base | generate-25 | 55.47 | fast | 36.12 | 1.54× | 34.35 | 0.94228 | 31.40 | 13 |
| Base | edit-40 | 127.41 | conservative | 94.70 | 1.35× | 92.76 | 0.99802 | 49.95 | 15 |
| Base | edit-40 | 127.41 | balanced | 86.44 | 1.47× | 84.50 | 0.99493 | 43.82 | 21 |
| Base | edit-40 | 127.41 | fast | 76.11 | 1.67× | 74.16 | 0.99206 | 40.73 | 25 |
| Base | generate-40 | 96.82 | conservative | 70.99 | 1.36× | 69.29 | 0.99236 | 40.90 | 15 |
| Base | generate-40 | 96.82 | balanced | 62.72 | 1.54× | 60.89 | 0.97631 | 35.16 | 21 |
| Base | generate-40 | 96.82 | fast | 54.65 | 1.77× | 52.96 | 0.95588 | 31.90 | 25 |

All Base 25/40 uncached generation/editing PNGs are byte-identical to the
pre-change native baseline. The two Base 40 uncached repeats are also identical.
Visual review of the four contact sheets preserved subject, composition and edit
intent (blue teapot changed to red). Faster settings increasingly change glaze
texture, highlights and local edges; the metrics do not certify arbitrary prompts.

### Ordinary LoRA, 25 steps

| Task | Off wall (mean) | Preset | Wall | Speedup | Denoise | SSIM | PSNR dB | Reused steps |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| generate-25 | 64.03 | conservative | 46.27 | 1.38× | 44.46 | 0.99381 | 43.83 | 8 |
| generate-25 | 64.03 | balanced | 42.21 | 1.52× | 40.39 | 0.98496 | 39.11 | 11 |
| generate-25 | 64.03 | fast | 40.36 | 1.59× | 38.52 | 0.97131 | 35.29 | 13 |
| edit-25 | 89.34 | conservative | 70.70 | 1.26× | 68.49 | 0.99784 | 51.12 | 8 |
| edit-25 | 89.34 | balanced | 61.98 | 1.44× | 59.78 | 0.99520 | 45.79 | 11 |
| edit-25 | 89.34 | fast | 58.87 | 1.52× | 56.57 | 0.99243 | 43.54 | 13 |

All 12 ordinary-LoRA requests completed on the BF16 GPU route, with 224 bound
projections and exactly 25 sampling steps. Each uncached repeat is byte-identical
to its earlier uncached control. LoRA changes the image relative to Base; cache
quality above is compared only against that adapter's own uncached output.
Visual review preserved the LoRA teapot shape, lighting and the red-color edit.
Fast mode changes glaze/highlight texture more than conservative or balanced.
Maximum alpha differences are 3/255 for generation and 1/255 for editing.

Timing variation is visible: LoRA generation's uncached controls were 60.26 and
67.80 seconds; editing's were 89.00 and 89.68. Base 40 generation controls were
93.10 and 100.54 seconds. Speedups use their means and are estimates from a small
local sample, not a guarantee. No thermal-throttling cause is asserted.


### Repeated editing with the App's default prefix cache

A separate Base 25-step single-image edit used the default prefix snapshot:

| Request | Wall | First sampling step | Prefix hit |
|---|---:|---:|---|
| Initial warmup | 96.08 s | 13.84 s | no |
| Off, identical edit repeated | 71.48 s | 2.76 s | yes |
| Balanced | 61.46 s | 13.83 s | no |

Balanced is **1.16×** faster than the already-warmed prefix path here, not the
1.33× measured against full-prefill Off in the primary Base 25 matrix. Its RGB
SSIM against that warmed control is 0.99331. This confirms that the existing
first-step prefix optimization changes the practical comparison. DiT reuse does
not remove the new-reference first-step cost. Ordinary-LoRA repeated-prefix
performance was not remeasured in this revision; its main 25-step matrix is
complete. Further optional matrices were omitted to prioritize App delivery.

### Suggested presets

| Setting | Intended use | Threshold / consecutive reuse |
|---|---|---|
| Off (default) | Full-step sampling and the fidelity baseline | none |
| Conservative | Detail-sensitive final images; closest cached outputs in these samples | 0.15 / 1 |
| Balanced | First option to try for a useful speed/detail tradeoff | 0.25 / 2 |
| Fast | Quick previews when extra detail changes are acceptable | 0.25 / 4 |

These recommendations are based on 512×512 teapot generation and single-reference
editing, Base 25/40 steps and the supplied ordinary LoRA at 25 steps. They do not
qualify all prompts, other adapters, 2–3-reference editing or all intermediate
step counts. The supported interface range is wider than this measured matrix.
The fast preset saved only about two more seconds than balanced in the LoRA
25-step generation sample, while increasing the image difference. Keep Off as
the application default; request a new comparison for fidelity-critical work.

## App and request interface

The right settings column exposes the cache switch and preset, keeps invalid
combinations recoverable with a direct Off action, and explains any disabled
choice. Draft persistence, configuration import and history reuse preserve the
mode. Old records default to Off. Task details show actual cache reuse rather
than assuming that selecting a preset saved work. Generation followed by an
independent upscale remains supported.

Schema v1 uses top-level `qwen21_dit_cache`; schema v2 places it in `execution`.
Enabled modes require `allow_approximation: true`. Presets support pure GPU,
512², 20–40 steps, PE off, Base or one compatible ordinary transformer LoRA,
and up to three normal-size editing references. They reject conflicting
experimental kernels/caches. Viggle's six-step student keeps DiT cache off.

The editing prefix snapshot introduced in the
[startup investigation](2026-09-30-qwen21-edit-startup.md) remains separate and
is bypassed when DiT cache is active in this revision.

## Validation

- The official native build completed, including the changed internal request
  ABI. Every Swift compile command completed in grouped builds.
- The full `make test` suite passed. After the final strict-loader changes,
  `make test-qwen21` passed again: 44 Python Qwen checks (one optional skip),
  36 Qwen protocol cases, sampling/rewriting checks and Swift workflow tests.
  The complete 120-case protocol suite also passed (three optional skips).
- Real-MLX small-model regressions exercise repeated/discontinuous steps,
  changed conditions, trace and callback failure, evaluated residual storage,
  25/40-step warmup/final-step/consecutive guards and skip accounting. Extra
  F1/W4 diagnostics have small-model coverage only, not full-model qualification.
- Ordinary-LoRA fixtures cover incomplete/unmatched pairs, unsupported tensors,
  orphan alpha, zero dimensions, excess output rows and cleanup after failure.
  The supplied adapter header audit passes; full runs must bind all 224 pairs.
- Every App test target passed. The deferred NSItemProvider test required the
  normal macOS execution environment outside the shell sandbox; its same binary
  passed there without a product-code workaround. The initial sandbox failure
  and successful rerun are retained in the validation logs.

The native production Session probe was updated for ordinary-LoRA projection
counts, cancellation after cache warmup and same-policy retry/rebind comparison.
It is compiled in this revision. Full-model cached-LoRA cancellation/rebind was
not rerun in this qualification; the small-model lifecycle regressions above
provide the automated coverage. The earlier uncached Viggle prefix lifecycle
results are not presented as cached ordinary-LoRA validation.

## Delivered App

The final App was rebuilt after its preset guidance was added. Packaging and
strict App/CLI signature checks passed; all 35 executable Mach-O sections match
the final build and required bundle resources are present. The packaged engine's
`doctor` reports the Apple M4 Pro GPU, MLX 0.32.0 and no required Python runtime.
The delivered local bundle is `dist/TurboCider.app`.

Live checks used that packaged code in a separately identified verification App
and `/tmp` state, preserving the user's production draft. Confirmed the visible
four-choice cache selector, Off selection, settings collapse/reopen from the
model entry point, history selection without changing the draft, and explicit
history parameter reuse restoring the ordinary LoRA, ordered input, 25 steps,
strength 1 and Balanced cache. System file-access waiting initially delayed
preview reads; previews subsequently loaded normally.

A real App submission then completed a 512×512 ordinary-LoRA single-image edit:
25 actual steps, 224 bound projections, 11 reused steps and 264 skipped middle
blocks. Its PNG is byte-identical to the matched native CLI Balanced output.
The App displayed progress, the first-step duration, the result and a new history
thumbnail, then released image weights. Its 77.10-second end-to-end UI job includes
cold conditioning and is not the warmed-encoding timing used in the comparison
tables. The isolated verification App was closed after the user's request to
stop testing. No further inference or test matrices were started.

Quit and reopen an already-running production TurboCider process to use the new
bundle. The change remains on `dev-verify`; no additional merge or push was made.
