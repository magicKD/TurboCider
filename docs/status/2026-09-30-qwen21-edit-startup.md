# Qwen Image 2.1 editing: first-step cost and bounded prefix reuse

This investigation concerns **image editing**, not text-to-image. All measurements
use the existing local Qwen Image 2.1 BF16 checkpoint on an Apple M4 Pro with
48 GiB unified memory, macOS 26.6, MLX 0.32.0, pure GPU execution, component-staged
residency, 512×512 output, dynamic text and prompt enhancement disabled. The
Viggle v0.2.1 r128 adapter uses six steps and strength 1.0; the base uses 25 steps.
No additional model was downloaded. Local raw evidence is under the ignored
`outputs/qwen21-startup-20260930/` directory.

## Why editing starts slowly

The first denoising step computes the text/reference prefix through all 32
transformer blocks and captures K/V. Later steps reuse that bank and compute
target-image rows. A 512×512 output has 1,024 target tokens, but the normal
reference encoder still uses a 1,024-pixel reference size: 4,096 tokens per
reference image. Output size does not reduce reference size automatically.

The original one-reference measurements were:

| Workload | First step | Median of remaining steps | Denoising | Request wall |
|---|---:|---:|---:|---:|
| LoRA six steps, first edit | 14.20 s | 2.96 s | 28.98 s | 39.62 s |
| LoRA, identical edit again | 14.37 s | 2.89 s | 28.92 s | 31.82 s |
| Base 25 steps, conditioning already warm | 12.79 s | 2.67 s | 76.78 s | 79.16 s |

Repeating identical conditioning saves the text/vision/VAE encoding, but the
original request-local transformer recomputes the prefix. A warmed second
request still has a 14-second first step. This is evidence that the dominant
cost is real reference-prefix computation, not just one-time compilation.

The App's speed label was also easy to misread. It averages up to five recently
completed sampling intervals, starting after two intervals. With step times
`14, 3, 3, 3, 3, 3`, it shows `8.5, 6.7, 5.8, 5.2, 3.0` seconds/step. Only the
first step was slow; the displayed average keeps falling through step six.
The UI now calls this “近期均速”, separately records “首步采样”, and preserves that
duration in history. Missing step-zero telemetry leaves the first-step value
absent; old history JSON remains readable.

## What is and is not a compilation cache

MLX caches compiled functions, while input shape/type changes and repeatedly
destroying their function owners can retrigger compilation. Its Metal backend
also maintains library/kernel caches. Releasing the allocator cache is not
equivalent to deleting every compiled kernel. See the pinned
[MLX compilation documentation](https://github.com/ml-explore/mlx/blob/v0.32.0/docs/src/usage/compile.rst)
and [Metal device implementation](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp).

Qwen uses the App's embedded native session. It does not launch a disposable
Qwen worker for every image. However, a staged request deliberately destroys
its transformer and compiled closures so their captured weights can be released
before VAE decoding. Keeping those objects indefinitely would undo that memory
policy. The prefill and target-only decode shapes also differ.

The added step profiler separates CPU graph construction from evaluation. In a
one-reference LoRA request these were 0.013 s and 13.530 s for the first step.
**This is not a measurement of all compiler time:** Metal driver compilation can
occur inside evaluation. Full-request warmup moves work earlier; it does not
eliminate the need to compute a new reference prefix.

## Change: retain completed K/V values across compatible edits

The ordinary GPU editing route now retains one compact, evaluated prefix bank,
without transformer closures or model weights. It defaults on for 512×512 edits,
normal 1,024-pixel reference processing, at least two steps, and PE off. It
supports the base and runtime LoRA paths. Other experimental routes, dumps and
quantized paths bypass this cache. `TURBOCIDER_QWEN21_PREFIX_SNAPSHOT=0` restores
the original request-local behavior; `1` explicitly enables the bounded route.

Same reference contents/order and instruction can hit even when the seed changes.
Changed conditioning, adapter identity, geometry, steps or route invalidate it.
Cancellation, failure and unloading clear it. A candidate bank is published
only after successful completion. Reference identity includes content hashes;
checkpoint files retain the existing immutable-installation assumption (do not
hot-replace weights while preserving file size and modification time).

The cap is the smaller of 8 GiB and one eighth of the configured memory ceiling
(physical memory, further limited by an explicit request budget). Export also
requires room for the copy and an 8 GiB margin relative to MLX active allocation.
This is a conservative allocator check, not an OS-wide memory-pressure guarantee.
On this 48 GiB Mac one reference retains about 2.02 GiB; two about 4 GiB. Three
references plus text exceed the 6 GiB cap and use the original path.

Measured same-process candidate results, with first-request and repeat encoding
costs kept separate:

| Workload | Cache | First step | Denoising | Request wall |
|---|---|---:|---:|---:|
| LoRA, seed 42 | miss | 13.54 s | 27.31 s | 37.71 s |
| LoRA, seed 42 again | hit | 2.79 s | 16.52 s | 19.42 s |
| LoRA, seed 43 | hit | 2.74 s | 16.57 s | 18.80 s |
| Base 25, seed 42 | miss, warm conditioning | 12.40 s | 74.63 s | 76.55 s |
| Base 25, seed 42 again | hit | 2.52 s | 63.96 s | 66.00 s |
| Base 25, seed 43 | hit | 2.55 s | 65.12 s | 66.87 s |
| LoRA, two references | miss | 25.53 s | 41.66 s | 58.24 s |
| LoRA, same two references again | hit | 3.20 s | 19.04 s | 22.81 s |
| LoRA, three references | over cap, original path | 43.77 s | 63.75 s | 85.92 s |

This saves about 10–11 seconds of reference-prefix work in this single-reference
workload. It does not promise this gain on the first edit or after changing its
instruction/images. Measurements are local samples, not confidence intervals
or guarantees across devices and workloads.

The two-reference miss and hit leave 4.024 GiB MLX active; the three-reference
fallback clears the old bank and ends with 0.00185 GiB active. This distinguishes
intentional bounded cache retention from accidentally keeping transformer weights.

The mathematical prefix is independent of target noise, but target-only first
steps change GEMM row counts/alignment and can change floating-point rounding.
Base 25-step PNGs for seeds 42 and 43 were byte-identical to their full-prefill
oracles. LoRA output mean absolute pixel differences were 0.269 and 0.146 on the
0–255 scale (PSNR 51.38 and 55.52 dB; maximum channel differences 29 and 15).
The user explicitly accepted these small numerical changes. This is not a claim
of bitwise LoRA parity or of quality equivalence for every prompt.
The two-reference LoRA sample has MAE 0.518/255, PSNR 43.80 dB and a maximum
channel difference of 87; visual inspection preserves the teapot/fox composition.

App result details now distinguish reuse of the editing prefix from reuse of
text encoding, so a text-cache hit alone is not presented as avoiding prefill.

## Validation

The tiny real-MLX lifecycle regression covers snapshot isolation, compact
storage, cap rejection, shape/config/reference rejection, 0–3 references,
base/LoRA weights, same/different seeds, and destruction of the source transformer
and weights before importing the snapshot. This small fixture is exactly equal;
the full-model numerical results above remain the quality evidence.

The final default-on build passed the production Session probe with two
references and six-step LoRA: isolated warmup, miss/hit repeats, reference
reordering/restoration, replacing reference bytes while retaining modification
time, cancellation on a hit, retry, LoRA→base→LoRA rebinding, unload and fresh
preparation. Retry/rebind full-prefill outputs matched their uncached oracle.
Cancellation left 1,425,930 active MLX bytes; unload left 522. The probe's
subsequent preparation reported a conditioning miss. The temporary warmup
disable also verifies the explicit `PREFIX_SNAPSHOT=0` escape hatch.

## Workspace presentation and interaction

The creation workspace now opens with a 270-point settings column on the right.
The toolbar can collapse and reopen it; the collapsed image workspace also keeps
a model entry point. Generation/editing show model, sampling, resolution, seed
and LoRA controls there. Upscaling uses the same column for its own settings.

A horizontal row of 52-point image thumbnails sits below the canvas for image
generation and editing, and below the scrollable preview area for upscaling.
The row includes completed image results and is hidden in video mode. Selecting
history previews that result without replacing the active prompt, parameters,
editing mode or upscale input. Parameter reuse is explicit. Upscale history has
an explicit “用作超分原图” button and context-menu action; changing the source
clears the historical preview, including when reselecting the same path.

The canvas/reference area and settings scroll separately from the generation
composer, keeping submission controls reachable. Existing image zoom controls
are also available in the upscale preview.

Live macOS checks used the final isolated `TurboCider Edit Verify` app with a
separate `/tmp` state directory and local fixture history. Checked default-open
settings, collapse/reopen, history selection in generation/editing/upscaling,
preservation of the editing draft, explicit history-to-upscale-source selection,
new-source preview reset, and image-history exclusion in video mode. The final
window was inspected visually for overlap and clipping. These checks did not
modify the user's production draft or launch additional generation jobs.

## Final build and package checks

- Native affected translation units rebuilt and CLI/shared library relinked;
  runtime identity and catalog checks passed.
- Every Swift compile command from `tools/native/build_app.sh` completed in
  batched builds; the App was rebuilt again after the final interaction fixes.
- `make test`, `make test-qwen21` and `make test-app` passed. Existing opt-in or
  unavailable-hardware suites reported their normal skips; this does not claim
  coverage of all optional model routes.
- The final full-model default-on Session lifecycle probe passed, as described
  above; small-fixture snapshot regression exercises 24 lifecycle cycles.
- `make package -o build` completed. App and CLI strict code-signature checks,
  bundled resources and all 35 App Mach-O section comparisons passed.
- The packaged App's embedded CLI `doctor` reports Apple M4 Pro,
  `gpu_available: true`, MLX 0.32.0 and no required Python runtime.

The updated local bundle is `dist/TurboCider.app`. A production App process that
was already running during this work must be quit and reopened to use this build.
