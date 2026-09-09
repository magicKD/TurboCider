# Z-Image flexible Core ML inputs — M4 Pro, 2026-09-08

Research and local experiments on Apple M4 Pro (48 GB), macOS 26.6,
MLX 0.32.0, coremltools 8.3.0. No model downloads. Uses the local Comfy
Z-Image checkpoint and the FLUX.2 Klein tokenizer/text encoder.

## Why the old cache failed

At 512×512 the transformer has 1024 image rows. Text tokens include the
chat template, and are padded to a multiple of 32 before concatenation.
The original 1056-row FFN only has room for 32 padded text tokens. A valid
55-token prompt requires 1088 rows. App preflight previously assumed 32
text rows even for longer prompts, while a reused native session checked
this under a generic FFN geometry error.

The tokenizer itself has a 512-token limit. Text beyond that limit is
rejected explicitly, never silently truncated. A 1536-row FFN covers this
limit at 512×512. Larger image sizes require larger row capacities.

## Core ML options

Apple recommends [EnumeratedShapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
for a finite set of shapes that can be optimized on device (up to 128).
The same page describes bounded RangeDim and the Infrequent reshape hint;
non-default shapes can take longer on their first prediction. The
[Core ML FAQ](https://apple.github.io/coremltools/docs-guides/source/faqs.html)
also recommends enumerated shapes when adapting an existing ANE model.

This experiment exports the actual MIL graph with a symbolic row axis,
not merely altered manifest dimensions. Shapes are [1,3840,1,rows].
The enumerated artifact has 16 choices, 1056...1536 in steps of 32. The
range artifact permits any length in that bounded interval; TurboCider
selects aligned lengths because the transformer already pads by 32.

Both use identical INT8 per-channel FFN weights, ANE prefix width 8192,
activation scale 8 and output scale 32. GPU executes attention and the
remaining FFN channels. Core ML uses CPUAndNeuralEngine; all variable
runtime sessions use reshapeFrequency=Infrequent. Compute-plan preferred
ANE devices do not prove actual hardware occupancy.

## Single-block screening

One actual noise-refiner FFN, 20 timed predictions after one first call,
deterministic FP16 inputs. Python prediction includes binding/copy costs;
these numbers are not full-pipeline timings. Milliseconds, medians:

| Export | Hint | 1056 rows | 1088 rows | 1536 rows |
|---|---|---:|---:|---:|
| Fixed | default | 28.75 | — | 40.06 |
| Enumerated | default | 28.77 | 29.51 | 41.05 |
| Enumerated | Infrequent | 28.77 | 29.56 | 41.05 |
| Range | default | 28.81 | 41.47 | 64.23 |
| Range | Infrequent | 29.19 | 29.98 | 41.56 |

Outputs were bit-identical to the corresponding fixed-shape artifact
at both endpoints in this probe. Without the hint, RangeDim's non-default
1536 shape was 60% slower than fixed1536. Enumerated shapes avoided this
large regression even with the default hint. Plans report the convolutions
and gated operations prefer the Neural Engine for all four exports.

Raw results: `outputs/z-image-flex-20260908/probe-report.json` and
`*-plan.json`. Export and benchmark tools live in `tools/coreml/`.

## End-to-end protocol

Native C ABI, one persistent engine per variant, 512×512, 9 steps, seed 42,
resident weights, no tensor dumps. Short prompt has 21 valid tokens;
long prompt has exactly 512 valid tokens. The long prompt repeats a safe
word as a controlled length stressor; this does not evaluate long-form
prompt adherence. Each run saves a PNG. Repeated prompts reuse text
conditioning, matching repeated-generation use. First shape changes and
model loads are reported separately from warm repetitions.

Pure GPU and each 1536-capable route run short ×3, long ×3, medium ×1,
short ×1. The legacy1056 route only runs short ×4 because it cannot serve
the 512-token input. Fixed1536 pads short prompts up to 1536 at the FFN;
variable routes use 1056 for short and 1536 for long. Attention always
uses the actual aligned sequence, with no extra long-padding attention.

Raw generation requests, PNGs, metrics and wall times are recorded by
`tools/native/benchmark_z_image_shapes.py` in `bench-*.json`.

## First complete run

| Route | Warm short (21 tokens) | Warm long (512 tokens) | Short vs GPU | Long vs GPU |
|---|---:|---:|---:|---:|
| GPU | 16.909 s | 23.964 s | 1.00× | 1.00× |
| Fixed1056 | 13.889 s | unsupported | 1.22× | — |
| Fixed1536 | 16.850 s | 19.779 s | 1.00× | 1.21× |
| Enumerated1056...1536 | 14.299 s | 20.425 s | 1.18× | 1.17× |

Warm values use the median of consecutive identical-prompt runs after the
first run at each length (three warm repetitions for legacy1056, two for
the other routes). Relative to fixed1056, enumerated short latency rose
2.96%; fixed1536 rose 21.3%. Relative to fixed1536 at 512 valid text tokens,
enumerated latency rose 3.27%. Supporting the longer length itself costs
more compute; flexible shapes avoid paying that cost for short prompts.

The first enumerated request took 133.819 s, including 115.527 s loading
Core ML models and doing device-specific optimization. Subsequent requests
reuse those models. First requests in the other processes were GPU20.009 s,
legacy fixed24.272 s and new fixed1536 41.888 s. These are first-process
observations with existing system caches, not controlled cache-erased cold
starts. No global OS/ANE cache was cleared.

The first long request took GPU28.864 s, fixed1536 24.710 s and enumerated
29.880 s. This includes text encoding and first use of a new shape. Thus,
there is no claim that every first request is accelerated. Returning from
medium to short took GPU20.205 s, fixed1536 20.476 s and enumerated18.335 s;
text conditioning was recomputed for these requests.

The range route initially failed during real ANE inference (ANE code8,
status0x15). A fresh process then generated successfully. This failure is
not explained by the token capacity: the input was 1056, inside the declared
range and at its default shape. The precise device/runtime root cause is
not established; it is retained as an experimental export option.

Pixel comparisons of completed images show enumerated versus matching
fixed artifacts are identical for both short and 512-token inputs. INT8
hybrid output still differs from pure BF16 GPU output; variable shapes do
not eliminate that pre-existing approximation. The short hybrid/GPU MAE is
3.174 (8-bit RGB), PSNR28.77 dB, matching the previous verification.

## Integration

- Exporter: `--shape-mode fixed|enumerated|range`, `--min-bucket`, `--bucket`.
  Legacy fixed exports retain their existing identities. Flexible shape
  definitions become part of the source and compilation cache identity.
- Native runtime: validate concrete input/output shapes; select the smallest
  compatible row count and rebind shared output storage while reusing the
  same loaded MLModels. Range runs also use the Infrequent hint.
- Exact tokenizer preflight is exposed through the additive C/Swift ABI.
  App cache discovery uses actual text length, model identity and LoRA
  identity/strength. A reused undersized legacy partition receives an
  explicit capacity error instead of the generic geometry mismatch.
- On M4 Pro48GB, the default offline ANE export recipe is the enumerated
  1056...1536 range, with prefix8192 for base and6144 for adapters. Existing
  explicit profiles remain fixed unless they request flexible shapes.
  The default execution device remains GPU; ANE is opt-in.
- Source and compiled manifests are retained; selecting a compatible
  existing compiled cache does not rerun the exporter or model compiler.
  Core ML may still perform OS-owned specialization after cache eviction,
  an OS update or a first use in a different application context.

Example explicit offline export profile:

```json
{"schema_version":1,"models":{"z-image-turbo":{"coreml_export":{
  "bucket":1536,"min_bucket":1056,"shape_mode":"enumerated","ane_mlp_width":8192
}}}}
```

## Fresh-process recheck and cache behavior

After the complete run and native regression tests, a separate process
repeated the routes. These are one warm repetition per length, not an
additional multi-sample median:

| Route | First short | Warm short | First long | Warm long |
|---|---:|---:|---:|---:|
| GPU | 19.886 s | 16.940 s | 28.433 s | 24.582 s |
| Enumerated | 24.814 s | 14.054 s | 24.795 s | 19.787 s |
| Range + Infrequent | 25.180 s | 14.635 s | 28.616 s | 20.033 s |

Enumerated model load dropped from 115.527 s to 6.913 s across process
restart. Shape changes within the same session preserved that load metric,
increased the existing per-session call counter and incurred zero output
copy bytes. It was 1.21×/1.24× faster than this recheck's GPU at short/long
lengths. The first-run throughput findings therefore survived a fresh
process and later GPU baseline check. The 3% difference from fixed shapes
should be interpreted alongside this observed run-to-run variation.

The range recheck and one isolated short generation succeeded, with pixels
identical to corresponding fixed routes. Only the original complete
sequence produced the ANE device error; no error-rate estimate is justified
by this small sample. Enumerated is the chosen default because it has
completed all tested generation sequences and is Apple's preferred finite
shape mechanism. Range remains opt-in for research.

Regression checks passed: exact native tokenizer count and 512/513 limit,
App cache routing and rejection of undersized legacy caches, Studio behavior
suite, four CoreML/LoRA unit tests, five Z-Image sharded-checkpoint tests,
and a native test predicting block0 at all16 enumerated shapes. Full32-block
generation covered1056,1088,1536. The native test also verifies that changing
shape does not reload models and that output backing is accepted without
copies. The full repository suite was not rerun for this scoped change.

## LoRA and delivery

The supplied local LoRA at strength1.0 was exported separately with prefix6144
and the same16 enumerated shapes. A short→512-token→short sequence completed:
1056→1536→1056 rows,180 applied projections, verified LoRA identity, and zero
output-copy bytes. Request walls were90.075 s,24.396 s,16.450 s respectively.
These include first model preparation or text recomputation, and are
functional checks rather than a repeated warm LoRA performance comparison.
The short image is pixel-identical to the previous fixed1056 LoRA result;
the long result is a valid512×512 fox image.

Reissuing compilation for both base and LoRA returned32/32 cache hits each.
All128 compiled files per route retained their timestamps. Evidence:
`outputs/z-image-flex-20260908/cache-reuse.json` and `lora/bench-enumerated.json`.

The App was rebuilt, packaged at `dist/TurboCider.app`, and its ad-hoc
signature passed deep/strict verification on branch `dev-verify`.
Native/Swift integration tests passed.

## App configuration and UI verification

On the subsequent configuration request, the unlocked App imported
`outputs/z-image-flex-20260908/app-flexible-config.json` successfully. This
configuration explicitly selects GPU + ANE and registers both validated
enumerated manifests (base8192 and LoRA6144). New App drafts still default
to GPU; the current saved draft uses GPU + ANE as requested. The supplied
LoRA remains unchecked at strength1.0, with its compiled cache registered.
The import artifact includes a safe example prompt because configuration
import validates that the prompt is nonempty.

Three actual App generations completed at512×512,9 steps,seed42:

| Input | Selected rows | App wall | Denoise |
|---|---:|---:|---:|
|512 tokens, first App load|1536|142.417 s|24.099 s|
|512 tokens, reused session|1536|24.245 s|23.618 s|
|21 tokens, same session|1056|22.808 s|17.335 s|

The first request included114.901 s of Core ML model loading. The App
reported reuse of the compiled cache without recompilation. Across all
three requests, the model-load metric stayed unchanged, calls increased
288→576→864, and output-copy bytes remained0: changing input length reused
the existing32-block session. Every generated image was pixel-identical
to its corresponding native enumerated reference. The Core ML compute
units were CPUAndNeuralEngine; measured ANE residency remains unknown.
These App timings are functional checks under the current desktop load,
not a replacement for the controlled benchmark above.

Evidence: `outputs/z-image-flex-20260908/app-ui-validation.json`.
The final saved draft has an empty prompt, random seeds restored,512×512,
GPU + ANE enabled, and LoRA disabled at1.0. Existing history was preserved.
This configuration supports variable sequence length (up to512 encoded
text tokens at512×512), not arbitrary output resolutions with the same
partition set.
