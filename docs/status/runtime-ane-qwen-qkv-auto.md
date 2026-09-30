# Qwen runtime QKV: complete-block on/off experiment

The explicit `runtime_qkv` route now accepts
`TURBOCIDER_RUNTIME_ANE_QKV_CHUNKS=auto`. Its default remains **fixed one
1536-row ANE tail chunk**. Auto mode does not choose a new row partition or
combine QKV with the FFN ANE graph; it decides whether the existing one-chunk
split beats the ordinary, unsplit GPU block.

The controller keeps independent timing for each layer and row count. It
warms a hybrid block and an ordinary GPU block once each, measures complete
blocks (including staging, QKV join, attention and GPU FFN), and samples the
opposite route again every 32 visits. An enabled layer times hybrid blocks
before and after the periodic GPU probe; a disabled layer times its hybrid
retry. Other steady blocks do not acquire whole-block timing fences. Hybrid
must beat GPU by 5% to remain on, and by
8% to turn back on after disablement. A GPU-only decision uses the ordinary
compiled block and stages no ANE weights. Its QKV output is not materialized
through the external projection ABI.

## 1024² five-step screen

M4 Max, Qwen Image 2.1 base, seed 42, 1024×1024, 5 steps; one cold and two
resident warm requests per route. Same final library in both route orders:
`c07c1bb0b34016e05f4d1155cf4da8951a194c891aca6b403c870e446ff09390`.
The numbers below are warm **request-wall seconds**, not projection timings.

| Order | GPU warm | QKV auto warm | GPU median | QKV auto median |
| --- | --- | --- | ---: | ---: |
| QKV → GPU | 25.277, 25.269 | 25.182, 25.324 | 25.273 | 25.253 |
| GPU → QKV | 25.263, 25.293 | 25.293, 25.367 | 25.278 | 25.330 |

Each QKV trial executed 128 ANE predictions and 352 full GPU blocks across
three requests, with zero failures and zero partial-tail fallbacks. QKV counts
are cumulative: after the first two requests the third request made **zero**
new ANE calls and ran all 160 blocks on the ordinary GPU route. In the forward
trial, the third QKV-auto PNG and the third GPU PNG have identical SHA-256
bytes. The mixed hybrid/GPU warm PNG is not identical (RGBA pixel MAE 0.580,
RMSE 2.001), so this is not a general visual-quality qualification.

Raw receipts, generated images and complete summaries are in
`outputs/runtime-ane/qwen-qkv-auto-screen-forward-v2/` and
`outputs/runtime-ane/qwen-qkv-auto-screen-reverse-v2/`. An earlier exploratory
QKV-only run under the preceding library was rejected by the screen harness:
the harness incorrectly required increasing Core ML calls on every warm
request even after the controller deliberately disabled ANE. That harness
contract is now tested: an idle auto request is accepted only if its GPU block
count advances; a fixed route, failure, or missing ownership receipt still
fails. The two complete screens above use the corrected same library.

**Decision:** the reverse-order difference changes sign and is much smaller
than a practical performance margin. This demonstrates functioning on/off
and an exact GPU fallback on the matched third request, not a proven QKV
whole-request gain. Keep default selection and fixed-chunk research behavior
unchanged. The optional graph remains resident during auto-disabled blocks;
these screens did not sample process-tree/driver memory. They cannot establish
a 40-step win, memory saving, stable multi-prompt quality, or physical ANE
residency. The previously faster runtime FFN and frozen W8A8 routes were not
re-benchmarked with this library, and Z-Image QKV remains out of scope.

Verification on this build: native build, Qwen compiled-block switching and
numerical parity, 121 native contract/host tests (one fixture skip), 50
screen/host tests, and 12 Core ML/MLX integration tests passed. The next
performance gate is a same-library 40-step, reversed-order, multi-hot screen
with process-tree memory sampling, then comparison with FFN runtime and frozen
W8A8; only a stable complete-request gain should motivate widening the route.

## 40-step same-library follow-up after the Q/K gate fix

The restricted 1024² resident base `runtime_qkv` route now accepts the same
explicit Q/K norm-RoPE switch as the GPU, FFN runtime and frozen routes. The
first `c07c1bb0…` four-route attempt remains **incomplete** because QKV was
rejected before inference; its first three timings are not pooled with these.
The new library SHA-256 is
`875b0d3dcf5c766feeced21d71f1a366acf0a1238bc3f0ed332098aa3b27b8cd`.
The hardware Core ML/MLX regression (`make test-runtime-ane`: six host and 12
integration tests) passed outside the sandbox before timing. The library hash
was unchanged after both Qwen and Z-Image screens.

Qwen Image 2.1 base, 1024², 40 steps, the same fox prompt/seed 42, resident,
Q/K norm-RoPE enabled on **all** routes, FFN runtime c1792/auto and QKV
one-chunk/auto. Each route ran one cold and one warm request; the timings below
are complete warm request-wall seconds (VAE/PNG included), not component times.
Each trial had an independent 100 ms process-tree sampler and a complete
verification report. The two sequences used the same library and manifests.

| Route | GPU → runtime → frozen → QKV | QKV → frozen → runtime → GPU | Mean warm s | GPU / route |
| --- | ---: | ---: | ---: | ---: |
| GPU | 183.067 | 183.087 | 183.077 | 1.000× |
| FFN runtime | 147.113 | 146.889 | 147.001 | 1.245× |
| Frozen W8A8 | 151.874 | 151.658 | 151.766 | 1.206× |
| QKV auto + GPU FFN | 182.907 | 183.014 | 182.961 | 1.001× |

Each QKV trial counted 192 ANE predictions/hybrid blocks and 2,368 ordinary
GPU blocks across its two requests (64 additional ANE calls in the warm
request), with no failures or partial-tail fallback. The auto controller
mostly chose GPU; its approximately 0.12 s pooled difference from GPU is not
a demonstrated product gain. Runtime FFN remains faster than the frozen and
QKV paths for this workload. The QKV warm images were visually close to GPU
for this one prompt, but not byte-identical (RGB MAE 0.567/255 in both orders).
This is neither multi-prompt quality qualification nor proof of physical ANE
residency.

Process-tree peak physical footprints ranged from 30.416–31.552 GB for GPU,
32.122–32.124 GB for FFN runtime, 39.052–40.226 GB for frozen and
32.773–33.909 GB for QKV. All eight sampling receipts completed with zero
sampled swap-in/out; frozen recorded 17.5/39.6 MB compression. These peaks
cover the process tree, not independent driver/service attribution, and are
not paired instantaneous incremental memory costs.

Raw requests, PNGs, JSONL, memory streams, verifier reports and complete
summaries: `outputs/runtime-ane/qwen-qkv-auto-40step-forward-875b/` and
`outputs/runtime-ane/qwen-qkv-auto-40step-reverse-875b/`. Preserve optional
QKV-auto for research; do not enable it by default or claim a 40-step win.
The matched Z-Image 512² regression on this library is recorded in
[the Z-Image whole-block note](runtime-ane-z-block.md).

### Fixed two-chunk QKV screen

The same library, manifest, Q/K switch, 1024²/40-step prompt/seed and
independent 100 ms memory sampling were used for a separate GPU → fixed-two
QKV comparison. One cold and one warm request per route completed; both
verifier reports are complete with no sampled swap-in/out. GPU warm was
183.094 s, fixed-two QKV warm was 206.814 s: QKV was **12.96% slower** than
GPU in this screen. QKV actually ran two 1536-row predictions in every one
of the 2,560 blocks across the two requests (5,120 successful predictions,
zero GPU-only blocks, failures or tail fallbacks). The sampled process-tree
peak footprints were 31.552 GB GPU and 33.554 GB QKV; this is not an
instantaneous incremental allocation measurement. The warm QKV and GPU images
looked close but were not identical (RGB MAE 0.582/255).

Evidence: `outputs/runtime-ane/qwen-qkv-fixed2-40step-gpu-qkv-875b/`.
Given the large complete-request regression, this candidate was not repeated
in the reverse order and is not proposed as a replacement for QKV-auto, FFN
runtime or the default GPU route. Further QKV partition experiments would
need a new measured rationale; component MatMul speed alone is insufficient.
