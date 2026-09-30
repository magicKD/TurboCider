# Qwen-Image-2.1 runtime LoRA with an unfused adapter and a fused base FFN

Status (2026-09-28): experimental, **no established whole-request advantage
over full GPU**. Earlier paired runs were slower; later interleaved runs
were near parity with an apparent ~1% advantage below run-to-run variation. Keep
full GPU runtime LoRA as the accurate/default route, and keep the existing
fused base GPU/Core ML path for requests without LoRA. No LoRA is merged into
the base checkpoint or Core ML weights.

The diagnostic `hybrid_mlp_mode: "lora_fused"` uses a complete frozen-base
Core ML FFN graph: gate/up projection, SiLU, multiplication, and base down
projection all remain on the Core ML side. `lora_fused` names the base-operator
fusion, not a merge of LoRA weights: the artifact is independent of the
adapter's file, rank and strength. At inference the GPU computes
runtime gate/up LoRA contributions and supplies them as a *second activation
input* before SiLU. The graph also returns its pre-hidden-A8 activation for
the GPU's runtime LoRA down projection. The GPU processes the complementary
6144 channels and all non-FFN LoRA projections normally. Thus gate/up and
down LoRA are not silently omitted; however W8A8 base arithmetic and FP16
Core ML boundaries are not bitwise equivalent to the BF16/FP32 GPU oracle.
This is distinct from the faster, **mathematically incomplete** suffix-only
diagnostic and the older gate/up-only graph that moves SiLU/down to GPU.

Export with `tools/coreml/export_qwen3.py --qwen21-fused-lora` plus the
existing fixed 1024-row, 6144-channel Qwen21 W8A8 settings. Compile the
manifest with `tools/coreml/compile_z_image_manifest.py --model
qwen-image-2.1`. Both source checkpoint provenance and the runtime LoRA
binding are checked; the compiled base graph contains no adapter identity.
The runtime mode is opt-in, is rejected with an incompatible manifest, and
does not change the base request's existing fused artifact or numerical policy.
The same compiled `lora_fused` artifact can be explicitly selected for a base
request (zero dynamic gate/up correction, no down-LoRA) and for successive
runtime adapters; the resident Core ML session is retained while MLX adapter
state is rebound. A base 512px request can use 5, 6, 20, or 40 steps in this
mode. The existing faster base default remains unchanged. Unlike the pinned
Viggle-only default LoRA route, explicit `lora_fused` permits a different
single transformer LoRA and strength in [-8, 8], but uses the **Viggle
six-step student schedule**: such alternate adapters need independent
schedule and visual qualification. No speedup or image-quality claim is made
for them. The adapter SHA and strength are part of the resident GPU binding
identity, never of the frozen Core ML artifact.

### Shared-artifact switch check (2026-09-28)

One resident CLI batch ran six-step base → local Viggle v0.2.1 LoRA at scale
1 → base, all with the **same** 32-layer W8A8 manifest. All three generated
valid 512px images; the two base PNGs were byte-identical (SHA-256
`421170451e1bfb3cc14da615ea3901794a96a82f624264d91d8360a54c7892cb`),
and the Viggle image was different. The base fox has softer detail than the
LoRA fox at this short, six-step base schedule; this is **not** a six-step
base quality qualification. A separate run of the same LoRA at strength 0.5
also produced a different image with this manifest; a later **single resident
batch** of strength 1 → 0.5 → base completed and reproduced the same base
PNG hash above. Only one *trained* Qwen adapter file is installed; a later
temporary synthetic second-file switch test is documented below, but cannot
qualify a second trained adapter's image quality or schedule. These
cold/switch timings are not comparable with the warm route measurements below.

## Measurements

M4 Max, 512×512, six-step Viggle v0.2.1 r256 runtime LoRA at scale 1,
identical fox prompt, seed 42, and matching text/noise dumps; both routes
used FP16 low-rank GPU matmuls and the Metal Q/K normalization/RoPE option.
Prepared and fully warmed resident sessions (Core ML model load excluded):

| Route | Wall run 0 | Wall run 1 | Ratio vs GPU |
| --- | ---: | ---: | ---: |
| Full GPU runtime LoRA | 7.814 s | 7.815 s | 1.000× |
| Core ML fused base with runtime LoRA activation input | 9.623 s | 9.551 s | **0.815×** |

The new route makes 160 Core ML predictions per measured request, with zero
reported Core ML output copies or runtime failures. Prediction API time alone
is 3.987 s on run 1, about 24.9 ms/call. End-to-end GPU/ANE is ~22.7% slower
than full GPU, notwithstanding a **1.48× single-FFN microbenchmark** (25.85
ms GPU vs 17.43 ms GPU/Core ML median for 1024 real held-out rows). The
single-layer speedup does **not** generalize: 32 serial blocks contend for
GPU/ANE shared resources, require 24 MiB of runtime delta input and 20 MiB of
output per prediction, synchronize LoRA delta before every Core ML call, and
transfer intermediate activations. The measured prediction API time includes
framework synchronization; it is not proof of physical ANE placement.

On one held-out block-0 input, both old base fused W8A8 and this runtime
LoRA fused path had ~3.40% relative FFN RMSE against their respective full
GPU reference. The block-0 LoRA contribution itself was tiny, so this is
not a sufficient all-block numerical qualification. At the complete-image
level the fox retains the main subject and background, but the forepaw and
tail vary visibly; normalized RGB RMSE 0.0559 and correlation 0.9777. More
seeds, text/detail challenges, and 1–3-reference edits remain unqualified.

The measured wall target for 1.10×/1.20×/1.30× versus this same GPU request
is respectively 7.10/6.51/6.01 s. From the current 9.59 s median this would
require shaving ~15.5/19.2/22.4 ms **per one of 160 decode calls**, assuming
all other work stays fixed. The 24.9 ms measured Core ML API time/call means
1.10× could be explored through lower synchronization/ABI cost, but 1.20×
requires a major redesign of how deltas and hidden activations cross the
GPU/Core ML boundary; 1.30× is **not supported by current evidence**. These
are required savings, not predicted speedups. The suffix-only path's earlier
~6.5 s is a useful performance reference but **cannot** qualify as correct
runtime LoRA because it omits prefix adapter contributions.

Next experiment if pursuing a speedup: profile individual GPU delta, Core ML
prediction, GPU suffix, and output-consumption barriers over all 32 blocks;
reduce the LoRA input/hidden output traffic without moving SiLU/down off the
fused base graph, and reject any version that fails end-to-end timing or image
inspection. Do not adopt this slower diagnostic as the CLI default.

### Further profiling and partition screen (2026-09-28)

The optional `TURBOCIDER_QWEN21_PROFILE_RUNTIME_LORA_FFN=1` logs per-block
`input_ready_seconds`, `gpu_graph_submit_seconds`, `lora_delta_wait_seconds`,
and `predict_and_submit_seconds`. Two actual six-step runs (320 calls total)
measured 4.52 s input readiness, 0.84 s LoRA delta evaluation, and 6.29 s
prediction **including** that delta evaluation. These spans overlap with
deferred GPU work; do not add them to get end-to-end wall time. The delta
barrier alone averages ~0.42 s/request. This is measurable, but removing it
entirely would still require a different, correct pre-SiLU LoRA boundary.

The original BF16 sliced GPU suffix was compared with a prepacked base
gate/up suffix plus separately added runtime LoRA. Same seed/manifest and
prompt produced a **byte-identical** PNG, but matched warm image times were
9.85/10.18 s (first ordering) and 10.55/9.63 s (reversed ordering). The
change was reverted: no speed claim is justified. Subsequent warm runs of
the restored route varied from 8.08 to 8.55 s, versus matched GPU 7.78–7.81 s;
the older separately prepared 9.55/9.62 s result above must not be silently
replaced by one favorable run. Neither set establishes a robust Qwen hybrid
win over GPU.

Holding the **same held-out real block-0 input and Viggle adapter** fixed,
the frozen-base W8A8 fused-LoRA FFN yielded:

| Core ML prefix | Single-layer GPU/Core ML median | Relative RMSE vs full GPU | Delta input + output per 1024-row call |
| --- | ---: | ---: | ---: |
| 4096 channels | 19.95 ms | 3.14% | 16 + 16 MiB |
| 6144 channels (existing full route) | **17.83 ms** | 3.40% | 24 + 20 MiB |
| 8192 channels | 23.18 ms | 3.69% | 32 + 24 MiB |

The matched GPU single-layer median was ~25.8 ms. The 4096/8192 source
packages were exported, compiled, and tested as **block-0 only**; their
geometry is accepted for offline probing but no 32-layer runtime route is
enabled. Neither width reduced single-layer latency versus 6144, so neither
was promoted to a costly full-image export. These CPU/Neural Engine-policy
call times still do not prove physical ANE placement. Future work should
reduce gate/up LoRA computation and per-block synchronization without
dropping any adapter projection, then re-run matched GPU and hybrid
end-to-end tests before changing defaults.

### Three-reference 512px edit check (2026-09-28)

The same frozen-base 6144-channel, 32-block manifest was exercised on a
six-step `image.edit` with three 512px-diagnostic references (beige teapot,
blue teapot, orange dragon decal), Viggle runtime LoRA, seed 29, and a 512px
output. The request uses `qwen21_reference_size: 512`; this is an explicit
downsampled-reference diagnostic, **not** full-resolution edit qualification.
The relative request files are
`results/qwen21/runtime-lora-fused-edit3-ref512-{gpu,hybrid}-request.json`.
Both routes bound all 227 LoRA projections; the hybrid used the same base
Core ML graph as text-to-image and made 160 predictions per request. In each
resident three-request batch, the first request was excluded as cold:

| Route | Two warm whole-request times | Two warm denoise times |
| --- | ---: | ---: |
| Full GPU runtime LoRA | 12.347, 12.324 s | 11.688, 11.678 s |
| Shared base Core ML + runtime GPU LoRA | 12.335, 12.659 s | 11.049, 11.382 s |

Median whole-request speedup is approximately **0.987×** (no established
win); denoise alone is approximately 1.04×, but the hybrid has ~0.60 s of
route overhead on each warm request. Both resulting images visibly preserve
the two distinct teapots and the small white-bordered dragon on the blue
teapot, not on the table. Compared pixelwise to the GPU image, the hybrid
has RGB RMSE 0.0371 on [0,1] and correlation 0.9921; modest decal/body
detail varies. This is one prompt/seed and a visual spot check, not a general
editing-fidelity claim or proof of physical ANE execution. GPU remains the
conservative default for Qwen runtime LoRA and for edits.

### Allocation experiment and shared-base timing (2026-09-28)

The same generic `lora_delta_slice` first-delta optimization used by Z-Image
was checked on this three-reference Qwen edit. The updated hybrid's two warm
whole-request times were **12.176/12.504 s**, with all 227 LoRA projections
bound and the resulting image visually unchanged; the previously recorded
GPU pair was **12.347/12.324 s**. The two-run medians are effectively tied.
These are not a controlled code-level A/B or proof of a speedup; continue to
prefer GPU for runtime LoRA until a repeatable whole-request win is shown.

One no-LoRA request using the *same explicit runtime-compatible Core ML
manifest* ran at 6.246/6.196 s warm. Caching its 24-MiB all-zero pre-SiLU
input once per session left the base PNG hash byte-identical but gave
6.366/6.284 s warm: the caching code was therefore reverted. This does not
affect or replace the separate fastest base-only graph and routing policy.
After the runtime delta change, a single resident base → Viggle LoRA → base
batch again returned byte-identical base PNGs (the same `42117045...` hash
above), with 227 adapter projections bound only on the middle request.

### Shared LoRA A experiment rejected (2026-09-28)

Qwen's runtime suffix and Core ML pre-SiLU prefix both consume the same
gate/up input. An experimental request-local cache shared each bound LoRA A
projection between those two branches; a synthetic two-branch numerical
probe passed for FP32/FP16 low-rank arithmetic, and the three-reference edit
PNG was byte-identical (`7a360947eeff4806477d17031701d245adf6ab378d8358669336f6ecaa9a7f19`).
However, a same-binary four-process off/on/on/off paired screen of the
six-step 512px Viggle text-to-image request showed two warm whole-request
times per process:

| Shared gate/up LoRA A | Process 1 | Process 2 | Four-run median |
| --- | ---: | ---: | ---: |
| Off | 7.584, 8.169 s | 7.947, 8.598 s | 8.058 s |
| On | 8.783, 8.296 s | 7.598, 8.994 s | 8.540 s |

The screen is variable and shows no repeatable whole-request win. The
shared-low implementation and its diagnostic switch were removed after the
screen, preserving the original runtime-LoRA adapter ABI and default path.
This negative result points toward serial Core ML boundary/hidden-output
costs instead of repeated LoRA A alone. Another performance idea must pass
an end-to-end comparison without dropping pre-SiLU or down-LoRA corrections.

### Packed FP16 delta input screen rejected (2026-09-28)

The base Core ML branch already needs a contiguous FP16 copy of each FFN
input. A temporary opt-in variant fed that copy rather than the original
BF16 tensor into only the GPU LoRA gate/up correction; it left the GPU suffix
and frozen base graph alone. This slightly changes rounding, so image quality
was checked rather than assumed. On the matched 512px fox, RGB RMSE versus
the prior hybrid output was **0.01090**, correlation **0.99917**; for the
three-reference 512px teapot/decal edit, RMSE was **0.01110**, correlation
**0.99929**. Subjects and decal placement looked the same at normal size.

A same-binary off/on/on/off resident text-to-image screen, with two warm
requests per process, measured off **7.930/8.123, 7.824/8.627 s** and on
**8.108/7.893, 7.827/7.881 s**: medians ~8.027 versus ~7.887 s. That small
apparent gain was not reproduced on the edit: two warm packed-input edit
requests took **12.612/13.157 s**, versus the earlier unmodified hybrid's
12.176/12.504 s. The experiment is therefore *not* a demonstrated useful
whole-request acceleration. The packed-input code and diagnostic switch
were removed; the base-only default and runtime-LoRA paths remain intact.

### Core ML call boundary breakdown (2026-09-28)

Phase telemetry was added to the existing session metrics. It times feature
binding, the synchronous `predictionFromFeatures` call and output handling
separately; the counters are session-cumulative, so the numbers below are
*differences* between consecutive warm requests. On two 512px/six-step
Viggle requests (160 Core ML calls/request), the measured spans were:

| Component per warm request | Run 1 | Run 2 |
| --- | ---: | ---: |
| Bind input and LoRA delta feature providers | 2.16 ms | 2.06 ms |
| Core ML `predictionFromFeatures` | 2.608 s | 2.613 s |
| Inspect/handle output | 0.86 ms | 1.01 ms |
| Full prediction bridge | 2.611 s | 2.616 s |
| End-to-end request wall | 7.935 s | 7.913 s |

The fixed-shape graph reported zero output copies. The host-side
`MLMultiArray`/feature binding and output handling together are under 0.02 ms
per call; optimizing those allocations cannot plausibly close the Qwen gap.
The ~16.3 ms/call inside `predictionFromFeatures` includes Core ML scheduling
and possible synchronization with GPU work; these host spans do **not** prove
that the entire graph physically ran on ANE. Next performance experiments
should address the fused Core ML model/partition cost or the per-layer
GPU/Core ML dependency, while still calculating all adapter branches.

### Adapter-independent base graph and linear-down screen (2026-09-28)

The `lora_fused` name refers to a *fused base FFN*, **not** LoRA baked into
Core ML weights. Its source manifest has only the base checkpoint provenance,
without `source.loras`. Gate/up LoRA enters through the dynamic pre-SiLU input,
and GPU down-LoRA uses the returned hidden activation. Both the original
conv-down graph and a separately exported 32-layer linear-down graph can be
reused for base and runtime adapters; the latter was tested with one installed
Viggle adapter, not a second distinct adapter file. The linear-down graph is
`results/qwen21/ane-mlp-fused-runtime-lora-linear-compiled-32/manifest-c31ef9577a879d1e2526f59c3e66e64e1b8e88d5a39e9beebf8dc994ad32577d.json`.

With `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` and
`TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1`, two warm six-step 512px text-to-image
requests in each of two alternating batches (conv → linear → conv → linear)
measured conv-down **8.026, 9.003, 8.884, 7.751 s** and linear-down
**7.743, 8.204, 8.184, 8.723 s**. Their four-run medians are approximately
8.455 and 8.194 s respectively. Run-to-run variance is larger than the small
apparent difference. Without these GPU/LoRA flags, matched batches measured
conv-down **10.777/10.755 s** and linear-down **10.856/10.990 s** warm.
Both down lowerings produced byte-identical LoRA PNGs under each matched
configuration; all runs bound 227 LoRA projections and used 160 Core ML calls.

For a 512px output with three diagnostic resized-512 edit references,
`TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC=1` was additionally required.
Two warm whole-request times were **13.171/13.343 s** with linear-down and
**12.383/13.146 s** with conv-down. Their edited PNGs were byte-identical
(SHA-256 `7a360947eeff4806477d17031701d245adf6ab378d8358669336f6ecaa9a7f19`),
but linear-down did not improve edit latency. Neither variant has an
established end-to-end LoRA speedup over GPU; the original conv-down graph
remains the documented diagnostic, not an automatic replacement. The faster
base-only default is unchanged.

The new linear-down manifest also completed one resident **base → LoRA →
base** batch with no reloaded Core ML model: cumulative call counts were
160/320/480; the LoRA request bound 227 projections and the two base requests
bound none. Separate base output files had identical SHA-256
`3d7d54431fa00fedfbbe71c64d5ea1a953ca8790df8bb678fb864906bd671e59`.
The two down lowerings need not produce identical *base* output PNGs, and a
short six-step base image is not a base quality qualification. These checks
establish artifact reuse across this installed adapter and no adapter-leakage;
alternate adapter files still require visual and schedule qualification.

### Direct-FP16 delta screen rejected for Qwen (2026-09-28)

An opt-in screen removed the BF16 round-trip from just the dynamic Core ML
gate/up LoRA input (FP32 low-rank accumulation → FP16 instead of FP32 → BF16
→ FP16); the frozen base graph and all adapter projections stayed intact.
With the same FP16 LoRA and Metal Q/K options, four warm whole-request times
across off → on → on → off batches were **7.337/8.023, 7.692/7.538 s off**
and **7.500/7.837, 8.116/8.197 s on**. Four-run medians were ~7.615 and
~7.977 s: no speedup. The changed Qwen PNG differed from the original by
RGB RMSE 0.0109, correlation 0.99917 on the checked fox, with no visually
meaningful composition change. The experimental Qwen branch was **removed**;
the established default and shared-graph runtime route keep their previous
BF16-to-FP16 boundary semantics. The smaller-rank Z-Image adapter has a
separate, opt-in positive result; it must not be generalized to Qwen.

### Separate GPU suffix stream rejected (2026-09-28)

An experimental second MLX GPU stream submitted the independent Qwen runtime
LoRA FFN suffix *before* materializing the pre-SiLU delta that Core ML needs.
With the same fixed 6144-channel base graph, Viggle adapter and Metal/FP16
GPU options, the two warm whole-request times were **10.368/10.656 s** with
the extra stream versus **7.641/7.864 s** without it, in successive resident
batches of the same binary. All 227 LoRA projections remained active and
the generated PNG was byte-identical. Extra GPU scheduling/stream dependency
or unified-resource contention is a plausible explanation, not a separately
measured root cause. The slow experimental stream branch was removed; moving
the suffix to another stream must not be advertised as GPU/ANE parallel gain.

### Matched current GPU versus shared-base route (2026-09-28)

Using one native binary, identical six-step 512px Viggle r256 requests, seed
42, `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1`, and
`TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1`, a GPU → shared-base Core ML →
shared-base Core ML → GPU process order yielded two warm whole-request
times per process (one cold request excluded):

| Route | Four warm requests | Median |
| --- | --- | ---: |
| Full runtime LoRA GPU | 7.805, 7.816, 7.751, 7.828 s | 7.811 s |
| Frozen base 6144-channel Core ML + runtime GPU LoRA | 7.643, 7.906, 7.557, 7.786 s | 7.715 s |

The nominal ratio is ~1.012×, not a robust speedup. All eight warm requests
bound 227 adapter projections; hybrid requests used 160 predictions each.
The corresponding PNGs were deterministic within each route, with the
previously documented visible forepaw/tail differences between routes. The
competing ComfyUI service was resident at 0% CPU and no download helper was
active during timing. This matched result supersedes **using the earlier
9.55-second hybrid result alone** as the current performance estimate, but
does not establish a general Qwen GPU/ANE win or justify changing the CLI
default. The explicit mode remains useful for testing frozen-base artifact
reuse and other adapters without recompiling Core ML.

### Distinct adapter-file identity check (2026-09-28)

`tools/validation/runtime_lora_shared_graph_switch.py` made a temporary
**synthetic** adapter file by halving all 227 LoRA-B tensors in the installed
Viggle checkpoint. One resident six-step base → Viggle → synthetic adapter →
base sequence used the same compiled conv-down Core ML graph throughout:
runtime call counts 160/320/480/640, bound projections 0/227/227/0, no
reloaded Core ML model, distinct PNG hashes for the two adapters, and
byte-identical base PNGs on return (SHA-256
`3d7d54431fa00fedfbbe71c64d5ea1a953ca8790df8bb678fb864906bd671e59`
with the Metal Q/K and FP16 GPU-LoRA options). The synthetic adapter and
scratch images were removed after the check. This proves *runtime file
identity rebinding and frozen-base reuse* for a second file, **not** output
quality or a valid six-step schedule for a second trained adapter.

### Full-GPU FFN blocks 3/5/7 rejected for runtime LoRA (2026-09-28)

A temporary explicit split left the same frozen 6144-channel Core ML graph
on 29 blocks, while routing blocks 3, 5 and 7 to full BF16 GPU FFNs with
*all* runtime gate/up and down LoRA projections. It made 145 rather than
160 Core ML calls per six-step 512px request and still bound all 227 adapter
projections. Against the same full-GPU PNG, RGB [0,1] RMSE improved from
**0.05594** (32 hybrid blocks) to **0.02632** (29 hybrid blocks), correlation
from **0.97773** to **0.99538**. The fox differed visibly less from full GPU,
but speed remains the priority once images are broadly acceptable.

On the same native binary with the Metal Q/K and FP16 LoRA options, warm
whole-request times were **8.872/8.609 and 8.047/8.990 s** for the fallback
route, versus **7.893/7.548 s** for the 32-block shared graph in an
interleaved batch. The fallback is slower than both the shared graph and
the previously matched ~7.81-second full GPU route, which is more accurate.
The experimental fallback entry point and implementation were removed after
this screen. The result suggests early-block W8A8 error amplification, but
does not isolate which of the three blocks is responsible or prove that
recompiling selected base Core ML blocks in higher precision would be
profitable.

### Deferring the GPU suffix rejected (2026-09-28)

The runtime gate/up LoRA delta must finish on GPU before the frozen-base Core
ML call. Ordinarily the independent GPU FFN suffix is then submitted before
the Core ML prediction to allow overlap. A temporary switch deferred that
same, unchanged suffix until after the prediction, testing whether avoiding
simultaneous GPU/Core ML traffic could help. The base graph, all 227 bound
LoRA projections, 160 predictions/request and output PNG remained identical.

An off → on → on → off four-process screen used the six-step 512px Viggle
request, the Metal Q/K and FP16 LoRA GPU options, and two warm whole requests
per resident process after one excluded cold request. The four original
wall times were **8.103/7.874/7.933/8.496 s** (median **8.018 s**), versus
**9.996/9.855/9.904/9.809 s** (median **9.880 s**) with the suffix deferred.
The late schedule is approximately **23% slower** in this screen and was
removed. This supports preserving the existing overlap, but neither Core ML
API latency nor the timing difference establishes which hardware executed
individual graph operations. The temporary request/PNG files were cleaned by
`tools/validation/hybrid_runtime_schedule_screen.py`; the reusable screen
script remains available for later opt-in scheduling ideas.
