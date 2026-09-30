# Z-Image Turbo: reusable base FFN + inference-time LoRA on GPU/Core ML

Status (2026-09-28): explicit experimental `hybrid_mlp_mode: "lora_fused"`.
The base artifact contains **no adapter weights or identity**, and the adapter
is never merged into the base checkpoint. The **same** explicit `lora_fused`
manifest now serves base (zero dynamic gate/up delta; no down-LoRA), one LoRA,
and a different LoRA. A resident session retains its Core ML model across
adapter changes with the same manifest; GPU transformer/adapter state is
reloaded. The mode name refers to fused **base FFN operators**; the runtime
adapter remains separate and can change without compiling another graph.
Automatic base GPU/Core ML routing and the fastest no-LoRA graph
remain unchanged. `lora_suffix` remains a
separate, mathematically incomplete diagnostic; `lora_merged` is an
adapter-bound artifact and does **not** satisfy the runtime LoRA requirement.
During this work we also found that the old *dense compiled GPU suffix*
silently bypassed its runtime adapter: prior suffix-only times and PNGs may
have omitted LoRA on **both** the ANE prefix and GPU FFN suffix. The suffix
branch now explicitly uses runtime-aware slice projections when adapters
are active; it still omits the ANE-prefix correction and is not the student
model. Do not compare the older suffix-only numbers against this correct route
as if both calculated the same function.

## Partition and correctness boundary

Export a fixed 1056-row, FP16-activation, 4096-channel prefix with
`tools/coreml/export_z_image.py --runtime-lora-fused --ane-mlp-width 4096
--bucket 1056 --variant int8_pc --activation-precision fp16` and compile its
full 32-block manifest with `tools/coreml/compile_z_image_manifest.py`. The
base Core ML graph retains gate/up, SiLU, hidden multiplication and base down.
At each block the GPU computes runtime LoRA gate/up for the prefix, supplies
both **before** SiLU as a dynamic FP16 input, and runs the independent full
runtime-LoRA suffix concurrently with Core ML inference. The graph returns
base down plus the pre-down hidden; the GPU computes the prefix LoRA down from
that hidden, applies the base output scale and sums both partitions. For the
two 1024-row noise-refiner blocks the host pads **both** base input and LoRA
delta to the 1056-row graph, then discards the padded outputs. Attention and
other projections continue to apply their runtime adapters on GPU.

The base gate/up and down matrices are INT8 per-channel weight compressed,
not W8A8: this first diagnostic deliberately keeps FP16 activations to avoid
adding activation calibration and hidden reparameterization into an unproven
LoRA ABI. Arithmetic and FP16 boundaries are approximate compared with full
BF16/FP32 GPU, but no LoRA projection is deliberately omitted. The API asks
Core ML for CPU/Neural Engine compute units; actual per-op ANE residency has
not been measured. Checkpoint SHA and graph geometry are validated at load;
the 1056-row manifest is not interchangeable with the existing base or Qwen
manifests. `execution=auto` never selects this experimental LoRA mode.

## Measurements on M4 Max, 512×512, eight steps

Same fox prompt, seed 42, local Comfy base checkpoint and genuine
`z_image_turbo_distill_patch_lora_bf16.safetensors` at scale 1. Resident CLI
`batch` requests are run three times per process; the first includes load and
is excluded. Four warm runs per route across alternating process order,
including VAE and PNG, yielded:

| Route | Four warm request wall times | Median | Relative speed |
| --- | --- | ---: | ---: |
| Full runtime LoRA GPU | 8.665, 8.733, 8.741, 8.597 s | 8.699 s | 1.000× |
| Frozen-base fused FFN + runtime LoRA GPU/Core ML, 4096 channels | 7.381, 7.263, 7.246, 7.225 s | 7.254 s | **1.199×** |

Both paths reported 238 bound runtime LoRA projections; the hybrid performed
256 predictions/request with no reported failure. In the last warm hybrid
request prediction API time was approximately 2.79 s (about 10.9 ms/call),
including framework synchronization. The two 512px output PNGs retain the
fox's face, legs, tail and snowy backdrop at the same composition; RGB RMSE
0.0242 on [0,1] and correlation 0.9944. This is a visual spot check of **one
prompt and seed**, not a broad aesthetic or instruction-following guarantee.
The source-graph one-block numerical probe with the **real** adapter found
relative base/adapted FFN output errors 0.65%/0.65% on eight sparse synthetic
rows, plus a nonzero gate/up effect and separately checked down-LoRA term.
It does not establish all-layer BF16 parity or validate arbitrary adapters.

The measurements were made after checking for competing AI processes. ComfyUI
was resident but idle (0% CPU); when its download helper became CPU-active,
timing was postponed until the helper exited. No ABBA in-process interleaving
or cross-seed timing distribution has been established yet. An earlier Qwen
Viggle r256, six-step **correct** runtime LoRA measurement was slower than
GPU (0.815×); a later matched crossover was near parity (~1.01× nominal,
not a robust win). Different adapter rank, model kernels, and boundaries
make this Z-Image result non-transferable.

## Further width experiment

The same base-only ABI supports 4096, 6144 and 8192 prefix channels behind
explicit manifest binding. Single-layer CPU/Neural Engine Core ML prediction
API medians with 12 warmed synthetic calls were ~13.7, 20.5 and 31.2 ms,
respectively; the real-adapter one-block numerical probe passed for all three.
Those are **not** whole-image GPU/ANE timings. The extra boundary traffic and
Core ML time make a wider partition uncertain even though the GPU suffix
shrinks. Two warm 6144-channel full-image requests took **9.431/9.371 s**:
slower than the qualified 4096-channel 7.254 s and the GPU-only 8.699 s
medians. No 8192-channel full-image timing is available. Do not promote a
width using only these microbenchmarks.

To select the qualified experimental 4096-channel variant explicitly, put
`"execution": "gpu_ane"`, `"hybrid_mlp_mode": "lora_fused"`,
`"lora_strategy": "inference_time"`, `"allow_approximation": true` and a
matching local `ane_manifest` in a 512×512, eight-step resident request. The
CLI also accepts `--hybrid-mode lora_fused --ane-manifest MANIFEST` with the
same runtime LoRA request. Use `tools/validation/z_image_runtime_lora_layer.py`
on the single-layer source package and the real adapter **before** exporting
all layers for a new width. Keep `gpu` as the conservative quality baseline.
For base inference with the *same* compiled manifest, omit `loras` and use
`hybrid_mlp_mode: "lora_fused"` explicitly with the same 512px/eight-step
resident settings; do not specify a non-auto `lora_strategy` without an
adapter. This shared-artifact base mode is a correctness capability, **not**
a measured speed improvement versus the existing automatic base route.

### Shared-artifact switch check (2026-09-28)

A single resident CLI batch ran base → the local rank-32 LoRA (strength 1)
→ base, always using the **same** compiled 4096-channel manifest. All three
512px/eight-step images completed. The two base PNGs were byte-identical
(SHA-256 `a4cd56b1cf5f317c7264a16264a5a1dd9f6b0703c921991123f8366ea5efd52b`);
the adapter output was different, and all three visibly depict a coherent
fox in snow. This checks that the adapter does not leak into the next base
request. No *second distinct adapter file* is installed locally, so visual
qualification for another adapter is still outstanding. The cold/switch
request timings in this check are not an independent speed benchmark.

An additional matched resident 512px/eight-step check with 238 bound runtime
adapter projections gave two warm GPU requests of **8.627/8.611 s** and two
warm frozen-base GPU/Core ML requests of **7.288/7.294 s**, around **1.18×**
relative to this GPU pair. Cold loads were excluded. ComfyUI was checked
before the runs; a separate integration task became active later, so these
two-point numbers complement rather than supersede the four-run campaign
above. The fastest qualified runtime-LoRA candidate remains the 4096-channel
prefix, and the original base-only automatic route is unchanged.

### Runtime delta allocation check (2026-09-28)

The generic runtime `lora_delta_slice` previously built a full FP32 zero
tensor and added the first low-rank delta to it even when the adapter already
covered the whole requested slice. It now begins with the first real delta;
further adapters still accumulate, while an absent or disjoint adapter
returns zero. A crossing-range/absent-adapter probe and the Qwen native
slice tests pass. The Z-Image runtime-LoRA hybrid output PNG has the same
SHA-256 before and after this change
(`dc8d98d44132a483568dc3f52081b633c700aba356eb7a415eab2b1e2b3f4728`).
Two new resident hybrid batches, each excluding the first cold request,
measured **7.126/7.092 s** and **7.087/7.076 s**; one interleaved GPU batch
measured **8.581/8.584 s** (238 projections bound throughout). The four-run
hybrid median is 7.089 s, or approximately **1.21×** against this GPU pair.
Earlier hybrid runs were 7.288/7.294 s, but scheduling/thermal variation was
not isolated with a same-binary A/B switch: do not attribute the full
difference to this allocation change. Keep the 4096-channel mode opt-in;
the fastest base-only selection has not changed.

The *shared graph* was also run with no adapter at 512px/eight steps: two
warm requests were 5.922/5.904 s. This verifies that it serves base as
well as runtime LoRA, not that it beats the separate fastest base graph.
After the delta change, a single resident base → LoRA → base batch again
returned byte-identical base PNGs (the same `a4cd56b1...` hash above), with
238 projections bound only on the middle request.

### Matching Core ML phase telemetry (2026-09-28)

The same session-cumulative phase counters were differenced over two warm
512px/eight-step runtime-LoRA requests (256 calls/request): feature binding
**2.11/2.24 ms**, synchronous Core ML prediction **2.750/2.792 s**, output
handling **0.77/0.82 ms**, and entire bridge **2.753/2.795 s**. The wall
times were 7.088/7.099 s. Binding and output copies are not the bottleneck:
the fixed-shape graph again reported zero output copies. Per-call prediction
API time is roughly 10.7–10.9 ms here, versus Qwen's ~16.3 ms; model graph,
split width, activation precision and GPU contention all differ, so this
comparison cannot isolate one cause or prove physical ANE placement.

### Optional direct-FP16 runtime LoRA delta (2026-09-28)

With `TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16=1`, the GPU's FP32-accumulated
runtime gate/up LoRA correction rounds directly to the Core ML graph's FP16
activation input. Without this flag, it rounds to the source BF16 activation
dtype first and *then* to FP16. The base graph, adapter weights, GPU FFN
suffix, down-LoRA and non-FFN LoRA are unchanged. This flag is **opt-in**:
the original matched-accuracy runtime route and base-only automatic policy
remain unchanged. It is a small lossy numerical change, not LoRA omission.

One same-binary off → on → on → off resident-batch screen, each with one
excluded cold request and two warm 512px/eight-step requests, recorded
whole-request times of **7.147/7.098 and 7.102/7.096 s off**, versus
**7.060/7.067 and 7.056/7.062 s on**. Four-run medians are ~7.100 and
~7.061 s, roughly **0.6% faster**. All requests bound 238 adapter
projections. The opt-in PNG preserved the fox pose, face, legs and snow
visually; versus the original hybrid PNG its RGB [0,1] RMSE was 0.0110,
correlation 0.99885. Against the same pure-GPU output, RMSE increased from
0.02424 to 0.02604. This is one prompt/seed and a small timing difference:
do not infer a general image-quality guarantee or a larger speedup.

After the final native rebuild, an additional matching 512px fox pair yielded
two warm pure-GPU wall times **8.579/8.601 s** and two direct-FP16 hybrid wall
times **7.067/7.048 s** (first cold request excluded in each resident batch):
median ratio **1.217×**. Both bound 238 projections; the hybrid made 256
Core ML calls per request and reproduced the expected direct-FP16 PNG hash.
This two-pair check confirms the optimized route still beats GPU in the final
binary, but is not a cross-prompt timing distribution.

A second 512px/eight-step request used a yellow-flower glass vase in a kitchen,
seed 17. GPU, original shared-base hybrid and direct-FP16 hybrid all bound
238 adapter projections and produced recognizably matching vase, flowers,
window and tabletop; the three images were inspected at native size. The
direct-FP16 image differed from the original hybrid by RGB RMSE **0.0195**
and correlation **0.99766**. Versus GPU, the original hybrid's RMSE was
**0.02891** and direct FP16's was **0.02980**. These single cold requests
verify another visual case, **not** cross-prompt speed or a universal quality
threshold. Keep the opt-in flag and the unmodified default boundary.

### Different adapter-file switching check (2026-09-28)

`tools/validation/runtime_lora_shared_graph_switch.py` derives a **synthetic**
second adapter by halving all 238 LoRA-B tensors in a temporary copy of the
real local adapter; it does not present the copy as a trained or quality-
qualified model. Two resident batches, one with the default boundary and one
with direct FP16 enabled, ran base → original LoRA → synthetic LoRA → base
against the *same* compiled base manifest. Both reported cumulative Core ML
calls 256/512/768/1024 without reloading the graph, LoRA bindings
0/238/238/0, distinct PNG hashes for the two adapters, and identical first
and last base PNG hash
`a4cd56b1cf5f317c7264a16264a5a1dd9f6b0703c921991123f8366ea5efd52b`.
The scratch adapter and outputs were deleted with the test-owned temporary
directory. This exercises the second-file identity path, but **does not**
qualify image quality or schedule for a second genuine trained adapter.
