# Z-Image 512² W8A16/W8A8 FFN research status

This is an experimental branch with a matched end-to-end speed gain, **not a
qualified production default**. The goal is a GPU W8A16 suffix in parallel with a Core ML W8A8
prefix for each gated FFN. All reported images use the original resident BF16
Z-Image-Turbo transformer as the reference, 512×512 and 8 denoising steps.

## Failure mode and experiments

The 1056-token transformer input is 1024 image tokens followed by 32 caption
tokens. Caption rows 1025–1026 are extreme activation outliers. Uniformly
sampling calibration rows misses them. A single hidden A8 scale cannot preserve
both the ordinary image rows and these large caption activations. A misleading
whole-tensor relative L2 can be low while the image rows are nearly zero.

`export_z_image.py --outlier-rows --region-image-rows 1024` calibrates both
regions independently at the input and SwiGLU-hidden A8 boundaries, then
concatenates them before each shared W8 projection. Add
`--sq-hidden-image-only` to derive the hidden SmoothQuant scale from image
rows; the caption region retains its own A8 scale. The graph gate verifies
that **both** regions of **both** projections traverse dynamic A8 Q/DQ and
compressed W8 weights. This graph check does **not** prove physical ANE
execution.

On a single held-out block-18 capture, the old artifact had image/caption
relative L2 of `0.283/0.998`. Two-region A8 gave `0.184/0.018`; image-only
hidden SmoothQuant gave `0.111/0.033`. Block 17 and 29 image rows also
improved, but local FFN error did not predict end-to-end image fidelity.

| Matched 8-step path (seed 42) | Image PSNR vs BF16 GPU | Warm denoise |
| --- | ---: | ---: |
| BF16 GPU | identical | 6.831 s |
| Full GPU W8 suffix + two-region ANE W8A8, 32 layers | 14.01 dB | 6.796 s |
| Full GPU W8 suffix + image-only SQ/two-region ANE W8A8, 32 layers | 15.42 dB | 6.802 s |
| GPU W8 disabled, ANE W8A8 only on blocks 0–1 | 36.22 dB | 6.820 s |
| GPU W8 suffix + ANE W8A8 on blocks 0–1; BF16 GPU blocks 2–31 | 32.13 dB | 6.878 s |

The last path scored `30.63 dB` and `6.870 s` on an independent astronaut
prompt/seed 123; its matched BF16 GPU took `6.825 s`. Each number above is
one warmed request, **not** an ABBA timing qualification. The 32-layer paths
are visibly degraded; the selective route approaches image fidelity but is
not faster than the GPU reference and uses BF16 GPU for 30 of 32 blocks.

## GPU W8 suffix latency diagnosis

`tools/native/benchmark_z_image_w8_suffix.py` now accepts original BF16
checkpoint suffix slices and a held-out FFN input. It preserves BF16 weight
bits during loading (the Core ML exporter's tensor reader rounds BF16 to FP16),
then follows the native path's FP16 affine W8 packing and BF16 activation.
The captured FFN input was stored in FP16, so it is an approximation to the
uncaptured BF16 runtime input. On the M4 Max, warmed/synchronized medians in
milliseconds, one 1056×3840×6144 suffix, 4096-prefix split:

| Held-out capture / group | BF16 suffix | Direct resident W8 suffix | GPU W8 image-region relative L2 |
| --- | ---: | ---: | ---: |
| Block 18 / g32 (48 samples/mode) | 10.85 | 13.41 | 0.00574 |
| Block 18 / g128 (48 samples/mode) | 10.86 | 13.22 | 0.00691 |
| Block 29 / g32 (32 samples/mode) | 10.83 | 13.42 | 0.00549 |
| Block 29 / g128 (32 samples/mode) | 10.94 | 13.22 | 0.00680 |

For block 18/g32, the separate gate/up pair takes 7.12 ms BF16 vs 8.73 ms
W8; the down projection takes 3.80 vs 4.53 ms. The packed g32 suffix occupies
79.6 MB versus 141.6 MB BF16. Dequantizing W8 **once before timing** and
retaining dense BF16 or FP16 weights yields about 10.8 ms, but excludes the
preparation cost and forfeits W8 residence; it is **not** a valid accelerated
W8 suffix. On block 29/g128 the whole suffix relative L2 is 0.00651 versus
0.00516 for g32 on the same capture. Larger groups barely improve this kernel's
latency and can worsen quality; the current direct W8 kernel is still slower
than BF16 in isolation. These are GPU-only timings with no ANE concurrency or
end-to-end quality qualification.

As a separate resident-W8 candidate, concatenating the packed gate/up
matrices into one quantized matmul on block 18/g32 gave 13.41 ms for the
suffix versus 13.46 ms for two W8 calls in the same 32-sample ABBA run;
outputs were identical. This small isolated difference does not address the
roughly 2.5 ms gap to BF16 and has not been promoted to the native runtime.
The alternative W8 `mxfp8` format (block 18/g32, 32 samples/mode) reached
12.18 ms versus 10.82 ms BF16, while image-region relative L2 jumped to
0.0795. It is neither fast enough nor an accuracy-preserving replacement for
the tested affine W8 path; no runtime change was made.

An FP32 activation-only screen with four independently calibrated hidden-A8
image-row tiles was mixed: on held-out block 18 rows, matched tile samples
improved relative L2 from 0.126 to 0.110; block 29 *regressed* from 0.081
to 0.094. This is not an exported Core ML graph or an image-level test, and
spatial tiling is not a uniform fix. Reproduce using
`tools/validation/z_image_w8a8_layers.py --image-rows 1024 --image-tiles 4
--blocks 18,29` with the above manifest, BF16 model, clean calibration
captures, and held-out captures.

### Channel-group A8 experiment

The optional `export_z_image.py --hidden-a8-channel-groups 4` keeps the first
and down convolutions W8 and both activation boundaries A8. It splits only
the 1024 image rows into four 1024-channel hidden-A8 groups and leaves the
caption rows on their existing A8 scale. `verify_w8a8` checks every nested
Q/DQ branch. Each block's calibration source, group count and static scales
are bound into the immutable export identity/receipts. The original exporter
and native default are unchanged.

On the same held-out inputs, CPU-only Core ML block-18 image-row relative L2
fell from `0.11075` to `0.08281`, and block 29 from `0.05731` to `0.05516`;
caption-row errors were unchanged. The CPU_AND_NE block-18 ABBA medians were
`14.34/14.34 ms` (original/grouped) for 16 warmed predictions per arm; all
recorded operators still **prefer** ANE in the compute plan. Neither a compute
plan nor `CPU_AND_NE` is a physical device trace, and the Python prediction
timings are not the native FFN critical path.

The complete 32-layer grouped export is at
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_all/manifest.json`;
the compiled manifest is under
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_compiled/`. On a
matched 512²/8-step fox prompt (seed 42), full-layer PSNR against BF16 GPU
rose only `15.42 -> 15.50 dB`, still visibly unacceptable. Its single warmed
denoise was `6.804 s` against BF16 GPU `6.821 s`; this is **not** qualified
speedup. The Core ML 512-call total was `4.738 s`. The grouped A8 graph is
therefore a diagnostic improvement, not a successful image-quality solution.

An output-aware 256-channel routing screen in
`tools/validation/z_image_w8a8_routing.py` chooses candidate ANE groups using
calibration rows only, then recomputes SQ/A8 scales and measures separate
held-out rows. Normalized to the full BF16 FFN output, block 3's A8 error
improved `0.0131 -> 0.0030`, and block 18's `0.0605 -> 0.0396`, but block
29 regressed `0.0207 -> 0.0222`, block 1 `0.0203 -> 0.0386`, and block 16
`0.0374 -> 0.0559`. This **FP32 A8-only screen** does not test GPU W8,
Core ML W8, noncontiguous weight reordering, or image quality. No routing was
installed in the runtime; an all-layer route would be unjustified.

A finer, *contiguous* hidden-A8 split is also not uniformly better. With 16
image-channel groups instead of four, an FP32 A8-only held-out screen improved
block 1's partial-FFN relative L2 from `0.0342` to `0.0254`, but worsened
blocks 16 (`0.0892 -> 0.0935`) and 29 (`0.0581 -> 0.0632`); block 18 was
essentially unchanged (`0.0596 -> 0.0600`). No 16-group graph was exported.

`screen_joint_hidden_channel_groups` is another **diagnostic only**: it
coordinate-selects the four group scales on calibration rows to minimize the
*combined* partial-FFN error, including input-A8 error, rather than minimizing
each group's error independently. Its much smaller training loss did not
generalize. On the separate two-sample/16-image-rows-per-sample held-out set,
the independently selected versus joint scales produced partial-FFN relative
L2 of `0.0342/0.0256` (block 1), `0.0141/0.0205` (3), `0.0892/0.1308`
(16), `0.0596/0.1301` (18), and `0.0581/0.0648` (29). Neither joint tuning
nor finer grouping is safe to apply to all 32 layers. Reproduce the joint
screen with `tools/validation/z_image_w8a8_layers.py --image-rows 1024
--hidden-channel-groups 4 --joint-hidden-channel-groups 4 --blocks
1,3,16,18,29` plus `--manifest` pointing to the grouped source manifest,
`--model models/Comfy-Org-z_image_turbo`, `--calibration-dir
outputs/z-w8a8-calibration-clean/capture`, and `--heldout-dir
outputs/z-w8a8-heldout/capture`. The output reports the training and held-out
errors separately, and makes no Core ML, W8, physical ANE, or image-quality
claim.
On all four held-out samples at 32 image rows each, independently calibrated
versus joint scales measured `0.0579/0.0516` (block 1), `0.0185/0.0289` (3),
`0.0889/0.1213` (16), `0.0836/0.1606` (18), and `0.0464/0.0498` (29).
The wider check strengthens the conclusion that joint selection overfits;
it does not qualify even the block-1 improvement for an end-to-end route.

For GPU performance, `w8_dequant_each` in the suffix benchmark retains packed
W8 matrices and measures FP16-affine-W8 expansion **inside each invocation**
before BF16 GEMM: block 18/g32 took `11.92 ms` versus `13.41 ms` direct W8 and
`10.90 ms` resident BF16. Unlike pre-dequantized diagnostics, the expanded
weights are expression-local, but transient BF16 memory and repeated expansion
remain. `TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM=bf16` selects this experimental
runtime branch and labels its precision `+on_demand_dequant_bf16_gemm`.
On the 32-layer grouped-A8 image, two warm denoises took `5.910/5.904 s`
versus `6.804 s` with direct GPU W8; MLX peak bytes remained `11.035 GB`.
But PSNR against BF16 fell from `15.50` to `13.79 dB`, so **this is not a
usable speedup**. Limiting hybrid W8A8 to blocks 0–1 (BF16 GPU for the other
30) gave only `28.69 dB` at `6.841 s`, again worse than the original
direct-W8 selective result (`32.13 dB` at `6.878 s`) and not faster than the
BF16 GPU reference. A tempting FP16 on-demand expansion took `11.46 ms` in
an isolated suffix benchmark, but a complete 32-layer image failed with a
nonfinite latent; the native FP16 mode was removed. None of these single
requests constitute a warm ABBA speedup qualification.

A repeated held-out block-18/g32 suffix check (eight measurements per mode)
found `13.43 ms` for direct resident W8 versus `11.95 ms` for on-demand BF16
expansion, but image-row suffix relative L2 versus BF16 grew from `0.00574`
to `0.00608`. The on-demand result is **not** numerically interchangeable
with the direct W8 kernel, even on one fixed input; the end-to-end PSNR loss
must not be attributed solely to A8 calibration. This isolated check does not
establish which individual rounding step causes the image-level divergence.

Another opt-in GPU experiment,
`TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM=gate_up_fp16`, expands **only** gate/up
W8 weights inside each invocation and performs those GEMMs in FP16. Their
outputs round back to BF16 before SwiGLU; the down projection keeps the
original direct W8 path. The packed W8 weights remain resident, and this is
not the removed, overflowing all-FP16-FFN path. Held-out g32 suffix medians
were `12.20 ms` (block 18) and `12.23 ms` (29), versus `13.41/13.37 ms`
direct W8 and `10.85/10.88 ms` BF16, each with 16 warmed measurements. The
image-row relative L2 was `0.00572/0.00560` versus `0.00574/0.00549` for
direct W8: a speed/accuracy tradeoff, not uniform improvement.

The grouped-A8 32-layer image completed without nonfinite values, but its
fox/seed-42 PSNR was `15.52 dB` (direct W8 `15.50 dB`), still unacceptable;
its one warm denoise was `6.552 s`, not a qualified ABBA speedup. Limiting
hybrid FFN to blocks 0–1 and keeping BF16 GPU for blocks 2–31 achieved
`39.76 dB` on fox/42 (`6.854 s` warm denoise) and `32.97 dB` on the
independent astronaut/123 (`6.852 s`). On the *same grouped graph* and fox/42,
direct-W8 first-two-blocks gave `37.36 dB / 6.878 s`. The separate BF16 GPU
anchors were approximately `6.82–6.83 s`; the selective variant is **not
faster**, and retains 30 complete BF16 GPU blocks. Extending it through
block 7 fell to `22.62 dB` on fox/42 (`6.801 s`). None of these are a valid
full-model W8A16/W8A8 speedup. The generated PNGs and reports are under
`outputs/z-w8a8-hidden-c4-gate-up-fp16-*`; physical ANE placement remains
unverified.

### Complete 8-step calibration and channel routing

The earlier A8 calibration captured only a two-step fox request. The new
training directory `outputs/z-w8a8-calibration-8step-diverse/capture` has 16
FFN inputs per block: all eight steps for fox/42 and astronaut/123. The
independent lighthouse/187 holdout has eight per block in
`outputs/z-w8a8-heldout-8step-lighthouse/capture`. The benchmark's constrained
`--capture-once --runs 0` mode makes a single GPU BF16 calibration request;
neither captures nor the subsequent one-request measurements qualify speed.

The 32-block, contiguous 4096-channel, four-hidden-group graph exported from
this training set is under
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_diverse8_{all,compiled}/`.
Against the fox/42 BF16 image, full-layer direct GPU W8 scored `16.19 dB`
at `6.789 s` warm denoise; on-demand gate/up FP16 with direct W8 down scored
`13.94 dB` at `6.544 s`. Better timestep coverage did not make either
full-model result acceptable. Against the *independent* lighthouse/187 BF16
image, full-layer direct W8 scored only `13.17 dB` at `6.786 s` versus a
`6.824 s` BF16 denoise. All timings here are single warm requests.

The experimental 4096-channel route picks ANE 256-channel groups on the
training set, reorders gate/up rows and down columns in the exporter, and
packs the exact sorted GPU complement as resident W8 in the native runtime.
The manifest binds the route to the checkpoint and each block's calibration
digest. The original contiguous partition remains the default. In an A8-only
holdout screen the route improved 25 of 32 blocks, yet the fox/42 image fell
to `15.52 dB` at `6.814 s`, worse than the new contiguous split. A smaller
2048-channel route reached `16.91 dB` but took `8.012 s`. The source and
compiled artifacts are under
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_routed{,_a2048}_diverse8_{all,compiled}/`;
their route records are in `outputs/z-w8a8-diverse8-channel-route*.json`.

Combining the 4096 route with gate/up FP16 and returning the 12 blocks with
the largest local A8 errors to full BF16 GPU yielded only `21.57 dB` at
`6.636 s`. Returning the worst 20 blocks yielded `21.75 dB` at `6.756 s`.
Thus local A8 ranking is not a reliable image-sensitivity ranking. A separate
contiguous-graph test left only blocks 6–12 on hybrid W8A8 and put the other
25 blocks on BF16 GPU: fox/42 reached `29.32 dB` at `6.815 s`, effectively
the approximately `6.82 s` BF16-GPU denoise time. On the independent
lighthouse/187 request this same selection achieved only `18.29 dB` at
`6.825 s`, versus `6.824 s` BF16 GPU. Thus even this selective quality gain
does not generalize; it neither meets full-model quantization nor demonstrates
a speedup. Its manifests, images and machine-recorded metrics are under
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_diverse8_compiled/`
and `outputs/z-w8a8-diverse8-gate-up-fp16-only6to12-{fox42,lighthouse187}/`.

On lighthouse held-out image rows, an FP32 *activation-only* diagnostic
attributed the dominant block-20 error to hidden A8: input-only versus
hidden-only relative L2 was `0.0281/0.1694`. With four held-out samples and
32 sampled image rows each, its combined error was `0.1719` with one hidden
scale, `0.1726` with four contiguous scales, and `0.1737` with eight. Block 6
was `0.0161/0.0164/0.0178` for one/four/eight scales; block 12 was
`0.0371/0.0385/0.0368`. Finer grouping is not a consistent fix on this
independent prompt and has **not** been exported. Input grouping and sorting
channels by their training hidden range also failed to give consistent held-out
improvements. These simulations exclude W8, Core ML rounding, physical ANE
execution and accumulated diffusion error.

Reproduce the four/eight-group A8-only holdout with
`tools/validation/z_image_w8a8_layers.py --manifest
models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_diverse8_all/manifest.json
--model models/Comfy-Org-z_image_turbo --calibration-dir
outputs/z-w8a8-calibration-8step-diverse/capture --heldout-dir
outputs/z-w8a8-heldout-8step-lighthouse/capture --image-rows 1024 --blocks
6,12,20 --samples-per-block 4 --rows-per-sample 32
--hidden-channel-groups 4`; replace the final 4 with 8 for the other screen.
This reruns an FP32 simulation, not a compiled-model or image test.

Finally, an isolated block-18 GPU suffix experiment with per-invocation FP16
dequantization of all three resident W8 projections and a scaled hidden/down
product ran around `11.84 ms`, but image-row relative L2 against BF16 was
`0.00593` versus `0.00574` for direct W8. This research-only mode is neither
in the native runtime nor validated on a complete image; the GPU W8 suffix
speed/quality issue remains open. Core ML `CPU_AND_NE` selection and ANE
operator preference in a compute plan are not physical placement evidence.

### Adaptive hidden-A8 scale bank (experimental, not qualified)

On independent lighthouse/187 captured image rows, a **non-deployable**
per-token max-derived hidden scale reduced block-20 FP32 A8-only partial-FFN
relative L2 from `0.1719` to `0.0571`. With eight *training-fixed* FP16 A8
scales chosen per token by its hidden peak, that held-out error was `0.0824`;
blocks 6/12/18/29 changed respectively from `0.0161/0.0371/0.0966/0.0678`
to `0.0133/0.0247/0.0607/0.0491`. A four-scale bank was mixed, and a
32-group static spatial scale was unstable: block 20 reached `0.1389` with
maximum-based scales but `0.1856` with training output-error thresholds. The
row-group scheme was not exported. Reproduce the eight-scale FP32 diagnostic
with `z_image_w8a8_layers.py` and the diverse8 manifest/training/lighthouse
directories shown above, plus `--image-rows 1024 --blocks 6,12,18,20,29
--samples-per-block 4 --rows-per-sample 32
--diagnose-adaptive-hidden-a8-bins 8`. These numbers omit Core ML, W8, and
diffusion feedback.

The opt-in exporter flags `--adaptive-hidden-a8-bins 8
--adaptive-hidden-a8-shared-qdq` build a real fixed-scale-bank graph. It
computes a per-image-token peak, selects a fixed FP16 multiplier and its
inverse, rescales the hidden activation around **one** constant-scale A8
Q/DQ, and retains the down-projection's compressed W8 weights. The caption
region keeps the old scale. The W8A8 graph gate accepts the restore multiply
only when its activation input comes through Q/DQ and its other input selects
only constants; tests reject bypassing either guard. This validates the MIL
path, **not** physical ANE execution or semantic equality of the scale bank.

An initial scalar-candidate graph silently lost the per-token selector shape
in Core ML: its selected ratio was shape `[1]`, yielding block-20 Core ML
image-row relative L2 `0.1923`. Explicit constant vectors of shape
`[1,1,1,1024]` restored the intended selection: the post-A8 activation
differs from the eight-Q/DQ-branch prototype by only `0.00442` relative L2.
On the *same* lighthouse block-20 capture, the original grouped graph versus
shared-Q/DQ graph scored `0.14409 -> 0.07652` Core ML image-row relative L2.
In a warm `CPU_AND_NE` single-block ABBA probe, medians were `14.35/14.61 ms`
(12 rounds); the eight-Q/DQ prototype took `18.25 ms` in a separate eight-round
probe. All selected operators **prefer** ANE in the compute plan, which is not
a physical trace or a mixed GPU/ANE speedup qualification.

The complete 32-block adaptive source and compiled manifests are under
`models/coreml/z_image_w8a8_adaptive8_shared_vector_diverse8_{all,compiled}/`.
In matched 512²/8-step direct-GPU-W8 tests, fox/42 scored only `15.58 dB /
6.838 s` warm denoise, *worse* than the contiguous diverse8 graph's
`16.19 dB / 6.789 s`; the independent lighthouse/187 gave `12.94 dB /
6.814 s` versus `13.17 dB / 6.786 s`. The coreml graph's local gain thus
**does not improve end-to-end quality**. A diagnostic with the same A8 graph
but BF16 GPU complement reached only `16.15 dB` on fox/42, though its one
warm denoise was `5.612 s`: A8 still dominates image failure, while the
present direct GPU W8 path adds substantial latency. The latter diagnostic
is *not* a W8A16 solution. Reports/PNGs are under
`outputs/z-w8a8-adaptive8-shared-vector-{direct-w8-fox42,direct-w8-lighthouse187,gpu-bf16-fox42}/`.
All complete-image times here are single warm requests, not an ABBA speedup
gate. No adaptive artifact is recommended for production.

### Third-prompt calibration screen (no new export)

A separate BF16 512²/8-step vintage-car/571 capture contributes eight training
samples per block under `outputs/z-w8a8-calibration-8step-city/capture`.
`--extra-calibration-dir` now adds this to the original fox/42 and
astronaut/123 training set without copying the captured arrays. The exporter
and layer diagnostic share an order/content-bound union digest; the old
single-directory digest and export identity remain unchanged. The lighthouse/187
capture is *only* held out. A diagnostic against the existing one-directory
manifest must use `--allow-unexported-calibration`, because the recalibrated
weights and A8 scales have not been exported.

On four independent lighthouse samples per block (32 selected image rows each),
the FP32 A8-only shared-Q/DQ partial-FFN relative L2 was:

| Block | Existing 16-step training | Plus car/571 (24 steps) |
| --- | ---: | ---: |
| 6 | 0.01336 | 0.01222 |
| 12 | 0.02471 | 0.02357 |
| 18 | 0.06074 | 0.06667 |
| 20 | 0.08238 | 0.09719 |
| 29 | 0.04908 | 0.04619 |

The worst two layers regress, especially block 20. This does not establish
that three-prompt calibration would make an *image* worse; the existing
two-prompt artifact already failed semantic fidelity. There is insufficient
held-out evidence to spend another complete export/compile/image-validation
cycle on this particular union, so no 32-layer three-prompt artifact was
created. Reproduce the candidate using the diagnostic command described above
with `--extra-calibration-dir
outputs/z-w8a8-calibration-8step-city/capture
--allow-unexported-calibration --diagnose-adaptive-shared-qdq`; omit the extra
directory and unexported flag for the existing-artifact control. The layer
tool rejects a changed directory count without the unexported-screen flag.

### Resident W8 with on-demand GPU expansion: measured speed, open quality gate

The GPU complement still holds packed 8-bit weights. In
`TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM=bf16`, each invocation expands the
three W8 projections *inside* the expression and immediately uses BF16 GEMM;
there is no resident dense-weight cache or change to the production default.
The runtime reports
`gpu_w8a16+coreml_w8a8+on_demand_dequant_bf16_gemm`. On the complete
adaptive graph, the peak MLX allocation reported for a warmed fox request was
`11,035,188,256` bytes. This validates packed-weight residency at the level
of the runtime implementation and telemetry, not a hardware INT8 arithmetic
claim. It incurs transient BF16 expansion every FFN call.

Two matched 512²/8-step, fresh-process `baseline,fused,fused,baseline` runs,
each with one excluded cold request and two warmed requests per process, used
the **same** native library hash and production-default GPU control:

| Prompt/seed | BF16 GPU warm denoise median | W8-resident hybrid median | Ratio | Warm samples/arm |
| --- | ---: | ---: | ---: | ---: |
| fox/42 | 6.824 s | 5.911 s | 1.154× | 4 |
| lighthouse/187 (independent) | 6.827 s | 5.909 s | 1.155× | 4 |

The matched wall ratios were `1.148×` and `1.150×`, respectively. Both
BF16-anchor system-state checks passed. The complete records are
`outputs/z-w8a8-adaptive8-shared-vector-on-demand-bf16-abba-{fox42,lighthouse187}/summary.json`.
Unlike the earlier grouped-A8 on-demand BF16 result, **this** graph sustained
the speed gain on two prompts. It does not prove ANE physical residency.

Full-image same-prompt/seed PSNR remains low: fox/42 `15.64 dB`,
lighthouse/187 `12.87 dB`, and a separate held-out watercolor tram/314
`13.01 dB`. Human visual inspection of the PNG pairs found recognizable,
clear fox and tram images with the key colors/styles, but markedly different
composition; lighthouse loses the crucial *glass hand-painted* interpretation
of the BF16 image and instead depicts an ordinary coastal lighthouse. Thus
neither PSNR alone nor three good-looking samples establish the design plan's
multi-prompt semantic/detail quality gate or an acceptable production default.
The lighthouse images and reports are under
`outputs/z-w8a8-adaptive8-shared-vector-on-demand-bf16-lighthouse187/` and
the tram comparison under `outputs/z-w8a8-{gpu-baseline-tram314,adaptive8-shared-vector-on-demand-bf16-tram314}/`.

The alternative opt-in `scaled_all_fp16` performs FP16 GEMMs from packed W8
with a `64×`-scaled hidden/down product to avoid the previous overflow. On
fox and lighthouse it generated finite images; separate matched ABBA denoise
ratios were `1.053×` on both prompts (four warm samples/arm), less than the
on-demand BF16 mode. Fox PSNR fell to `13.87 dB`. The CPU_AND_NE compute plan
still only reports ANE **preference**. The `xctrace` binary on this machine
cannot run with the selected Command Line Tools developer directory (it
requires full Xcode), and `powermetrics -s ane_power` requires superuser;
noninteractive sudo has no authorization. Physical GPU/ANE concurrency and
ANE residency have therefore **not** been established.

A 4096-prefix block-18/g32 held-out suffix benchmark tested concatenating
the packed gate/up pair before expression-local BF16 expansion. Its warmed
suffix median was `12.10 ms` versus `11.90 ms` for separate gate/up BF16
expansion; output error was unchanged. It has **not** been installed in the
native path. The tested isolated candidate does not improve the remaining
GPU W8 speed gap.

Returning only the first two refiner blocks to complete BF16 GPU while
keeping the other 30 blocks on the adaptive W8-resident/ANE-W8A8 hybrid did
not repair the semantic gap: fox/42 PSNR was `16.38 dB` at `5.988 s` warm
denoise, lighthouse/187 `12.90 dB` at `5.993 s`. These are one-request
diagnostics, not ABBA-qualified speeds, and they no longer quantize all 32
FFNs. Their reports and images are under
`outputs/z-w8a8-adaptive8-on-demand-bf16-fallback01-{fox42,lighthouse187}/`.

Reproduce the fast matched comparison with
`tools/native/benchmark_z_image_metal.py --model
models/Comfy-Org-z_image_turbo --hybrid-manifest
models/coreml/z_image_w8a8_adaptive8_shared_vector_diverse8_compiled/manifest-e346b1ec846a09587e54b44b74367125c13a5a4839a12a391ef5d4596d57bc90.json
--hybrid-w8-mode bf16 --control-default --runs 2 --output <new-output-dir>`;
add the lighthouse prompt and `--seed 187` for the second schedule. Do not
set ambient `TURBOCIDER_Z_*` overrides in schedule mode. No release should
infer acceptable quality or hardware placement from the speed ratio alone.

Reproduce the first row with `.venv/bin/python
tools/native/benchmark_z_image_w8_suffix.py --model
models/Comfy-Org-z_image_turbo --capture
outputs/z-w8a8-heldout/capture/block18/sample-9685-115992559947125-18.npy
--block 18 --group 32 --repetitions 24 --warmup 6`. Change to `--block 29`,
the block-29 capture, and/or `--group 128` for the other rows. Use
`--quant-mode mxfp8` for the rejected alternative. This benchmark
does not yet profile GPU hardware counters or test a faster resident W8 kernel.

### Image-only hidden SQ and caption scale-bank ablations

The prior shared-Q/DQ graph used `sq_alpha2=0.5` for all blocks and both token
regions. A new training-only search splits the fox/42 and astronaut/123
captures by recording process, fits each fold independently, and scores the
other fold. It never chooses an alpha using lighthouse. Across 32 blocks, the
minimum training cross-validation alpha improved only 10 blocks on the
independent lighthouse A8-only screen, regressed 16, and left six unchanged.
For example, block 20 favored `0.25` in both training folds and lighthouse,
but block 8 regressed by about 60% relative on lighthouse. A full per-block
alpha map is therefore not justified by this screen. Reproduce with
`tools/validation/z_image_w8a8_sq_search.py --manifest
models/coreml/z_image_w8a8_adaptive8_shared_vector_diverse8_all/manifest.json
--model models/Comfy-Org-z_image_turbo --calibration-dir
outputs/z-w8a8-calibration-8step-diverse/capture --heldout-dir
outputs/z-w8a8-heldout-8step-lighthouse/capture`.

Changing block 20's *shared W8 weights* to `alpha2=0.25` reduced Core ML
image-row relative L2 on four lighthouse capture steps from about
`0.073–0.079` to `0.063–0.065`, but increased caption error from
`0.054–0.056` to `0.088–0.093`. To preserve caption and weight quantization,
`--image-hidden-sq-alpha2 0.25` instead rescales **image hidden activations
only** around their A8 boundary, with a fixed per-channel inverse after Q/DQ;
the preexisting `0.5` W8 projections and caption path remain unchanged.
`--image-hidden-sq-blocks` selects an exported subset; the production/default
path is unaffected. The graph gate accepts the inverse only if the activation
data path traverses Q/DQ and the second operand is constant. It cannot prove
physical ANE execution.

The real Core ML block-20 image-row error then became `0.064–0.066` across
the four independent lighthouse steps, while caption error matched the
original graph exactly. On block 1, image-row error on the last two of those
steps changed `0.0759→0.0242` and `0.0966→0.0196`. Blocks 2 and 3 also
improved image rows on all four steps; block 23 improved mostly on later
steps. These are local partial-FFN results, not full-model quality. Warm
`CPU_AND_NE` block-20 medians were `14.61→14.85 ms` (12-round ABBA); the
compute plan prefers ANE for both, which is not a physical trace.

With the GPU complement still packed W8 and expanded for BF16 GEMM per
invocation, a complete graph changing **only** blocks 1, 2, 3, 20, 23 scored
the following same-seed PSNR against the BF16 GPU image:

| 512²/8-step prompt | Original adaptive graph | Image-only five-block graph |
| --- | ---: | ---: |
| fox/42 | 15.64 dB | 16.15 dB |
| lighthouse/187 | 12.87 dB | 12.97 dB |
| tram/314 (independent) | 13.01 dB | 12.97 dB |

Matched fresh-process `baseline,fused,fused,baseline`, two warmed requests
per process (four warm samples per arm), retained the earlier speed on the
five-image-block graph: fox/42 BF16 GPU `6.825 s` versus hybrid `5.916 s`
denoise (`1.154×`); lighthouse/187 `6.827 s` versus `5.910 s` (`1.155×`).
Both baseline health anchors were valid; median wall ratios were `1.148×`
and `1.149×`, with the same native library hash
`3f447bcdb709bbbc2bd042213aecb5212676306c09dceb61c948217aae0a7e35`.
Full records are under
`outputs/z-w8a8-adaptive8-shared-vector-image-a025-1_2_3_20_23-abba-{fox42,lighthouse187}/summary.json`.
This qualifies a matched speed difference, **not** physical ANE placement.

The independent tram remains a clear yellow tram in a rainy watercolor
street, but its same-seed pixel metric slips slightly. The images remain
recognizable and clear, but the lighthouse still loses the
reference's hand-painted glass structure. These are one warmed request each,
not a matched ABBA speed claim or a passed semantic gate. Artifacts are under
`models/coreml/z_image_w8a8_adaptive8_shared_vector_diverse8_image_a025_1_2_3_20_23_{all,compiled}/`
and images/reports under
`outputs/z-w8a8-adaptive8-shared-vector-image-a025-1_2_3_20_23-{fox42,lighthouse187,tram314}/`.

A separate opt-in `--adaptive-caption-a8-bins 8` adds the same one-Q/DQ,
training-fixed per-token scale selection to caption hidden activations;
`--adaptive-caption-a8-blocks` scopes its blocks. On held-out block 2 and 3,
FP32 A8-only caption partial-FFN relative L2 changed `0.455→0.329` and
`0.269→0.144`. The *actual* Core ML block 3 improved only from about
`0.27–0.29` to `0.17–0.18` on four samples; block 2 barely improved from
`0.50` to `0.49`, implicating additional W8 or FP16 error. An isolated
block-3 12-round `CPU_AND_NE` ABBA gave `14.52→14.40 ms` medians, but again
only ANE preference rather than physical residency.

Adding the caption bank on block 3 to the five-image-block graph **reduced**
fox PSNR `16.15→15.81 dB` and lighthouse `12.97→12.62 dB`. The lighthouse
still looked like an ordinary lighthouse. Thus better held-out local caption
L2 neither repaired the semantic failure nor improved whole-image fidelity.
This bank remains experimental and is not enabled by default. The composite
artifacts and one-request images are under
`models/coreml/z_image_w8a8_adaptive8_shared_vector_diverse8_image_a025_1_2_3_20_23_caption8_b3_{all,compiled}/`
and `outputs/z-w8a8-adaptive8-shared-vector-image-a025-1_2_3_20_23-caption8-b3-{fox42,lighthouse187}/`.

### Routed ANE-width tradeoff with packed W8 and on-demand BF16 GEMM

Returning only unified blocks 2 and 3 to complete BF16 GPU on the five-image-
block graph still produced an ordinary lighthouse. It incurred roughly
`6.21 s` warm wall on a single lighthouse request; the work was not
ABBA-qualified. This weakens the hypothesis that those two high caption A8
errors alone explain the missing glass semantics.

Existing training-only 2048- and 4096-channel routed four-hidden-group
artifacts were remeasured with the packed-W8/on-demand-BF16 GPU complement.
The 2048 route restored the *glass lighthouse* interpretation, but its one
warm lighthouse wall request took about `7.19 s`, above the BF16 GPU wall
near `7.07 s`. The 4096 route also restored the glass structure, suggesting
that channel *choice*, not only ANE width, is important. A new 3072-channel
route selects 12 aligned 256-channel ANE groups per block from the original
fox/astronaut training captures. The independent lighthouse was only used to
score the route, never to choose groups. Its route JSON is
`outputs/z-w8a8-diverse8-channel-route-a3072.json`; all 32 source/compiled
graphs are under
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_routed_a3072_diverse8_{all,compiled}/`.

Matched 512²/8-step original-BF16 GPU versus routed hybrid, fresh-process
ABBA with four warmed requests per arm and healthy M4 Max anchors:

| ANE route / prompt | BF16 denoise | Hybrid denoise | Ratio | Wall ratio |
| --- | ---: | ---: | ---: | ---: |
| 3072 / fox/42 | 6.827 s | 6.437 s | 1.061× | 1.059× |
| 3072 / lighthouse/187 | 6.823 s | 6.443 s | 1.059× | 1.057× |
| 4096 / fox/42 | 6.827 s | 5.909 s | 1.155× | 1.149× |
| 4096 / lighthouse/187 | 6.827 s | 5.911 s | 1.155× | 1.149× |

All four summaries recorded the same native library SHA256
`3f447bcdb709bbbc2bd042213aecb5212676306c09dceb61c948217aae0a7e35`.
Reports are under `outputs/z-w8a8-routed-a{3072,4096}-on-demand-bf16-abba-{fox42,lighthouse187}/summary.json`.
The GPU complement stores W8 but these timings use BF16 GEMM after
per-invocation dequantization, **not** GPU INT8 arithmetic. The Core ML
compute plan preference does not prove actual ANE/GPU overlap.

Same-seed image PSNR against original BF16 GPU and visual inspection:

| Route | fox/42 | lighthouse/187 | tram/314 |
| --- | ---: | ---: | ---: |
| 3072 | 16.45 dB | 16.63 dB | 14.36 dB |
| 4096 | 15.80 dB | 16.59 dB | 13.14 dB |

The existing public macOS Vision feature-print helper gives the following
**descriptive** BF16-to-hybrid distances on those same warmed PNGs (revision
2, scale-fill, macOS 26.6.2, build 25G83; lower is more similar):

| Route | fox/42 | lighthouse/187 | tram/314 |
| --- | ---: | ---: | ---: |
| 3072 | 0.2485 | 0.4294 | 0.2469 |
| 4096 | 0.2967 | 0.4476 | 0.2575 |

The 3072 route is closer on these three paired feature prints, consistent
with their PSNR ordering, but neither metric has a calibrated Z-Image pass
threshold. Fox and lighthouse use the same-hash ABBA warmed image pairs;
tram uses separately generated BF16/hybrid images from the same native hash.
An identical-image Vision control scored 0. This OS-dependent feature metric
does not assess prompt alignment or fine text/gear correctness.

Both route widths produced clear fox and yellow rainy watercolor tram images
and a distinct transparent, patterned *glass* lighthouse, unlike the
contiguous adaptive graph. The composition still differs substantially from
the BF16 reference, so these three examples do not replace the design's
multi-prompt latent/LPIPS/DINO/CLIP, detail, artifact-rate, or broader human-evaluation
gates. The 3072 route favors fidelity at the cost of roughly half the 4096
route's speed gain; neither is yet a qualified production default.
One-request PNGs/reports are under
`outputs/z-w8a8-routed-a{3072,4096}-on-demand-bf16-{fox42,lighthouse187,tram314}/`.

Four further **4096-route-only** BF16/hybrid same-seed comparisons used the
same native library hash as the matched timings. Each path used a fresh
`baseline,fused` process pair with one excluded cold and one warmed request;
all BF16 health anchors passed. These runs examine quality, **not** speed
qualification. Pixel PSNR and human inspection of warmed `image-1.png` pairs:

| Prompt/seed | PSNR vs BF16 | Vision distance | Observed result |
| --- | ---: | ---: | --- |
| Elderly woman with round eyeglasses / 909 | 17.97 dB | 0.3283 | Both retain plausible facial/skin detail and glasses; framing differs. |
| Antique pocket watch with Roman numerals and gears / 271 | 11.87 dB | 0.3999 | Both have sharp clock faces and visible gears; the gear arrangement differs substantially. |
| Rainy neon sign reading “MOON CAFE” / 628 | 13.69 dB | 0.4528 | Hybrid spells the horizontal sign more closely; BF16 renders “MOO” stacked above “CAFE”. Single-sample result only. |
| Red apple, blue cup, yellow lemon / 742 | 18.35 dB | 0.2570 | Both retain all three objects and their colors; sizes/placement differ. |

Reports and PNGs are under `outputs/z-w8a8-routed-a4096-quality-{portrait909,watch271,sign628,objects742}/`.
The longer initial watch prompt could not run on this fixed-shape Core ML
artifact: its 1088 total token rows exceeded the graph's 1056-row capacity.
The baseline image and explicit rejection remain under
`outputs/z-w8a8-routed-a4096-quality-pocketwatch271/`; it has **no** matched
hybrid image or summary and must not enter aggregate comparisons. These
four cases broaden semantic/detail inspection but are too few to measure
artifact incidence or prove parity. The large same-seed pixel differences,
especially on intricate geometry, remain a quality concern. Multi-prompt latent,
LPIPS/DINO/CLIP and blinded side-by-side gates are still outstanding;
the current Python environment has no ready-to-run versions of their model
dependencies or checkpoints. Vision is only an additional descriptive
perceptual check and cannot fill those gates. The 1056-row bound also needs
an explicit product decision before supporting longer prompts.

The benchmark's optional `--parity` request also dumped all eight denoising
latents for **lighthouse/187** with each route, against separate same-hash
BF16 controls. It was excluded from warm speed samples. Every initial-noise
tensor was byte-identical, the two BF16 controls were identical at every
step, all tensors were finite, and each arm's warm image was byte-identical
to its subsequent parity image. Relative L2 to the BF16 latent (computed in
FP64 after loading the safetensors) shows divergence accumulating through
diffusion:

| Denoising step | 3072 route | 4096 route |
| --- | ---: | ---: |
| 1 | 0.0193 | 0.0222 |
| 2 | 0.0431 | 0.0450 |
| 3 | 0.0758 | 0.0781 |
| 4 | 0.1279 | 0.1289 |
| 5 | 0.2026 | 0.2005 |
| 6 | 0.3063 | 0.3050 |
| 7 | 0.4358 | 0.4370 |
| 8 / final | 0.5469 | 0.5521 |

The final latent cosine is `0.8491/0.8446` for 3072/4096. Early numerical
advantage does **not** prevent a large final latent difference. The paired
dump files and request provenance are under
`outputs/z-w8a8-routed-a{3072,4096}-latent-lighthouse187/`.
This is one prompt/seed, not a validated latent threshold or a replacement
for perceptual/semantic assessment. Both captures use the same native binary
hash as the matched ABBA timings; their separate one-warm-request timing
figures are not a speed qualification.

### Later-block sensitivity and caption-aware routing screen

To locate image-level error without changing the production default, the
4096 routed graph was ablated by placing selected complete blocks back on
BF16 GPU. On independent lighthouse/187, all cases used the *same* native
hash and BF16 reference; each is one warmed worker request, **not** an ABBA
speed qualification:

| BF16 GPU blocks | PSNR vs BF16 | Vision distance | Warm wall |
| --- | ---: | ---: | ---: |
| None (existing routed path) | 16.59 dB | 0.4476 | about 6.15 s |
| 0–15 | 16.52 dB | 0.3412 | 6.68 s |
| 16–31 | 18.25 dB | 0.3201 | 6.65 s |
| 16–23 | 15.95 dB | 0.3767 | 6.40 s |
| 24–31 | 16.61 dB | 0.4370 | 6.39 s |

The stronger 16–31 result is nonadditive: neither 8-block half produces
that PSNR on its own. All retain the glass structure; no fallback subset
qualifies a 32-FFN W8A16/W8A8 solution. Reports and PNGs are in
`outputs/z-w8a8-routed-a4096-fallback-{first16,last16,16to23,24to31}-lighthouse187/`.

The existing 4096-group route was fitted to *image rows only*. Real CPU-only
Core ML prediction on independent lighthouse FFN inputs exposed a distinct
caption issue: block 3 had image/caption partial-output relative L2
`0.0333/0.3661`; block 20 had `0.1818/0.0256` on separate captures.
`z_image_w8a8_routing.py` now has opt-in caption-region route scoring and
`--caption-image-slack` to attempt caption-improving group swaps inside a
training-image SSE budget. Its default image-only route is unchanged. This
is a **FP32 A8-only** approximation, excluding Core ML W8/FP16 effects,
GPU W8, denoising feedback, and physical device placement.

For block 3 (4096 ANE channels), the old image-only route versus a 1%-caption
weighted route scored `0.00237→0.00341` image and `0.14749→0.07098` caption
held-out error, both normalized to their respective full FFN output on
lighthouse/187. Vintage-car/571 held-out scoring likewise gave
`0.00241→0.00383` image and `0.13996→0.06275` caption: a pronounced
tradeoff, not a simultaneous improvement. With a 5% **training image SSE**
budget instead, the constrained route for block 3 scored
`0.00237→0.00277` image and `0.14749→0.12918` caption on lighthouse.
For block 2, the constrained route scored `0.00516→0.00511` image and
`0.11269→0.11001` caption on lighthouse, but `0.00503→0.00513` image and
`0.10181→0.09715` caption on vintage-car. A training-image budget plainly
does not guarantee held-out image improvement. Selection weights were
inspected on these held-outs during research, so neither set is an untouched
hyperparameter-selection test for a future export.

At the time of this local screen the route was **unexported**; a later full
export and image test is documented below and did not justify promotion.
Reproduce an image-only control with `--evaluate-caption`; the
weighted screen takes `--caption-route-weight 0.01`, and the constrained
screen takes `--caption-route-weight 1 --caption-image-slack 0.05`.
Use the diverse8 contiguous source manifest, fox/astronaut calibration,
independent lighthouse or vintage-car FFN captures, block 2 or 3, and
`--ane-width 4096`. No GPU runtime, exported graph, or production selection
was changed by this screen. The eight immutable machine-readable diagnostic
records (image-only versus constrained, blocks 2/3, two independent
held-outs) are under `outputs/z-w8a8-caption-aware-route-screen/` with
manifest/checkpoint/calibration/held-out digests.

The packed GPU W8 group-size check did not reveal a cheaper suffix either.
On the same held-out block-18/g32-vs-g128 capture, 16 warmed measurements
per mode gave `11.931/11.975 ms` for expression-local BF16 expansion plus
GEMM, while the image-row suffix relative L2 increased
`0.00608→0.00734`. The BF16 controls were `10.867/10.849 ms` in those
separate processes. The g128 weight shard is smaller (`73.0` versus `79.6`
MB) but **neither faster nor more faithful** in this isolated test, so it
was not promoted to a full-model artifact or production setting.

## Reproduction and safeguards

The final 32-block source and compiled manifests are under
`models/coreml/z_image_w8a8_regions_image_sq_all/manifest.json` and
`models/coreml/z_image_w8a8_regions_image_sq_compiled/`. Benchmark reports
and PNGs are in `outputs/z-w8a8-regions-image-sq-8step/` and
`outputs/z-w8a8-regions-image-sq-w8-fallback-2to31-v2-8step/` (the other
prompt adds `-seed123`). The BF16 reference is
`outputs/z-w8a8-gpu-baseline-8step/`.

For the selective test, set
`TURBOCIDER_Z_HYBRID_BF16_BLOCKS=2,3,...,31` and use the same compiled
manifest. Leave `TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE` unset to retain GPU W8 on
blocks 0–1. Selected fallback blocks retain complete BF16 GPU weights; the
native precision report includes `+selective_bf16_gpu_blocks`. An all-block
fallback produced a PNG **pixel-identical** to the BF16 GPU reference.

Use `tools/validation/z_image_w8a8_branch.py --image-rows 1024` to see image
and caption errors separately. `z_image_w8a8_layers.py --image-rows 1024`
provides an FP32 activation-boundary ablation, not a substitute for Core ML
prediction or image comparison. Tests: `.venv/bin/python -m unittest
tests.native.test_z_image_smoothquant tests.native.test_z_image_w8a8_layers
tests.native.test_z_image_w8a8_coreml tests.native.test_z_image_w8_suffix_benchmark`.

The on-demand BF16 GEMM mode now has a matched two-prompt warm ABBA denoise
speed gain with packed W8 storage; its *isolated* GPU suffix is still slower
than resident BF16, and there is no physical ANE trace. The remaining work is
to establish physical ANE placement and GPU/ANE overlap, measure a broader
semantic/detail and latent-quality gate, then improve per-prompt fidelity and
the isolated GPU W8 suffix. Do not qualify the original objective on speed
alone or silently make this experimental mode the production default.

### Visual-first acceptance clarification and caption-aware route follow-up

The user clarified the acceptance target: **at least 90% of FFN layers** must
take the ANE W8A8 plus GPU-complement path (29/32 minimum); visual output
should look close and useful, but need not reproduce the BF16 image exactly.
PSNR is diagnostic, not an automatic rejection threshold. The current 4096
route has a 32-block W8A8 manifest, and a warmed 512²/8-step report records
512 successful Core ML predictions (32 blocks per denoiser evaluation), with
GPU work submitted asynchronously before each prediction. This verifies the
runtime route and intended overlap; `observed_ane_residency` remains `unknown`,
so it does **not** verify physical ANE placement or concurrency.

Human side-by-side inspection of the same-seed 4096-route images shows usable
portraits, recognizable pocket-watch gears and Roman numerals, all three
requested colored objects, and a readable `MOON CAFE` sign. Their compositions
can differ conspicuously from BF16, especially intricate watch internals; on
this sign sample the hybrid actually spells the requested words more closely.
The glass lighthouse keeps a glass lighthouse interpretation, though its
silhouette/decorations differ. Under the clarified visual-first criterion,
these examples are **promising rather than failed because of PSNR**. They are
still too few to establish a low artifact rate across prompts/seeds, and no
user visual acceptance of a representative set has been recorded.

An opt-in caption-aware route changed only blocks 2 and 3, with all 32 source
graphs exported to
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_routed_caption_slack05_b2_3_diverse8_all/`.
The managed cache compiled all 32 to manifest
`models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_routed_diverse8_compiled/manifest-cc576df421868ffab11b67af79d9351cfe88bcaeabcfb8d7d2478bdfd022ef50.json`.
All matched image requests ran with the unchanged native library SHA256
`3f447bcdb709bbbc2bd042213aecb5212676306c09dceb61c948217aae0a7e35`.
Compared with the original route against the *same* BF16 image:

| Prompt/seed | Old/new PSNR dB (descriptive) | Old/new macOS Vision distance | Visual finding |
| --- | ---: | ---: | --- |
| Portrait/909 | 17.97 / 18.42 | 0.3283 / 0.3548 | Both natural, close-up, glasses intact. |
| Watch/271 | 11.87 / 11.88 | 0.3999 / 0.3455 | New watch gears seem structurally nearer BF16, but hands differ. |
| Sign/628 | 13.69 / 13.40 | 0.4528 / 0.5176 | Both spell the words; new circular sign/framing diverges more. |
| Lighthouse/187 | 16.59 / 14.64 | 0.4476 / 0.5082 | Both depict glass, but new silhouette/decor differs more. |

These are individual samples, not a statistically significant ranking; the
lighthouse and route-selection diagnostic captures have already influenced
hyperparameters. The caption-aware export is **not** promoted over the
original route; its single warmed request wall times are about 6.15 s and
cannot replace a matched ABBA speed test. Reports and PNGs are under
`outputs/z-w8a8-caption-aware-{quality-portrait909,watch271-matched,sign628,lighthouse187}/`.
The one-off watch experiment in `quality-watch271/` uses a *different prompt*
and must not be included in that paired comparison.

W8A8 is not categorically faster than W8A16 for a *real Core ML graph*.
On M4 Max, a block-18/1056-row captured-input, 12-round warmed
`CPU_AND_NE` ABBA using the same contiguous 4096-channel W8 weights measured
13.414 ms for FP16-activation W8A16 against 11.860 ms for the simpler
W8A8 graph (`1.131×` single-branch throughput). However, the quality-oriented
two-region + SmoothQuant + four-hidden-group W8A8 graph measured 14.33 ms
on the same block against about 13.42 ms for the simple W8A16 graph; a
contiguous/c4 versus routed/c4 ABBA was essentially tied at 14.333/14.350 ms.
The simple A8 graph is **not** the visually acceptable routed graph and its
large output delta cannot be traded for speed without full-image validation.
The Core ML compute plan prefers `MLNeuralEngineComputeDevice` for both
projection convolutions and Q/DQ operations, but preference is not a runtime
placement trace. Reproduce using
`tools/coreml/benchmark_z_image_a8_channels.py --block 18 --rounds 12 --warmups 3`
with the relevant source manifests and lighthouse capture
`outputs/z-w8a8-heldout-8step-lighthouse/capture/block18/sample-21100-126521800954333-242.npy`.

A stronger same-artifact placement clue uses
`tools/coreml/benchmark_z_image_compute_units.py`: the existing **routed**
block-18 W8A8 `.mlmodelc`, same captured input, 12-round warmed ABBA,
`CPU_ONLY`/`CPU_AND_NE` medians were `23.300/14.340 ms` (`1.625×`). The
machine-readable report with manifest and capture SHA-256 is
`outputs/z-w8a8-routed-a4096-block18-compute-units-abba.json`.
The two compute-unit selections do **not** produce bitwise-equivalent outputs:
the image-row relative L2 is `0.03751` and the caption-row L2 `0.01682`.
This makes CPU-only an inappropriate fidelity oracle for the accelerated path.
The large timing gap plus preferred-device plan are strong evidence that the
NE-enabled execution path is different and faster, but are still not a
physical ANE residency or GPU/ANE overlap trace.

The runtime's existing opt-in `TURBOCIDER_Z_PROFILE` instrumentation directly
tests the *software schedule*: `overlap` submits GPU suffix asynchronously,
then calls Core ML; `serial` waits for that GPU output before calling Core ML.
With the original 4096 route, identical fox/42 request and native library,
four fresh processes in overlap/serial/serial/overlap order produced four
warmed requests per mode:

| Profiled schedule | Median 512²/8-step denoise | Median warmed wall |
| --- | ---: | ---: |
| GPU submit before Core ML (`overlap`) | 5.971 s | 6.211 s |
| GPU completion before Core ML (`serial`) | 8.672 s | 8.925 s |

The `serial/overlap` denoise ratio is `1.452×`; all four `image-1.png` hashes
were identical, and each process logged two complete 256-block warm traces.
The representative `overlap/serial` median *branch windows* were about
`11.9/21.4 ms`, with about `9.4/9.2 ms` inside the Core ML call and a
roughly `2.5 ms` GPU tail wait for overlap versus a `12.1 ms` pre-Core-ML
GPU wait for serial. These profile evaluations perturb scheduling and are
**not** an uninstrumented ABBA speed claim against BF16 GPU. They provide
strong causal evidence of useful software overlap, but not a physical trace
of simultaneous ANE and GPU utilization. Per-run reports, PNGs and JSONL
traces are under `outputs/z-w8a8-overlap-abba-{0,1,2,3}/` and sibling
`outputs/z-w8a8-overlap-abba-{0,1,2,3}.jsonl`.

### Wider routed ANE partition: 4608 channels (new speed candidate)

The route suite can now select an ANE width greater than the **source
calibration manifest's** 4096 channels: that width described its prior
export, not the extent of the checkpoint's 10240 hidden channels. The same
training captures and checkpoint SHA-256 bind all 32 routes. The new
training-only route, 32 source graphs, and compiled manifest are:

- `outputs/z-w8a8-diverse8-channel-route-a4608.json`
- `models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_routed_a4608_diverse8_all/manifest.json`
- `models/coreml/z_image_w8a8_regions_image_sq_hidden_c4_routed_a4608_diverse8_compiled/manifest-467221c6f89f7eb66074dba0d8ecceb88773417362f44c2e887b393ed6d9e4fd.json`

An isolated held-out block-18 screen explains the search direction, **not**
the end-to-end result. The GPU suffix's resident-W8, per-invocation BF16
expansion plus GEMM measured about `11.93/11.10/10.03 ms` at 4096/4608/5120
ANE widths on the same capture. A corresponding contiguous four-hidden-group
Core ML W8A8 graph measured about `14.25/15.20/16.57 ms`. These are separate
process/device measurements; their sums or maxima do not predict mixed
end-to-end latency. Route selection for block 18 changed the held-out FP32
A8-only output L2 versus the full FFN from `0.0416` (4096 route) to `0.0431`
(4608 route); the small local increase is not a visual-quality verdict.

Same original BF16 GPU versus **new 4608 hybrid**, 512²/8-step, two-prompt
fresh-process ABBA, four warmed requests per arm and healthy BF16 anchors:

| Prompt/seed | BF16 denoise | 4608 denoise | Denoise speedup | Wall speedup |
| --- | ---: | ---: | ---: | ---: |
| Fox/42 | 6.829 s | 5.695 s | 1.199× | 1.191× |
| Glass lighthouse/187 | 6.825 s | 5.696 s | 1.198× | 1.190× |

The original 4096 route versus the 4608 route, also fresh-process
4096/4608/4608/4096 with four warmed fox requests per arm and the **same**
native library SHA256 `3f447bcdb709bbbc2bd042213aecb5212676306c09dceb61c948217aae0a7e35`,
measured `5.912→5.696 s` denoise and `6.155→5.938 s` wall (`1.038×`
denoise improvement). Reports and PNGs are under
`outputs/z-w8a8-routed-a4608-on-demand-bf16-abba-{fox42,lighthouse187}/`
and `outputs/z-w8a8-a4096-vs-a4608-abba-{0-old,1-new,2-new,3-old}/`.
The warm report records `block_count=32`, 512 successful predictions across
two requests, no runtime failures, `compute_units=cpuAndNeuralEngine` and
`observed_ane_residency=unknown`. This meets the *software-route* 90% FFN
coverage target, not a physical trace of NE residency or simultaneous GPU/NE.

A profiled 4608 fox request gave `10.269 ms` median Core ML call,
`0.803 ms` GPU tail wait and `11.087 ms` total branch window versus the
4096 profile's `9.412/2.493/11.924 ms`. The 4608 win is consistent with
reducing GPU wait while keeping the Core ML side near balance. Profiling
changes evaluation scheduling, so those numbers explain rather than qualify
the uninstrumented ABBA result. A 4864-channel **single-block-only** follow-up
measured `15.254→16.076 ms` Core ML (4608→4864) and approximately
`11.10→10.54 ms` isolated GPU W8/BF16 suffix. The opposing changes do not
make a compelling case to allocate another full 32-graph artifact before
optimizing either branch; no 4864 end-to-end speed or image claim is made.

Visual inspection of fox, portrait, watch, neon sign, three objects, and
glass lighthouse found coherent, mostly prompt-matching images; the sign
spells `MOON CAFE`, the watch has Roman numerals/gears, and the lighthouse
remains glass. The three-object 4608 image has a less obvious cup handle,
and the watch's hand/gear arrangement differs strongly from BF16. The
4608 lighthouse has a different silhouette/interior; under the user's
visual-first requirement it has **not** obviously collapsed, but this is not
a broad human acceptance study. Paired descriptive BF16-to-4608 metrics:

| Prompt/seed | PSNR dB | Vision feature distance |
| --- | ---: | ---: |
| Fox/42 | 16.52 | 0.3073 |
| Lighthouse/187 | 14.74 | 0.4699 |
| Portrait/909 | 17.56 | 0.3836 |
| Watch/271 | 12.22 | 0.3796 |
| Sign/628 | 13.56 | 0.4764 |
| Objects/742 | 16.47 | 0.3873 |

These metrics are **not acceptance thresholds**, and several differ more
from BF16 than the 4096 route despite the extra speed. Non-ABBA quality
outputs are under `outputs/z-w8a8-routed-a4608-quality-{portrait909,watch271,sign628,objects742}/`.
The route remains experimental rather than silently becoming the default.

An additional block-18 optimization probe replaced four independent
hidden-channel image-region A8 Q/DQ pairs with a single per-channel-vector
Q/DQ. Its computed output was **bitwise identical** on the captured input,
and the compute plan reduced four image-channel Q/DQs and concatenations to
one. Yet 16-round warmed ABBA medians were `14.333/14.325 ms` for the old/
vector graphs: no material improvement. The optional exporter change was
reverted rather than promoting another 32-block artifact.

### Matched comparison with the earlier GPU/ANE W8A16 partition (September 25, 2026)

The September 7 acceptance record reported **1.212×** for the older BF16 GPU
complement plus W8A16 Core ML partition at **1024²/9 steps**. That number is
not comparable to the current W8A8 **512²/8-step** speedups. The old compiled
4096-channel, 1056-row, 32-block W8A16 manifest is available locally, so the
following comparisons instead use the same M4 Max, checkpoint, 512²/8 steps,
prompt/seed and native binary per crossover. Each arm has two fresh processes
in old/new/new/old order, one excluded cold request and two measured warm
requests per process. All reported times are medians of four warm requests.

| Same-library comparison | Old W8A16 + BF16 GPU | New W8A8 (4608) + GPU W8/BF16 on-demand expansion | New W8A8 (4608) + resident BF16 GPU |
| --- | ---: | ---: | ---: |
| Fox/42 denoise (s) | 5.598 / 5.603 | 5.692 | 5.449 |
| Fox/42 wall (s) | 5.848 / 5.851 | 5.932 | 5.692 |
| Transparent-glass lighthouse/187 denoise (s) | 5.601 | not measured for this prompt | 5.432 |
| Transparent-glass lighthouse/187 wall (s) | 5.847 | not measured for this prompt | 5.679 |

The two fox comparisons use different but individually identical binaries:
SHA256 `3f447bcdb709bbbc2bd042213aecb5212676306c09dceb61c948217aae0a7e35`
for the first two arms and experimental isolated-build SHA256
`6f9d74f04d88b09f977b89c37352140a600d8e8d6ad6ed850ba4531fda6f9238`
for old versus BF16-GPU W8A8. Thus the shipped-style W8 GPU complement was
**1.6% slower** than the old path in the fox crossover, while the BF16 GPU
complement was **2.83%/2.79% faster** on fox (denoise/wall) and **3.12%/2.95%
faster** on lighthouse. Separate BF16-only warm controls under the second
binary were 6.828/6.826 s denoise for fox/lighthouse, putting its W8A8+BF16
GPU speedups at about 1.25–1.26× for these two 512² prompts. These are
explicit experimental requests, not a production default or a prediction
about the 1024² workload.

The two image prompts generated finite, recognizable fox and transparent
glass lighthouse images with all 32 FFNs using Core ML; identical requests
across fresh processes produced identical PNG hashes. Relative to matching
BF16 GPU PNGs, fox old/new PSNR was `23.84/16.22 dB` (RGB MAE `5.40/19.47`),
and lighthouse old/new was `33.18/19.24 dB` (MAE `1.92/12.78`). These are
large same-seed differences, especially in the lighthouse structure, **not**
claims of a catastrophic-looking image or an automatic visual rejection.
Two usable images do not establish acceptance across a larger prompt suite.

Crucially, the advantage is **not** evidence that this quality-oriented A8
Core ML graph computes faster than W8A16. Measured warm per-prediction Core ML
elapsed time was roughly `8.0–8.1 ms` for the old W8A16 graph versus
`9.9 ms` for the new 4608-channel routed W8A8 graph. The new GPU branch has
fewer channels, and removing GPU W8 expansion also removes work from the
critical path. A 4096-channel routed W8A8 + BF16 GPU single-process pilot
gave `5.592/5.593 s` warm denoise, effectively tied with the 4096-channel
W8A16 control (`5.603 s` crossover median); it is not ABBA-qualified.
The routed two-region, SmoothQuant and hidden-A8 Q/DQ graph has overhead
absent from the simpler A8 microbenchmark. All paths requested
`cpuAndNeuralEngine`; physical NE residency/concurrency remains `unknown`.

Detailed JSON and exact PNGs are under
`outputs/z-w8a16-vs-w8a8-a4608-20260925/`,
`outputs/z-w8a16-vs-w8a8-bf16gpu-20260925/` and
`outputs/z-w8a16-vs-w8a8-bf16gpu-lighthouse-20260925/`. The BF16 GPU
complement path is opt-in with `TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE=1`,
implemented by selecting the routed GPU channel complement as resident BF16
matrices. It was built in `build/native-w8a8-bf16` and has not replaced
`build/native/libturbocider.dylib` or been committed.

### Where the A8 single-layer gain goes (September 25, 2026)

On the same M4 Max and held-out 1056-row inputs, separately compiled
4096-channel Core ML graphs were compared in **16-round warmed
`CPU_AND_NE` ABBA** probes. The image-row relative L2 below compares
the two Core ML outputs, not full generated images.

| Block | W8A16 ms | Simple W8A8 ms (change) | Two-region + SQ + four-group W8A8 ms (change) | Image-row L2, simple / grouped |
| --- | ---: | ---: | ---: | ---: |
| 3 | 13.357 / 13.343 | 11.811 (-11.6%) | 14.233 (+6.7%) | 0.927 / 0.0368 |
| 18 | 13.410 / 13.406 | 11.853 (-11.6%) | 14.327 (+6.9%) | 0.907 / ~0.098 |
| 29 | 13.284 / 13.283 | 11.718 (-11.8%) | 14.175 (+6.7%) | 0.793 / 0.0574 |

The two W8A16 columns in each row come from separate paired probes; block
18 was measured earlier. The simple graph has two activation Q/DQ pairs,
while the grouped graph has seven plus eight indexed slices and three
concats. This makes **graph overhead** a strong explanation for the reversal,
though these timings do not isolate each operator's cost. The simple A8
graph's huge local output differences preclude treating its speed as an
acceptable quality-preserving replacement. All reported operators prefer
ANE in the compute plan; these tests cannot establish physical INT8 MAC or
runtime device occupancy. The routed 4608 graph further changes width and
channel selection, so its native timings are not same-width comparisons.

Warm full-request `TURBOCIDER_Z_PROFILE` traces of the resident-BF16-GPU
complement each included 256 FFN calls:

| A8 width | GPU pre-FFN + pack | Core ML calls | GPU tail wait | Parallel branch window | GPU merge | Warm denoise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2.785 s | 2.305 s | 0.420 s | 2.735 s | 0.104 s | 5.632 s |
| 4608 | 2.796 s | 2.508 s | 0.028 s | 2.546 s | 0.104 s | 5.453 s |

Concurrent branch times **must not be added**. The 4096 branch still waits
on GPU for a median 1.755 ms/call; at 4608 that is only 0.096 ms/call,
while the Core ML median rises from 8.889 to 9.748 ms/call. Thus widening
the ANE share reduces GPU work but makes Core ML nearly the entire branch
critical path. Even a hypothetical free 1 ms off *every* exposed Core ML
call saves no more than 256 ms, about 4.7% of a 5.453 s denoise. A separate
GPU-only `gpu_split` diagnostic measured 4.617 s full-width BF16 FFN within
7.588 s **instrumented/unfused** denoise (pre-FFN GPU 2.841 s). This is not
the optimized 6.83 s GPU control or an exact Amdahl decomposition of the
matched crossover. Logs: `outputs/z-w8a8-bf16gpu-{a4096-profile,a4608-profile}-20260925.jsonl`
and `outputs/z-w8a8-gpu-full-split-20260925.jsonl`.

One default-off diagnostic uses the existing compiled GPU pre/post hybrid
graphs only with routed A8 plus BF16 GPU, selected by both
`TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE=1` and
`TURBOCIDER_Z_W8A8_COMPILED_HYBRID=1`. Using the same isolated native library
(SHA256 `b99ef100a684cb37c35eb51e98165723d1845f38dd9826e83afe38f6c4e01283`),
an eager/compiled/compiled/eager crossover with two warm fox/42 requests
per process measured **5.4449 -> 5.3890 s denoise** and **5.6872 ->
5.6320 s wall** (about 1.0% gain); every image hash agreed. The compiled
warm trace summed 2.766 s GPU pre-FFN, 2.543 s Core ML and 0.015 s GPU
tail. These traces are intrusive diagnostics, and the ~1% crossover has
not been qualified as a production default. Reports are under
`outputs/z-w8a8-compiled-abba-20260925/`. The certified default native
library was not overwritten.

`../splash/README.md` uses another workload: Qwen3.8-27B with a Q4 GPU
control, 2048-row FFN chunks and 8704 of 17408 FFN channels on the ANE-
preferred A8 branch. Its one captured FFN took 83.63 ms on GPU versus
42.22 ms parallel; 32K-token prefill took 143.631 versus 106.728 s
(1.346x, single-request points). `../splash/notes/2026-09-20-ane-phase2-critical-path-and-integration.md`
also records an analogous lost benefit: F6144 W8A8 reduced its ANE time
38.2 -> 28.6 ms without improving the ~54.2 ms joined FFN, because GPU
remained critical; widening to F8704 balanced the branches. Different GPU
precision, row count, FFN share and *prefill versus full-image* denominator
mean its 1.346x cannot be transplanted to Z-Image. The matched current
512-square comparison instead shows only 2.8-3.1% advantage over the old
W8A16 hybrid with resident BF16 GPU; the direct packed GPU W8 variant was
slower than the old hybrid. None of these measurements prove physical ANE
placement or qualify image quality across many prompts.

### Full-width token-row split versus channel split (September 25, 2026)

The experimental FFN splits the **10,240 SwiGLU intermediate channels**, not
the 3,840 input embedding channels. The 1,024-image-row variant excludes 32
caption rows from ANE but **continues to split each image row by intermediate
channels**. Splash's measured GPU/ANE prefill also uses this channel split;
its M/token-row split is an unintegrated candidate.

An opt-in, **single-block-only** `export_z_image.py --row-split-probe` exports
all 10,240 channels on ANE with fixed image-row buckets; the native runtime
rejects these full-width artifacts. `benchmark_z_image_row_split.py` uses an
independent block-18 1,056-row capture and original BF16 GPU weights, dispatches
the remaining rows' complete FFN asynchronously on GPU while predicting on
Core ML `CPU_AND_NE`, and joins both branches inside its measured window.
Inputs and weights are ready/resident beforehand. The test omits native
GPU→ANE transport, output scatter and all-layer model residency. Each mode
had three warmups and 16 or 24 timed samples in a symmetric alternating
schedule. W8 and both dynamic A8 boundaries passed the graph gate; compiled
convolutions prefer ANE. Physical INT8 arithmetic has not been traced.

| ANE image rows / GPU remaining rows | BF16 full FFN ms | GPU remaining ms | ANE full FFN ms | parallel ms | image-row L2 vs BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 / 800 | 17.83 | 13.94 | 7.12 | 14.01 | 0.0381 |
| 384 / 672 | 18.04 | 11.87 | 10.66 | 11.74 | 0.0521 |
| **416 / 640** | **17.95** | **11.04** | **11.35** | **11.37 (1.58× isolated)** | **0.0513** |
| 512 / 544 | 17.96 | 9.67 | 15.06 | 14.87 | 0.0628 |
| 768 / 288 | 18.00 | 5.65 | 22.03 | 21.87 | 0.0614 |

The best **pre-bridge** row-split window is already near the ~11.09 ms
profiled native 4,608-channel branch window from a *different experiment*;
these are not a matched head-to-head result. Full-width ANE weights require
2.5× the bare W8 bytes of a 4,096-channel ANE model. Whole-output L2 is
misleadingly small because the 32 caption rows remain exactly BF16 GPU: the
image-row errors above matter more, but do not predict final image quality.
JSON receipts are in `outputs/z-w8a8-row-split-block18-{256,384,416,512,768}-20260925.json`;
source manifests are in `models/coreml/z_image_w8a8_fullwidth_rows*_block18_probe/`.
No row-only runtime or 32-block row artifact has been installed.

### 1,024 image rows / 5,120 channels: faster complete BF16-GPU/W8A8-ANE path

The opt-in image-only BF16 GPU route computes the **full caption FFN** on GPU,
the GPU image suffix in parallel with the ANE W8A8 image prefix, then merges.
It is **not pure token-row split**. On block 18, moving from 4,096 to 5,120
channels raised the isolated CPU_AND_NE median `12.306 → 14.190 ms` (16-round
ABBA), but shortened the isolated resident BF16 GPU image suffix
`10.206 → 8.638 ms` (eight forward/reverse rounds). These Python timings
cannot be added to predict native concurrent execution.

An initial 4,096-channel image-only pilot succeeded on fox/42 with 5.644 s
warm denoise. A **single-library** old/image-only/4,608/4,608/image-only/old
crossover with one excluded cold and one warm request per process found warm
denoise medians **5.608/5.636/5.437 s** respectively. Thus omitting captions
at 4,096 channels alone did not beat the old W8A16 hybrid. Images were
deterministic per configuration and fox and a separate glass-lighthouse
prompt looked coherent; the latter was not a paired quality comparison.
Reports: `outputs/z-w8a8-imageonly-threeway-20260925/` and
`outputs/z-w8a8-imageonly-bf16gpu-lighthouse187-20260925/`.

The new 32-layer 5,120-channel source manifest is
`models/coreml/z_image_w8a8_imageonly_hidden_c4_a5120_b1024_diverse8_all/manifest.json`;
the compiled manifest is in the corresponding `_compiled/` directory. The
experimental native binary SHA256 is
`fd027561ef327ab144423e84b0fa91ab1713f8f715f0650a2bae51cedfd77cab`.
On the same M4 Max, fox/42, 512² and eight steps, six fresh processes in
new/4,608/old/old/4,608/new order each ran one excluded cold and one warm
request:

| Same-library route | Warm denoise median (s) | Warm request wall (s) | ANE calls/image | ANE prediction ms/call |
| --- | ---: | ---: | ---: | ---: |
| 4,096-channel W8A16 ANE + BF16 GPU | 5.603 | 5.850 | 256 | 8.04 |
| 4,608-channel routed W8A8 ANE + BF16 GPU | 5.453 | 5.698 | 256 | 9.90 |
| **5,120-channel image-only W8A8 ANE + BF16 GPU** | **5.270** | **5.515** | **256** | **9.31** |

The *same library* gave a separate two-warm-request BF16 GPU-only control of
`6.824 s` denoise and `7.067 s` wall. The new path is therefore about
**1.295×** faster in denoise than this control, **6.3%** faster than the old
W8A16 hybrid and **3.5%** faster than the 4,608-channel BF16-GPU route.
These are warm-request results, not cold-start speedups: first-request wall
times including model setup took roughly 12–21 s. Same-library/seed/hash agreement was
checked across two processes per hybrid arm. `CPU_AND_NE` and preferred-ANE
compute plans do not prove physical integer MAC usage. Raw reports and PNGs:
`outputs/z-w8a8-imageonly-a5120-threeway-20260925/` and
`outputs/z-w8a8-imageonly-a5120-bf16gpu-control-fox42-20260925/`.

Additional fox, glass-lighthouse/187, watch/271 and neon-sign/628 images
were finite and visibly depicted the requested objects; `MOON CAFE` is
legible. The watch's internal gear/hand design differs substantially from
BF16 but remains a plausible watch; the lighthouse is transparent and close
in overall structure to its prompt-matched BF16 image. These **four prompts**
pass a preliminary visual-usefulness screen, not broad quality
qualification or numeric parity. Reports/PNGs:
`outputs/z-w8a8-imageonly-a5120-{bf16gpu-fox42,quality-lighthouse187,quality-watch271,quality-sign628}-20260925/`.
The route remains default-off. At the time of this pilot it needed both
`TURBOCIDER_Z_W8A8_IMAGE_ONLY=1` and
`TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE=1` plus the marked manifest. The later
explicit manifest-driven path below removes these environment requirements
without promoting the route into the `auto` policy. Profile the native GPU
tail/ANE wait and expand the independent visual set before any automatic
policy change.

Five additional **paired visual** prompts ran with the later isolated
library SHA256 `d986ad841bb57d82f76472a31894d5375d0bd0e349b9792d16df1ddfbe14c3ed`,
original BF16 GPU versus the same 5,120-channel/1,024-image-row manifest,
same prompt/seed in each pair, one excluded cold and **one warm** request per
arm. Every hybrid warm request accumulated 512 successful Core ML predictions
including its cold request, i.e. 256 per image and all 32 FFNs per denoiser
step. The system short-prompt anchor passed; each arm's cold/warm PNG matched.
The timings below are only single warm samples, **not ABBA rankings**:

| Prompt / seed | Real text tokens / padded rows | BF16 GPU / hybrid warm denoise | Paired visual finding |
| --- | ---: | ---: | --- |
| Ceramic tea set / 313 | 47 / 64 | 7.159 / 5.326 s | Blue teapot, pale-green cup, yellow lemon all retained. |
| Rainy bookstore alley / 733 | 45 / 64 | 7.161 / 5.333 s | Bicycle, umbrella-bearing person and wet cyan-lit alley retained; sign glyphs vary. |
| Elderly violinist / 421 | 41 / 64 | 7.176 / 5.313 s | Round glasses, violin and bow retained; facial and hand details differ. |
| Red apple, blue cup, yellow toy car / 742 | 44 / 64 | 7.162 / 5.323 s | All three objects, their colors and the cup handle retained. |
| Watercolor viaduct / 533 | 103 / 128 | 7.617 / 5.796 s | Steam train, stone arches, turquoise river, two yellow hikers and watercolor style retained. |

I inspected both warm PNGs of each pair at 512²; none of these five shows a
gross subject or style collapse. The tea set and three-object image are
visually close in arrangement; violinist facial features and watercolor
bridge/train details differ more. This brings the visually screened candidate
to several prompt classes, not a calibrated failure rate or a user-approved
quality gate. The separate 494-token repeated-prompt performance sample is
not counted as a natural-language quality test. Reports and PNGs are in
`outputs/z-w8a8-imageonly-a5120-quality-{tea313,alley733,violin421,objects742,watercolor533}-20260925/`.

Five further **seed-variation visual pairs**, using the exact watch and
`MOON CAFE` prompts from the prior screen and the new violinist prompt,
tested the same compiled 5,120-channel manifest and isolated native binary
SHA256 `d986ad841bb57d82f76472a31894d5375d0bd0e349b9792d16df1ddfbe14c3ed`.
Each arm ran one excluded cold plus one warm request in its own process;
its cold/warm PNG hashes agreed, all hybrid arms accumulated 512 Core ML
calls (256 per image), and each GPU baseline met the short-prompt system
health anchor. The speed values below are **single matched warm points**, not
fresh-process ABBA qualifications:

| Prompt / new seed | BF16 GPU / image-only W8A8 hybrid warm denoise | Side-by-side 512² visual observation |
| --- | ---: | --- |
| Pocket watch / 37 | 6.830 / 5.292 s | Both have a clear dial, Roman numerals and gears; center gearing/hand shape differ. |
| Pocket watch / 519 | 6.823 / 5.274 s | Both retain numeral ring and exposed movement; different gear and hand arrangement. |
| MOON CAFE / 144 | 6.829 / 5.266 s | Both spell `MOON CAFE` legibly in a rainy street scene. |
| MOON CAFE / 965 | 6.824 / 5.275 s | Hybrid spells both words; the BF16 `MOON` has letters obscured by the round neon border. |
| Elderly violinist / 604 | 7.159 / 5.327 s | Both retain portrait, round glasses, bow and violin; hand anatomy and face details vary. |

I inspected both warm PNGs for every seed. None shows an obvious catastrophic
quality failure under the user's visual-first criterion. Text and intricate
watch details can change substantially, and a handful of selected prompts
cannot establish the failure rate over unseen prompts or user acceptance.
Receipts/PNGs: `outputs/z-w8a8-a5120-seed-{watch37,watch519,sign144,sign965,violin604}-20260925/`.

Follow-up **instrumented** same-library fox profiles each recorded all 256
warm FFN calls: the 5,120 image-only route summed `2.768 s` pre-FFN/pack,
`2.394 s` Core ML, `0.050 s` GPU tail wait, `2.462 s` branch window and
`0.086 s` merge; the 4,608 routed route summed `2.795/2.511/0.027/2.549/0.105 s`.
The 5,120 image-only branch is already close to balanced (Core ML median
9.210 ms, GPU tail median 0.174 ms). Instrumentation changes graph scheduling;
these totals explain the direction of the uninstrumented win, not its exact
size. Further widening ANE without profiling is unlikely to be free. Traces:
`outputs/z-w8a8-a5120-vs-a4608-profile-{new,best}-20260925.jsonl`.

An image-only 5,120-channel **two**-hidden-A8-group single-block probe did
reduce Core ML Python-call medians by about `0.34–0.36 ms` versus four groups
on held-out blocks 3/18/29. But its ANE partial-output image-row relative
L2 *against the four-group graph* was `0.0345/0.0386/0.0177`, respectively.
The current native 5,120-channel profile already waits on the GPU tail for a
median `0.174 ms` after Core ML returns. Less ANE work cannot be assumed to
shorten those GPU-limited calls, and it introduces additional quantization
error. Therefore only three source single-block probes were exported under
`models/coreml/z_image_w8a8_imageonly_hidden_c2_a5120_b1024_{b18,b3_29}/`;
no 32-block/two-group promotion or speed/quality claim is made.

### Variable caption rows and a rejected post-merge micro-optimization

The source route originally required exactly `1,024 image + 32 caption`
rows. The Core ML image-only graph itself always consumes exactly 1,024
image rows; the existing GPU caption FFN and post merge already handled a
variable number of additional rows. The explicit image-only selector now
accepts a **512×512** image with `32...1,024` caption rows in 32-row
increments. It still rejects other image shapes, requires its marked
1,024-row Core ML manifest and full resident BF16 GPU caption weights, and
does not affect the automatic or default GPU policy. The first two noise
refiner blocks remain image-only; all 30 main blocks keep the GPU/ANE
parallel channel split plus the complete BF16 caption FFN.

With isolated library SHA256
`d986ad841bb57d82f76472a31894d5375d0bd0e349b9792d16df1ddfbe14c3ed`,
the 21-token fox/42 request still used 32 caption rows and yielded the
**same PNG SHA256** as the earlier 5,120-channel binary. Separate one-process
requests on two independent long prompts produced the following **single
warm-request points** (one excluded cold request per process):

| Real text tokens / padded caption rows | Pure BF16 GPU denoise | 5,120-channel image-only W8A8 denoise | ANE calls per warm request | Observation |
| ---: | ---: | ---: | ---: | --- |
| 62 / 64 | 7.158 s | 5.351 s | 256 | Astronaut and glass lighthouse recognizable; different details |
| 120 / 128 | 7.615 s | 5.791 s | 256 | Astronaut, glass lighthouse, paper and distant vessel preserved |

Both routes used the same library and prompt/seed within each row, but these
are **not fresh-process ABBA speed qualifications**. The Core ML bucket
remained 1,024 image rows at 64/128 caption rows, and both hybrid PNGs were
identical across their cold and warm requests. The prompt-matched GPU images
were also visually inspected. Raw reports/PNGs:
`outputs/z-w8a8-image-caption-rows-{baseline-foxlong,hybrid-long64,baseline-long128,hybrid-long128,short32}-20260925/`.
At this initial checkpoint, 256/512/1,024 caption rows had not been measured;
supporting their shapes was a guarded experimental geometry, not a performance
assertion. Subsequent geometry results follow below.

Follow-up **fresh-process ABBA** (baseline/hybrid/hybrid/baseline), one
excluded cold and one warm request per process, used the same experimental
library SHA256 `d986ad841bb57d82f76472a31894d5375d0bd0e349b9792d16df1ddfbe14c3ed`
and the same source prompt/seed and compiled 5,120-channel manifest in each
pair. The benchmark then gained an explicit `--hybrid-image-only` flag: only its
hybrid arm enabled the two research presence switches, while the baseline
arm retains unmodified BF16 GPU settings. The M4 Max baseline health anchor
passed in both schedules:

| Real tokens / caption rows | Warm BF16 GPU denoise median | Warm hybrid denoise median | Denoise speedup | Request-wall speedup |
| ---: | ---: | ---: | ---: | ---: |
| 62 / 64 | 7.166 s | 5.314 s | 1.349× | 1.333× |
| 120 / 128 | 7.614 s | 5.794 s | 1.314× | 1.301× |

Each arm has only **two warm samples**, not a broad speed distribution.
The hybrid request in each process records 256 Core ML calls for 32 FFN
blocks × eight steps; the source 1,024-image-row ANE bucket is unchanged.
For both prompts, same-arm warm PNG SHA256 values match across fresh
processes, and the paired GPU/hybrid images visibly retain the orange
astronaut and transparent glass lighthouse, with some differing details.
Reports and PNGs are in `outputs/z-w8a8-image-caption-rows-abba-long{64,128}-20260925/`.

Additional **geometry-only**, single-process cold/warm hybrid requests used
repeated prompt sentences to hit padded caption rows 256, 512, 928 and 1,024
(243, 494, 926 and 1,000 actual text tokens). Every request finished its
256 FFN calls and wrote a valid 512² PNG, including the maximum 1,024-caption
case. These artificial repeats are *not* quality qualification or matched
performance measurements. In particular, the 512-row probe's warm denoise
was **11.779 s** versus its cold denoise **8.712 s**, so it must not be used
to assert a stable speedup. Receipts are in
`outputs/z-w8a8-image-caption-rows-geometry-{exact256,512,near1024,1024}-20260925/`.
The 64/128-row ABBA alone does not establish a win at 256–1,024 caption rows.

A separate synthetic 243-token/256-caption-row fresh-process ABBA, with
one excluded cold and one warm request in each baseline/hybrid/hybrid/baseline
process, gave **8.448 s BF16 GPU vs 6.780 s hybrid** median warm denoise
(**1.246×**, request wall **1.239×**). Each hybrid warm process accumulated
512 successful calls including its cold request; same-arm PNGs agreed across
fresh processes. This two-warm-sample result is a shape/performance screen,
not a natural-prompt quality or broad speed qualification. Its >128-token
length correctly receives `not_applicable` from the short-prompt health
anchor; raw reports: `outputs/z-w8a8-image-caption-rows-abba-256-20260925/`.

The synthetic 494-token/512-caption-row prompt also had a separate
fresh-process BF16-GPU/hybrid/hybrid/BF16-GPU ABBA: two warm requests per arm
gave **10.050 s GPU vs 9.967 s hybrid** denoise medians, only **1.008×**;
hybrid individual samples were 10.477 and 9.458 s, so this is **not** a
stable qualified gain. Both routes reproduced the same-arm PNG across
processes, and every hybrid request had 256 calls. The saved summary's short-
caption 9-second system-health gate labels this long-caption baseline
`invalid_for_performance`; that gate is inapplicable here, not evidence of a
degraded machine. The harness now reports `not_applicable` for >128 actual
text tokens. Raw report: `outputs/z-w8a8-image-caption-rows-abba-512-20260925/`.

Matched-shape **instrumented** hybrid requests explain why the long prompt
loses most of its advantage; these are diagnostic scheduling points, not
the uninstrumented ABBA latency:

| Caption rows | Warm pre-FFN/pack sum | Core ML sum | GPU tail wait sum | Parallel branch-window sum | Median GPU tail wait/call |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.020 s | 2.387 s | 0.324 s | 2.728 s | 1.377 ms |
| 512 | 4.405 s | 2.594 s | 1.915 s | 4.527 s | 7.581 ms |

Both include exactly 256 calls and keep the fixed 1,024-image-row ANE
artifact; 240 main-block FFNs see 1,152 versus 1,536 full input rows,
while 16 noise-refiner FFNs remain image-only. The longer complete BF16 GPU
caption FFN, plus other row-dependent GPU work before the fork, dominates
the change. Making the W8A8 image branch faster alone cannot remove the
~7.6 ms median GPU tail at 512 caption rows. Traces:
`outputs/z-w8a8-image-caption-rows-profile-{128,512}-20260925.jsonl`.

A separate, now-**reverted** compiled post-merge probe omitted the padded
zero ANE caption rows and concatenated the BF16 GPU caption FFN after the
image-row add. Its fox/42 four-process control/compact/compact/control
crossover generated byte-identical PNGs but shifted median warm denoise
only `5.283 → 5.261 s`. On the independent glass-lighthouse/187 prompt,
the compact request instead took `5.304 s` against control `5.259 s`;
again the PNG hashes matched. The small inconsistent timing did not
justify an extra runtime path, so only evidence remains under
`outputs/z-w8a8-a5120-compact-post-{abba,lighthouse187}-20260925/`.

### Rejected fused-caption GPU branch probe

For the 494-token/512-caption-row case, an opt-in isolated binary substituted
the existing BF16 Metal SwiGLU GEMM for the GPU caption branch's ordinary
`z_ffn`. It left the 5,120-channel image GPU/ANE partition unchanged.
Single-process warm requests initially suggested 11.972 → 8.809 s denoise,
with identical PNGs, but the **same-library** four-process
control/fused/fused/control crossover contradicted that inference:

| GPU caption branch | Warm denoise samples | Median | Warm request-wall median |
| --- | ---: | ---: | ---: |
| Original BF16 `z_ffn` | 10.220 / 8.925 s | 9.573 s | 9.830 s |
| Metal SwiGLU candidate | 10.733 / 8.794 s | 9.764 s | 10.017 s |

All four warm PNGs are bitwise identical, all requests executed 256 Core ML
calls per image and used isolated library SHA256
`a126382013ed245848fee8a806a164f4ff96b01c0cc51b4ce8656b110158c2df`.
Both arms drifted substantially, so the fused caption path has **no verified
benefit** and its source/runtime flag and dedicated ABBA CLI were removed.
The isolated binary and reports remain only as rejected experimental receipts
under `build/native-w8a8-caption-mpp/` and
`outputs/z-w8a8-caption-mpp-512-{a-first,b-first,abba}-20260925/`.

### Long-caption single-block partition screen

To test whether token-row splitting or quantizing **the caption as well as the
image** might recover the lost 512-caption-row speed, a capture-only block
filter (`TURBOCIDER_Z_FFN_CAPTURE_BLOCK=18`, effective only with an FFN
capture directory) was added. One BF16 GPU request with 494 real text tokens
and 512 padded caption rows wrote exactly eight block-18 captures of
`[1536,3840]` FP16, ~94 MB in total, not the other 31 blocks. An independent
watercolor prompt with 483 real tokens produced eight further block-18
captures of the same shape. Both source sets are in
`outputs/z-w8a8-longcaption-{row-block18-capture,watercolor-heldout-block18}-20260925/capture/`.

The single-block benchmark now takes captures of 1,056...2,048 total rows
and accepts either the existing full-width ANE **row** probe or an explicitly
selected, 1,536-row ANE-prefix/**GPU-suffix channel** probe. It rejects a
wrong block, bucket, region or split identity before model loading. The two
new single-block W8A8 channel artifacts use 3,072 or 4,096 intermediate
channels on ANE for **both** 1,024 image and 512 caption rows; both were
calibrated only on the repeated-prompt capture. The watercolor captures were
held out from export. Input/weights are resident and the joined output is
measured, but there is no native bridge, 32-layer installation, full-image
test or physical ANE instruction trace.

| Block-18 split on 1,536 rows | Repeated-prompt BF16 full / joined median | Held-out watercolor BF16 full / joined median | Held-out image / caption relative L2 |
| --- | ---: | ---: | ---: |
| ANE 512 **full-width rows**, GPU other 1,024 rows | 25.17 / **17.04 ms** | 25.25 / **17.11 ms** | 0.0409 / 0 (GPU caption) |
| ANE 3,072 **channels, all rows**, GPU other 7,168 channels | 25.16 / **17.87 ms** | 25.14 / **17.81 ms** | 0.0318 / 0.00569 |
| ANE 4,096 **channels, all rows**, GPU other 6,144 channels | 25.13 / **20.14 ms** | 25.17 / **20.06 ms** | 0.0484 / 0.02450 |
| Existing ANE 5,120 **channels, image-only 1,024 rows**, GPU suffix image + full BF16 caption | 25.10 / **17.47 ms** | 25.13 / **17.31 ms** | 0.0435 / 0 (GPU caption) |

Three real timesteps of the repeated prompt gave 512-row parallel
windows **17.04, 17.15, 17.05 ms** and 416-row windows
**19.10, 19.29, 19.24 ms**. A subsequent **matched-harness** screen loaded
the existing 32-block image-only source artifact but called only block 18;
it reused the *same captures, original BF16 weights, warm schedule and
GPU/ANE join boundary* as the row probes. Its windows were
**17.47, 17.44, 17.44 ms** on those three repeated-prompt timesteps and
**17.31 ms** on held-out watercolor. The best row probe was thus only
~**0.2–0.4 ms/call** ahead **before** its additional native bridge/scatter
and full-width ANE residency, not an e2e win. Its held-out block-18 image
L2 (`0.0409`) was close to the current artifact (`0.0435`), but cannot
establish generated-image quality. The 3,072/4,096 all-row channel pilots are
slower than the best measured row-only probe **before** paying an integrated
bridge. The row-only probe's ~17.1 ms pre-bridge window is near the current
5,120-channel image-only route's **separately instrumented native**
17.3 ms/window on a 512-caption-row prompt; the Python head-to-head is
closer matched but still excludes native work. There is no evidence that a
full row-only integration will win. Unlike
the current route it also needs full 10,240-channel ANE weights and has no
end-to-end visual qualification. We do **not** export 32-layer all-row or
row-only models on these single-block receipts. Individual JSONs are under
`outputs/z-w8a8-{row-split,channel-split}-block18-long512-*-20260925.json`
and `outputs/z-w8a8-longcaption-watercolor-block18-*-20260925.json` plus
`outputs/z-w8a8-imageonly-channel-block18-long512-*-20260925.json`;
single-block sources are under
`models/coreml/z_image_w8a8_longcaption_r1536_a{3072,4096}_block18_probe/`.

### Additional held-out visual pairs and physical-trace availability (September 25, 2026)

Two more 512-square/eight-step BF16-GPU versus image-only 5,120-channel
W8A8-ANE/BF16-GPU pairs used the same isolated native binary (SHA256
`d986ad841bb57d82f76472a31894d5375d0bd0e349b9792d16df1ddfbe14c3ed`),
compiled manifest and prompt/seed within each pair. Each arm ran one excluded
cold and **one** warm request; these are visual screens, not qualified ABBA
speed distributions. Cold and warm PNG hashes matched within each arm. Each
hybrid process recorded 512 successful Core ML calls, i.e. 256 per image
across all 32 FFN blocks, with no runtime failures.

| Prompt / seed | BF16 GPU / hybrid warm denoise | Side-by-side observation |
| --- | ---: | --- |
| Woman holding a transparent water glass, 812 | 7.172 / 5.321 s | Both preserve a plausible hand, individual fingers, transparent glass and waterline. Finger placement and small details within the glass differ; neither image proves the requested **five** fingers are anatomically accurate. |
| Minimal `OPEN 24/7` poster with red umbrella, 445 | 7.178 / 5.326 s | Both spell `OPEN 24/7` legibly in black above one red umbrella. Umbrella canopy and ribs differ, without an obvious prompt-level failure. |

The baseline health anchor passed on both requests. Unit tests involving
Core ML briefly overlapped the *first pair's baseline process*, so even the
single warm speed ratio should not be treated as an uncontaminated performance
qualification. Reports and PNGs are under
`outputs/z-w8a8-a5120-quality-{glasshand812,open247-445}-20260925/`.
These additional usable pairs still do not bound the failure rate on unknown
prompts, settle anatomical/detail acceptance, or demonstrate physical ANE
instruction execution. The machine's active developer directory is
`/Library/Developer/CommandLineTools`; there is no installed Xcode.app in
`/Applications` and `xcrun --find xctrace` cannot find the profiler. Core ML
`cpuAndNeuralEngine`, the preferred-device compute plan and successful
prediction counts remain *software-route* evidence; physical ANE residency
is still `unknown` until a suitable hardware trace is available.

### 544-row follow-up to the long-caption row-split screen

The 512-row full-width ANE branch on a 1,536-row long-caption block-18
capture took about 14.8 ms, whereas the GPU's remaining 1,024 rows took
about 17.0 ms. To test whether a smaller imbalance improves the joined FFN,
the guarded **single-block probe only** now also admits a 544-row model.
Its 16 real calibration samples and SmoothQuant recipe match the previous
512-row export (source digest
`f95ff8f3cfa72f728f85e1ef847d16949835b85333e17af39f62dad5cf7de9d4`);
the independent watercolor prompt was held out. All models remain
full-width 10,240-channel W8A8; the native runtime still rejects them.

| Captured block 18, 1,536 rows | 512 ANE rows / GPU 1,024 | 544 ANE rows / GPU 992 |
| --- | ---: | ---: |
| Repeated-prompt joined FFN | 17.04 ms | **16.82 ms** |
| Repeated-prompt GPU / Core ML alone | 17.01 / 14.82 ms | 16.77 / 16.44 ms |
| Repeated-prompt image-row relative L2 | 0.0722 | 0.0850 |
| Held-out watercolor joined FFN | 17.11 ms | **16.88 ms** |
| Held-out GPU / Core ML alone | 16.98 / 14.85 ms | 16.81 / 16.50 ms |
| Held-out image-row relative L2 | 0.0409 | 0.0529 |

The row boundary has moved toward balance but gained only **0.22–0.23 ms**
per captured FFN against the 512-row probe, before native GPU→Core ML
transport, output scatter, all-layer residence and image-level quality.
An additional same-held-out-capture **512→544→544→512** crossover ran
12 symmetric warm rounds per arm in separate processes; the parallel-window
medians were **17.139 / 16.884 / 16.837 / 17.119 ms** respectively. The
512-row average of arm medians was 17.129 ms and the 544-row average was
16.861 ms, a **0.268 ms/FFN** pre-bridge difference in this narrow screen.
This repeats the small directional advantage but is not an end-to-end
performance or multi-layer quality qualification. Raw reports:
`outputs/z-w8a8-longcaption-watercolor-row512-vs544-abba-{0-512,1-544,2-544,3-512}-20260925.json`.
Even multiplying 0.268 ms by all 256 FFN calls of an eight-step image gives
only ~0.069 s as a *hypothetical* scale of the 512→544 micro-gain; the first
two blocks see different row shapes and no native bridges are included, so
this is not an end-to-end prediction.
Against the previously measured 5,120-channel image-only *channel* split
(17.47 / 17.31 ms on the respective captures), the 544-row gap is
0.43–0.65 ms in this probe; still not proof of an end-to-end win. Increasing
ANE rows also quantizes more image tokens and worsened both measured local
image-row errors. This is insufficient evidence to export 32 row-split
blocks or replace the current image-only channel route. Source artifact:
`models/coreml/z_image_w8a8_fullwidth_rows544_block18_probe/`; raw reports:
`outputs/z-w8a8-row-split-block18-long512-544-20260925.json` and
`outputs/z-w8a8-longcaption-watercolor-block18-rows544-20260925.json`.
No physical ANE execution trace was available. `powermetrics` can sample
ANE rail estimates but requires superuser access here, and noninteractive
`sudo -n` reports that a password is required; rail power would not itself
prove which FFN operators executed on the ANE.

### Two-group hidden-A8 graph measured at the complete FFN fork/join boundary

The existing image-only, 5,120-channel **two-group** single-block W8A8
models were screened against the current four-group 32-block source
artifact. Both use the same BF16 checkpoint, image-only 1,024-row bucket,
same 16 calibration samples/digest, same original BF16 GPU 5,120-channel
complement and full BF16 GPU caption FFN. The single-block harness now has
an explicit `--two-group-image-probe` identity gate: only block 18 or the
paired block 3/29 experimental manifests pass, and no two-group artifact is
presently installed as a native 32-block route. This measures the entire GPU+
Core ML FFN join with resident inputs, **not** native GPU→Core ML bridge,
all-layer residency or image-generation quality.

| Held-out lighthouse capture / block | Four-group parallel | Two-group parallel | Four-group / two-group image-row relative L2 vs BF16 GPU |
| --- | ---: | ---: | ---: |
| Block 18, same-input four-process c4/c2/c2/c4 | 14.154 / 14.070 ms | 13.810 / 13.823 ms | 0.05098 / 0.05093 |
| Block 3, one warm screen per model | 14.069 ms | 13.729 ms | 0.00941 / 0.00959 |
| Block 29, one warm screen per model | 14.095 ms | 13.724 ms | 0.07357 / 0.07373 |

The block-18 pair of paired-arm medians improves about **0.295 ms/FFN**;
the two other blocks improve about **0.34–0.37 ms/FFN**. The GPU-only
complement was around 9.6–9.9 ms while Core ML prediction was around
13.9–14.3 ms, so these probes were ANE-bound. Their one-prompt *full
joined* FFN relative L2 changes are small, but the two-group ANE partial
outputs differed noticeably from the four-group graph on other held-outs;
neither local metric qualifies final image quality. In the actual current
instrumented 32-block/short-prompt route, median GPU tail wait is about
0.174 ms after Core ML returns, so an ANE-only 0.3 ms improvement need not
survive wholly in the native join. Even an optimistic 0.3 ms × 256 calls
is **0.077 s** per eight-step image, not a measured end-to-end gain.
Do not spend another ~1.8 GiB source plus ~1.8 GiB compiled model on a
32-block candidate without native critical-path evidence and a stronger
quality screen. Raw single-block records are under
`outputs/z-w8a8-block{18,3,29}-a5120-c4-vs-c2-*-20260925.json`;
the two-group sources already existed under
`models/coreml/z_image_w8a8_imageonly_hidden_c2_a5120_b1024_{b18,b3_29}/`.

### Same compiled production-candidate graph: CPU-only versus ANE-preferred

The `benchmark_z_image_compute_units.py` probe now accepts an explicitly
marked 1,024-image-row W8A8 manifest as well as its previous 1,056-row
caption-inclusive models. It trims only the last 32 conditioning rows from
an independent, previously captured block-18 FFN input; the compiled model,
weights and shape are **identical** between `CPU_ONLY` and `CPU_AND_NE`.
Twelve warmed symmetric ABBA rounds on the actual 5,120-channel image-only
compiled artifact gave **27.720 ms CPU-only versus 14.405 ms CPU+NE**, a
**1.924×** difference for the exact graph used by the current native route.
Both outputs were finite, but their image-row relative L2 was `0.03775`;
CPU-only is *not* a bitwise oracle for the ANE-preferred image result.
The report with artifact/input SHA-256 is
`outputs/z-w8a8-imageonly-a5120-block18-compute-units-abba-20260925.json`.

This is stronger placement evidence than a compute-plan preference alone:
the **same artifact and input** behave very differently when NE is an
eligible compute unit. It still does not trace the executing hardware,
pin down individual INT8 MAC operators, or establish simultaneous physical
GPU+ANE overlap. Those fields remain unverified; no automatic deployment
policy is changed by this diagnostic.

### Further paired visual stress cases (September 25, 2026)

Three independent 512-square/eight-step seeds broadened the screen to bicycle
geometry, repeated architectural structure and two exact lines of lettering.
Each BF16 GPU/image-only 5,120-channel W8A8 hybrid pair used the same prompt,
seed and isolated native binary SHA256
`d986ad841bb57d82f76472a31894d5375d0bd0e349b9792d16df1ddfbe14c3ed`.
One cold request and one warm request per arm yielded identical PNG hashes
within that arm; each hybrid warm report accumulated 512 successful Core ML
calls, i.e. 256 per image. All GPU baselines passed the short-prompt health
anchor. These timings are **one warm point per arm**, not performance ABBA.

| Prompt / seed | BF16 GPU / hybrid warm denoise | Paired 512² visual finding |
| --- | ---: | --- |
| Yellow-raincoat cyclist / 209 | 7.164 / 5.336 s | Both show two entire wheels, a rider, wet cobblestones and reflections. The frame, pose and details differ; no obvious topology collapse. |
| Symmetrical concrete museum / 663 | 7.160 / 5.323 s | Both show regular rows of arches and centered steps; subtle differences in entryway and symmetry. |
| Two-line railway board / 331 | 7.158 / 5.325 s | BF16 renders `PARIX 08:30` / `ROME 09:15`; the hybrid correctly renders `PARIS 08:30` / `ROME 09:15`. This does **not** establish that A8 generally improves spelling. |

I inspected both 512² warm PNGs for each seed. None of these three shows a
catastrophic visual failure; together with the prior seed variations they
support the user's visual-first criterion on the inspected set, not a bound
on untested prompts. Records and PNGs are in
`outputs/z-w8a8-a5120-quality-{cyclist209,arches663,board331}-20260925/`.

### Explicit manifest-driven 512² W8A8/BF16 runtime without research switches

The fastest existing 32-FFN image-only W8A8+BF16-GPU path previously needed
two process environment variables in addition to an explicit `gpu_ane`
request. The native runtime now reads the manifest's marked
`image_only_token_rows=1024` identity, requires
`allow_approximation=true`, fixed 1,024-row INT8 shape, 512² input, a
32...1,024-row caption bucket and full BF16 GPU caption weights, then
selects the BF16 GPU complement and compiled GPU pre/post segments directly.
The `HybridSession` still verifies all 32 compiled branches, BF16 checkpoint
path/size/SHA, output geometry and source identities before inference. An
unmarked artifact retains its previous semantics. The production `auto`
table remains untouched; this path needs an explicit manifest and explicit
`gpu_ane` selection. The original two flags still reproduce the same route.

An isolated native build at
`build/native-w8a8-marked-native/libturbocider.dylib` (SHA256
`35a08e9a9bdaf19e555a8540b69fc04dafce0a5a66a04470636042b62683aff4`)
ran fox/42, 512², eight steps with the unchanged compiled image-only
manifest. The no-flags and two-old-flags warm PNGs were byte-identical
(SHA256 `7109f36e7b110c70125f5e9a39d54aa2c2bb9cf05ba5e46d443854625a6ab33a`);
each request reported all 256 FFN Core ML calls, zero output-copy bytes and
zero runtime failures. One warm point was 5.294 s without flags versus
5.264 s with old flags; this point is for correctness, **not** a speed
ranking. The new benchmark `--hybrid-image-only-marked` mode sets no
`TURBOCIDER_Z_*` variables in either arm and records the manifest-driven
selection distinctly from the historical flag mode.

The same new library then ran a fresh-process BF16-GPU / marked-hybrid /
marked-hybrid / BF16-GPU crossover, one excluded cold and one warm request
per process. Warm denoise medians were **6.8245 versus 5.2655 s**
(**1.296×**); warm request wall **7.0693 versus 5.5144 s**
(**1.282×**). Both same-route PNG hashes matched across processes; the
short-prompt GPU health anchor passed. The entire schedule and rejection
reports live in `outputs/z-w8a8-manifest-explicit-optin-20260925/`.
Negative controls: omitting approximation rejected the request with
`gpu_ane requires allow_approximation=true`; providing the same manifest
to `auto` at 512² fell back to BF16 GPU with identical baseline PNG and
no hybrid calls; requesting 768² rejected the image-only shape. These
checks verify the opt-in boundary, not general safety across all prompts
or physical ANE arithmetic. The default native library and production
automatic policy were not replaced.

After hardening the manifest preflight to reject flexible input modes on a
marked image-only artifact, the **final** isolated native library SHA256 is
`e60abc0188017a9ec3f6f4444b06de4839dbbb49f098d59e91a7b55d6764b92b`.
The same fox/42 BF16 / marked-hybrid / marked-hybrid / BF16 schedule under
this final binary measured **6.8238 / 5.2651 s** median warm denoise
(**1.296×**) and **7.0695 / 5.5106 s** warm request wall (**1.283×**).
Each route again produced identical PNG hashes across its two processes,
the hybrid reported `gpu_bf16+coreml_w8a8+image_only_ane_gpu_bf16_caption`,
512 cumulative predictions per cold+warm process, zero output-copy bytes,
and the GPU health anchor passed. The final-binary report is
`outputs/z-w8a8-manifest-explicit-optin-20260925/abba-fox42-final/`;
the preceding `abba-fox42/` and `new1/oldflags/` records are from the
earlier isolated binary and are not mixed into this final speed ratio.

The final binary also ran the **legacy W8A16/4096-channel** fox/42 manifest
without a regression: its single warm denoise was 5.59947 s, with 256 Core ML
calls per image. The warm PNG SHA256
`7b41612e406ced97070fa4e55303b4ce0e59b552c1b732064752c4e1d8b42a60`
matches both warm images in the earlier `z-w8a16-vs-w8a8-bf16gpu-20260925`
and `z-w8a16-vs-w8a8-a4608-20260925` W8A16 crossover arms. This is a
same-input **functional regression check**; the one warm W8A16 point is not
paired with the final-binary BF16/W8A8 ABBA for a qualified speed ranking.
The final isolated dylib was rehashed as
`e60abc0188017a9ec3f6f4444b06de4839dbbb49f098d59e91a7b55d6764b92b`.

### Row partition decision against the current integrated path

The current image-only branch partitions the **10,240 intermediate FFN
channels**, assigning 5,120 to W8A8 Core ML and 5,120 to BF16 GPU for each
image row; the full caption FFN stays on GPU. It does not split the 3,840
input embedding dimension. A pure row partition would give disjoint complete
FFN rows to GPU and Core ML and concatenate their outputs, eliminating the
partial-down sum but requiring full-width Core ML weights. Both designs
still need the preceding GPU attention/norm to finish before the FFN input
is ready; row splitting the FFN alone does not provide attention/FFN
producer overlap. Splash's fully measured prefill path also partitions FFN
channels; its separate layer-0 row probe did not replace the full-model
channel path (`../splash/notes/2026-09-21-ane-sequence-row-split.md`).

For 1,056 captured rows, the best measured full-width row partition was
ANE 416 / GPU 640: **17.95 -> 11.37 ms** full BF16 versus joined single
FFN, before native transport/scatter. For 1,536 captured rows, ANE 544 /
GPU 992 measured **25.14 -> 16.82 ms** on the calibration-prompt input
and **25.04 -> 16.88 ms** on held-out watercolor. Under the same
single-block harness and captures, the existing 1,024-image-row/5,120-channel
partition joined in **17.47 / 17.31 ms**, respectively. Thus the row probe
leads only **0.65 / 0.43 ms per FFN before native overhead** on these two
long-caption captures; the 512-to-544-row improvement itself was 0.268 ms
in a separate alternating crossover. Multiplying the latter by 256 calls
would yield only 0.069 s as a hypothetical upper-bound *scale*, not a
measured denoising gain. The 32-block row branch is not exported, installed
or image-quality tested. It would approximately double the bare W8 ANE
weight bytes relative to the current half-width branch. Static Core ML
row buckets and their calibration further limit how freely a request could
change the partition. The next legitimate comparison is a native single-
block fork/join with GPU-to-Core-ML transport and output placement included;
promoting full-width row artifacts from Python timings alone is not justified.

On this checkout, `python -m unittest discover -s tests/native -p
'test_z_image*.py' -v` finished with **80 tests, 14 explicitly opt-in Metal
tests skipped, zero failures**; `git diff --check` passed. Neither proves
ANE physical INT8 execution, persistent-session route switches, broad
cross-prompt visual quality or a production `auto` policy. These remain
open before the full optimization goal can be considered qualified.

### Persistent-session GPU/hybrid route switching

The benchmark now has a correctness-only `--probe-route-switch` worker mode:
one engine receives GPU, explicit marked-image-only W8A8/BF16 hybrid, GPU,
and the same hybrid again, without process environment overrides. It refuses
unrelated shapes, `auto`, parity, or missing approximation consent; it checks
that every hybrid visit executes all `32 × steps` FFN predictions with no
reported Core ML output copy, that the GPU visits have no hybrid session, and
that the two PNG hashes per route match. All four requests are explicitly
marked warmup/diagnostic and **cannot enter a speedup aggregate**.

The final isolated dylib (`e60abc0188017a9ec3f6f4444b06de4839dbbb49f098d59e91a7b55d6764b92b`)
passed the fox/42, 512², eight-step sequence in one process. GPU image
SHA256 was `3410c09300a9425d0f43afc29bb430f1a0ff37d904c6edd5508887fbea1f99c6`
both times; hybrid image SHA256 was
`7109f36e7b110c70125f5e9a39d54aa2c2bb9cf05ba5e46d443854625a6ab33a`
both times. Each hybrid visit recorded 256 runtime FFN calls and zero output
copies. Report/PNGs: `outputs/z-w8a8-persistent-switch-fox42-20260925/`.
The request wall times include initial model setup or reloading and are not
used to compare GPU versus ANE performance. This removes the specific
same-session GPU ↔ marked-image-only correctness gap for this prompt/shape;
it does not prove arbitrary manifest transitions, broad image quality or
physical ANE instruction scheduling.

### Short-row fused GPU QKV before the hybrid FFN (September 25, 2026)

The marked image-only W8A8 route previously used a conventional BF16 GPU
matmul for QKV followed by Q/K normalization, RoPE and layout conversion in
its compiled pre-Core-ML graph. The GPU-only path already uses a fused
projection/QKV-prepare Metal kernel for the qualified M4 Max short-row
shape. The hybrid pre-graph now selects that same kernel **only** for the
explicit 1,024-image-row W8A8 manifest on a qualified M4 Max when the
current block has at most 1,056 rows. Other hardware and longer main-block
captions retain the original pre-graph; the two 1,024-row noise-refiner
blocks still qualify on long-caption requests. The production `auto` route
is not changed. `TURBOCIDER_Z_HYBRID_DISABLE_FUSED_QKV=1` is an opt-out;
`TURBOCIDER_Z_HYBRID_FUSED_QKV=1` remains an explicit research override.
Both variants use the same BF16 GPU FFN complement, 32 W8A8 Core ML
branches, output scaling and manifest. No new Core ML weights are needed.

An initial isolated candidate crashed at **process exit** after writing
both PNGs. The macOS crash report showed static MLX compiled-closure
destruction accessing a finalized compiler cache, not a fault inside the
inference calls. Lazy graph construction alone did not solve it; the two
bounded compiled graph objects now have process lifetime and are not
destroyed during DSO teardown. A fresh baseline worker and all subsequent
fresh-process crossover workers exited with status zero. Timings from the
crashed worker are **discarded**. This lifetime fix also covers the old
non-fused hybrid compiled graph when it is selected.

On the exploratory, opt-in isolated binary SHA256
`857cefd29d8ef560109c2038953705d887e8985ccae92c26776e6604f64f6101`,
fresh-process old-hybrid/QKV/QKV/old-hybrid schedules with one excluded cold
and one warm request per process measured:

| 512² prompt/seed | Old hybrid warm denoise | Fused-QKV warm denoise | Difference |
| --- | ---: | ---: | ---: |
| Fox / 42 | 5.2672 s | 5.2059 s | 0.0613 s (1.012×) |
| Glass lighthouse / 187 | 5.2592 s | 5.2028 s | 0.0565 s (1.011×) |

Each of the four warm PNGs within each schedule had the **same SHA256**,
despite switching GPU pre-graphs. Both controls were explicit 32-block W8A8
hybrids, not GPU-only. Raw reports:
`outputs/z-w8a8-hybrid-pre-qkv-abba-{stable-fox42,lighthouse187}-20260925/`.
Two warm points per arm support a small directional benefit on these two
prompts, not a general distribution or arbitrary-hardware qualification.

The final default-on **isolated** library SHA256 is
`09066cf41b8cfb4e500a2cc608371e5199a45ffa50a76cd6ea57099e74295dd4`.
Under this same binary, explicit opt-out/hybrid-default/hybrid-default/opt-out
on fox/42 measured **5.2585 → 5.1910 s** warm denoise and
**5.5061 → 5.4360 s** warm wall. All four warm PNG hashes equaled the old
image-only hybrid (`7109f36e7b110c70125f5e9a39d54aa2c2bb9cf05ba5e46d443854625a6ab33a`). A
separate no-experiment-switch worker exited normally, ran all 256 FFNs per
image, and produced the same PNG. An uncontaminated BF16-GPU versus
explicit marked hybrid ABBA **with no process research switches** and the
same final library measured **6.8330 → 5.2004 s** median warm denoise
(**1.314×**) and **7.0771 → 5.4439 s** warm request wall (**1.300×**).
The same-route PNGs agreed across fresh processes, and the GPU health
anchor passed. Reports are under
`outputs/z-w8a8-hybrid-pre-qkv-default-{abba-fox42,vs-gpu-fox42}-20260925/`
and `outputs/z-w8a8-hybrid-pre-qkv-marked-no-flags-fox42-20260925/`.
Do **not** combine the older dylib's 1.296× speed ratio with this final
library's medians. This is a measured optimization of the explicit
GPU/ANE route; it does not trace physical ANE MACs or qualify broad prompt
quality and automatic deployment. Z-Image-related Python tests: 85 run,
14 optional Metal tests skipped, zero failures, with `git diff --check` clean.

A 120-real-token/128-caption-row astronaut+lighthouse follow-up used the
final no-flag explicit hybrid route (one cold, one warm request). Its warm
image SHA256 `adea590d64bbece182fe17f71a5d9e82e9e3afa58ef32151dd1041b99c1ae383`
exactly matches the earlier unmodified hybrid image for the same prompt and
seed 187; it again completed 256 FFNs per image and had no runtime failure.
Warm denoise was 5.818 s, **not** a paired long-caption speed qualification.
The 128-caption-row main blocks use the original QKV pre-graph; this request
checks that the short-row noise refiner eligibility did not alter the image.
Receipt: `outputs/z-w8a8-hybrid-pre-qkv-long128-noflags-20260925/`.

The final default-on isolated library also passed the one-engine
GPU→hybrid→GPU→hybrid switch probe: matching GPU PNGs, matching hybrid PNGs,
256 complete W8A8 FFN calls per hybrid request, and zero reported output
copies. This remains a correctness check, not a speed measurement. Receipt:
`outputs/z-w8a8-hybrid-pre-qkv-default-route-switch-20260925/`.

### Qualified-shape fused gate/FFN-input norm (September 25, 2026)

The GPU-only block also has a two-reduction, BF16-boundary-preserving
`gate_norm_virtual` Metal kernel. An isolated hybrid pre-graph reuses it to
produce both the attention residual and normalized/modulated FFN input,
replacing two separate `fast::rms_norm` expressions. It leaves the BF16 GPU
FFN complement and all 32 Core ML W8A8 branches unchanged. The same
hardware/shape gate as fused QKV applies: marked image-only explicit
manifest, qualified M4 Max and at most 1,056 input rows per block. The
128-caption-row main blocks keep the previous pre-graph. The explicit
research override is `TURBOCIDER_Z_HYBRID_FUSED_GATE_NORM=1`; the opt-out
for a paired control is `TURBOCIDER_Z_HYBRID_DISABLE_FUSED_GATE_NORM=1`.
No default `auto`-selection policy or model artifact changed.

With QKV fusion already present in **both** arms, one excluded cold plus one
warm request per fresh process in old-gate/new-gate/new-gate/old-gate order
measured on isolated library SHA256
`e5a17dd4a713b0d3f6358b1aa36f182a08250f2d921b35a8b9b83a986b55576b`:

| 512² prompt/seed | QKV-only hybrid warm denoise | QKV+gate/norm hybrid warm denoise | Difference |
| --- | ---: | ---: | ---: |
| Fox / 42 | 5.1926 s | 5.1420 s | 0.0506 s (1.010×) |
| Glass lighthouse / 187 | 5.2023 s | 5.1514 s | 0.0509 s (1.010×) |

The four warm PNGs in each schedule were byte-identical; both same-route
processes completed normally. Raw reports:
`outputs/z-w8a8-hybrid-gate-norm-abba-{fox42,lighthouse187}-20260925/`.
These are two warm samples per arm, not a broad timing distribution.

The final default-on **isolated** library SHA256 is
`c024767fe6eb8620bcb0600ac37fc72f0f3daf0d85c50fefd6b4f72da0f62653`.
Under this same binary, opt-out/new/new/opt-out measured **5.2066 →
5.1480 s** warm fox/42 denoise with matching PNGs. A fresh-process BF16
GPU / marked-hybrid / marked-hybrid / BF16 GPU schedule with **no research
environment switches** measured **6.8371 → 5.1428 s** median warm denoise
(**1.329×**) and **7.0803 → 5.3887 s** median warm wall (**1.314×**).
All 256 FFN calls/image used the compiled Core ML branches, and the
short-prompt GPU health anchor passed. A separate no-switch worker
reproduced the exact hybrid PNG. The single-engine GPU→hybrid→GPU→hybrid
probe passed with both same-route PNG pairs equal. The 120-token/128-caption
row request matched the earlier hybrid PNG exactly and completed its 256
Core ML FFNs. Reports:
`outputs/z-w8a8-hybrid-gate-default-{abba-fox42,vs-gpu-fox42,no-flags-fox42,route-switch,long128-no-flags}-20260925/`.
The final library has not replaced `build/native/libturbocider.dylib`.

I visually inspected the final fox/42 BF16/hybrid image pair and the
glass-lighthouse control/candidate pair. Fox composition, fur, four limbs
and tail remain plausible though detail positions differ from BF16; the
transparent lighthouse is structurally stable. The old and new hybrid
images are byte-identical on the paired prompts, so the gate/norm speed
optimization itself introduced no measured image difference. These cases
are not a broad prompt-failure-rate qualification or physical ANE INT8
trace. Z-Image-related Python regression: 86 tests run, 14 optional Metal
tests skipped, zero failures; `git diff --check` passes.

### Post-merge virtual RMSNorm probe: insufficient gain (September 25, 2026)

An isolated experimental library (`build/native-w8a8-post-probe/`, SHA256
`3114f0699344a0f5f1cd06e34ab942f204bcaed2a1df1afef56d97c3c04b0e2b`)
switched only the marked image-only, short-row hybrid's compiled post-FFN
RMSNorm/gated residual to the existing 128-physical-thread virtual Metal
kernel. It reused the already-tanhed BF16 modulation, retaining the original
BF16 merge before normalization. The default-on QKV and gate/norm pre-graph,
full 32-block W8A8 Core ML artifact, and resident BF16 GPU complement were
identical in both arms. Both arms used the *same* library, fresh processes,
one excluded cold request and one warm request per process in control /
candidate / candidate / control order, 512-square and eight steps.

| Prompt / seed | Default warm denoise median | Virtual post-norm median | Difference |
| --- | ---: | ---: | ---: |
| Fox / 42 | 5.16025 s | 5.14908 s | -0.01117 s (-0.22%) |
| Transparent glass lighthouse / 187 | 5.14583 s | 5.14066 s | -0.00517 s (-0.10%) |

The four warm PNG hashes were identical within each prompt; both ABBA runs
completed without a runtime error and the fox health anchor was healthy.
Each candidate process recorded 256 successful Core ML FFN predictions per
request, with zero reported Core ML output-copy bytes.
This is at the noise floor for whole-request qualification, so the virtual
post-norm option and its temporary benchmark switch were reverted from
source. The isolated experimental dylib and raw PNG/reports remain under
`outputs/z-w8a8-hybrid-post-virtual-abba-{fox42,lighthouse187}-20260925/`.
Do not combine this optional probe's timings with the earlier final default
library's GPU-versus-hybrid speedup. The next higher-value candidate is a
native single-block row-vs-channel fork/join with actual GPU/Core ML bridge
and output placement included, before considering 32-block row artifacts.
After the source revert, the project's Python 3.11 environment ran the
Z-Image test discovery with 86 tests, 14 opt-in Metal skips and zero
failures; `git diff --check` passed. The system Python 3.9 is too old for
the benchmark-audit `hashlib.file_digest` and lacks NumPy, so it is not a
valid interpreter for that full suite.

### Native single-block bridge probe and generic CLI manifest selection

The guarded `tools/native/z_image_row_bridge_probe.cpp` now runs block 18's
original BF16 GPU FFN and both W8A8 splits in a symmetric warm schedule
using `CoreMLPartitions`: BF16 GPU→contiguous FP16 packing, synchronous Core ML
with MLX shared output backing, asynchronous BF16 GPU branch, and the
respective output join. It uses original BF16 model weights and a real 1,536-row
saved FFN input prepared by `prepare_z_image_row_bridge_probe.py`; both Core ML
programs were already available as existing single-block row-probe and 32-block
image-only channel artifacts. One dedicated compiled 544-row model and two
236 MiB test bundles remain in `build/native-w8a8-gate-default/`. The control
library SHA256 remains
`c024767fe6eb8620bcb0600ac37fc72f0f3daf0d85c50fefd6b4f72da0f62653`.
Each input had three warmups per mode, a correctness check and 12 alternating
paired rounds (two samples/mode/round). All trials returned 28 Core ML calls
per branch and **zero reported output-copy bytes**. Model load, preceding
attention/norm and full 32-block scheduling are excluded.

| Real block-18 1,536-row input | Full BF16 GPU | 544-row ANE / 992-row GPU | Image-only 5,120-channel ANE / BF16 GPU | Image relative L2: rows / channels |
| --- | ---: | ---: | ---: | ---: |
| Independent watercolor (capture SHA `182ce0b9e433e1af…`) | 25.0956 ms | 17.1116 ms | 17.4274 ms | 0.05290 / 0.04356 |
| Repeated long caption (capture SHA `348cbbaf37302f63…`) | 25.1304 ms | 17.1244 ms | 17.4601 ms | 0.08496 / 0.06336 |

The row split leads by **0.316–0.336 ms/FFN** even after this one-block native
transport/join, less than the earlier Python pre-bridge gap. All caption rows
are exactly BF16 GPU in this probe. Both Core ML models are resident *at once*
in this diagnostic, which is not how a final 32-layer row implementation
would run. These narrow real-input receipts do not establish image quality,
an end-to-end speedup, or physical ANE INT8 MAC; keep the integrated 5,120-
channel route as the fastest qualified **full-model** candidate. The row
model also requires full 10,240-channel ANE FFN weights. Do not promote a
32-block row artifact from a ~0.3 ms single-block lead.

The native CLI now accepts a trailing `--ane-manifest MANIFEST.json` for
`plan`, `generate` and `batch`. Both request schemas are rewritten **in
memory only** to explicit `gpu_ane` with `allow_approximation=true`; a missing
file or conflicting request manifest is rejected before model loading. This
does not hard-code an ANE profile: the existing runtime still verifies the
chosen checkpoint, dimensions, block count, width, LoRA identity and compiled
artifacts. The 5,120-channel marked W8A8 profile remains explicit; the
previous `auto` policy and other ANE models are unchanged. The matching
512²/eight-step CLI smoke against the isolated fastest library generated
SHA256 `7109f36e7b110c70125f5e9a39d54aa2c2bb9cf05ba5e46d443854625a6ab33a`,
identical to the earlier tested hybrid PNG. Cold wall was ~23.65 s, not a
new warm speed qualification. CLI schema-1/schema-2/non-Z-Image request and
conflict tests live in `tests/native/test_cli_ane_override.py`; see
`docs/public/Z_IMAGE_ANE.md` for the supported invocation and speed limits.

The regular `build/native` library (SHA256
`68baf7174f43fb8b684a785e5f61ec900f807f83caba108896959080dcc1719c`)
also passed the CLI `plan` and 512²/eight-step `generate` with that explicit
compiled manifest. The CLI PNG SHA256 was again
`7109f36e7b110c70125f5e9a39d54aa2c2bb9cf05ba5e46d443854625a6ab33a`;
the single cold request reported 5.9671 s denoise and 18.9363 s request wall,
**not** a warm performance point. On this same regular-build library, a
fresh-process BF16 GPU/hybrid/hybrid/BF16 GPU crossover (one excluded cold
and one warm request per process) gave 6.8256 → 5.1466 s median warm denoise
(1.326×) and 7.0688 → 5.3934 s warm request wall (1.311×). Both GPU warm
PNGs matched SHA256 `3410c09300a9425d0f43afc29bb430f1a0ff37d904c6edd5508887fbea1f99c6`;
both hybrid PNGs matched the CLI PNG. All hybrid warm requests completed 256
Core ML calls, zero output-copy bytes and zero failures. The GPU health
anchor passed. Raw reports: `outputs/z-w8a8-final-build-abba-20260925/`.
This independently verifies that the regular CLI build retains the fastest
measured hybrid route; it still does not prove physical INT8 ANE execution or
quantify an acceptable image-quality threshold across arbitrary prompts.
The local `dist/cli` was refreshed from this regular build without replacing
`dist/TurboCider.app`; the old CLI bundle was retained as
`dist/cli-before-ane-20260925/`. After ad-hoc signing and bundle-relative
MLX library linking, the packaged CLI passed all four override contract
tests, `plan`, and a separate cold `generate`. The packaged PNG SHA256 again
matched `7109f36e7b110c70125f5e9a39d54aa2c2bb9cf05ba5e46d443854625a6ab33a`.
This package smoke checks distribution correctness, not warm speed.

Disk maintenance: seven superseded, TurboCider-managed W8A8 experiment
compilation caches were cleared after verifying that their corresponding
`_all/manifest.json` source packages remain intact: the four
`z_image_w8a8_adaptive8_shared_vector_diverse8*` caches, the routed
`z_image_w8a8_regions_image_sq_hidden_c4_routed_diverse8_compiled` cache,
and its `a2048` and `a3072` variants. The managed-cache preview listed
10.211 GiB of compiled entries, and each matching plan token was applied.
Historical compiled-manifest paths for those experiments now require
recompilation with `tools/coreml/compile_z_image_manifest.py` against the
preserved source manifest. The fastest image-only 5,120-channel cache, the
4,096-channel W8A16 cache and the 4,608-channel comparison cache were not
cleared; source `.mlpackage` directories, checkpoint weights, reports and
generated image evidence were not removed.
