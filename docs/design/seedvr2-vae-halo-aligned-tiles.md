# SeedVR2 VAE: halo-aligned rectangular Core ML tiles

## Scope (2026-10-15)

This extends the isolated original-weight causal-convolution experiment, not
the public VAE backend or App defaults. The supported experimental extents now
include62/94 as well as32/64/128, independently for rows and columns. The generic
export/native-IO fixture tools also retain their existing256 extent; the actual
causal decoder executor does not accept256.

The observed64-square output bank requires a66-square spatial input. Its actual
FP16 IOSurface row stride is96 elements, not66. Trying output widths62 and94
makes the required input widths64 and96. This is a measured-layout hypothesis,
not proof that a compiler or another device will choose the same schedule or
that reduced padding implies lower latency. Keep the height64 initially so
only the row-width tradeoff changes.

Both original layers are retained in full:

- `decoder.up_blocks.3.resnets.0.conv1`:256->128, with full-frame global
  statistics and GPU affine/SiLU/tap/halo packing (`norm`/`dual_norm`).
- `decoder.up_blocks.2.upsamplers.0.conv`:256->256, with original upstream
  projection/shuffle, temporal histories and separate FP16 bias rounding.

Initial frames retain the native FP32 folded convolution. Subsequent frames
read their two complete old/current frame owners, never independent temporal
crops. Global GroupNorm and all-spatial attention are unchanged. No new public
selector, admission reduction, <=16-GiB qualification or >121-frame support is
inferred from this experiment. Public compute units remain CPU+NE; physical ANE
occupancy and hardware overlap have not been measured.

## Actual IO and native correctness

Each bank has two nonaliasing public IOSurface slot pairs. Both slots must have
the same actual input/output strides. The executor records them, and the fresh
decoder benchmark exports them. The ABBA driver validates the selected geometry
and positive nonoverlapping row/channel/batch extents, in addition to actual
frame/prediction/backing counts. It does not assume a universal pitch formula.

| Output tile | Actual input/output row stride, FP16 elements | Hotconv two-slot bytes | Upscale two-slot bytes | Combined bytes |
| --- | --- | ---: | ---: | ---: |
|64x64|96 /64|21,626,880|23,724,032|45,350,912|
|64x62|64 /64|15,138,816|17,235,968|32,374,784|
|64x94|96 /96|22,675,456|25,821,184|48,496,640|

These are measured external IO mappings, excluded from MLX active accounting.
They are not total Core ML allocations or enforceable process RAM bounds. The
62-wide pool saves12,976,128 bytes; the94-wide pool is3,145,728 bytes larger.

At1024-square, each noninitial frame makes256/272/176 predictions per layer
for widths64/62/94. Across24 subsequent frames this is6144/6528/4224 per layer.
Thus62 trades lower surface storage for more calls, while94 trades fewer calls
for larger outputs. Border tiles still execute their entire compiled geometry,
zero padding is outside normalization, and assembly writes only valid pixels.

Both new geometries pass the actual native tests for both original layers:

- Original-coefficient FP32 convolution witnesses and signed ABA inputs.
- Ragged17x19,65x71,64-square and97x101 operands; real offset/noncontiguous
  backing, replicated history aliases and output-retention independence.
- All packed and assembled half words checked independently. Each62-wide test
  checks178,421,760 packed words; each94-wide test checks184,909,824. The128/256
  output consumers check15,619,712/31,239,424 words.
- Wrong-layer/shape/dtype rejection, pre-cancellation, cancellation after an
  actual prediction, same-runtime recovery, actual VAE sink failure, complete
  nine-frame same-VAE recovery and zero active MLX allocations after teardown.

The46 VAE CPU guards pass with the NumPy/safetensors validation interpreter.
All120 SeedVR2 CPU/contracts pass with the explicitly selected latest packaged
ordinary library. The first default-directory discovery used stale `build/native`
and failed nine existing contracts; selecting the actual resource-advice package
resolves those failures without modifying/replacing that library or the App.
The managed/coreml export interpreters skip nine reader-dependent guards; those
skips are not counted as passing tests.

## Full decoder comparison

The v2 ABBA driver supports same-route/different-bank comparisons with independent
`reference`/`candidate` labels. It binds every selected manifest and compiled
tree, original checkpoint, complete latent/pixel fixture and actual executable
before/after the cohort. Same-route identical-bank or unused-bank overrides are
rejected. Never aggregate both banks under the common `dual_norm` route label.

The reference is the already optimized dual-convolution/full-statistics norm-pack
route with64x64 banks, not the earlier full temporal concatenation or single-
convolution route. Each process performs one cold and two warm fresh complete
1024-square25-frame decodes. Every output value and every frame is checked;
first-frame words/zero Core ML calls, complete per-layer call counts and unchanged
original latent bytes are mandatory. Host pixel gates are included in timing;
model construction, bank loading, reference scans and cold calls remain separate.

The62-wide ABBA is terminal complete: all twelve fresh decodes pass every frame,
initial-frame words/zero Core ML calls and the exact6144/6528 per-layer call
counts. Four warm observations per route give:

| Complete decoder component |64x64 reference|64x62 candidate|
| --- | ---: | ---: |
|Warm median seconds, host pixel gate included|53.5753754795|54.0276752915|
|MLX active maximum bytes|11,922,148,962|11,922,148,962|
|External IO slot bytes|45,350,912|32,374,784|
|Maximum sampled process footprint, all cold/warm portions|13,620,151,448|13,606,962,376|

The candidate is about0.84% slower, with no active-maximum reduction. The measured
smaller external pool does not establish a new overall memory profile; sampled
process maxima include cache/cold portions and are not enforceable bounds. Keep
the geometry available for offline exploration, not an App acceleration/default.

The94-wide independent ABBA is also terminal complete. All twelve fresh decodes
pass the same gates, with4224 actual predictions per layer on every candidate
iteration, not6144 cached/relabelled observations:

| Complete decoder component |64x64 reference|64x94 candidate|
| --- | ---: | ---: |
|Warm median seconds, host pixel gate included|53.6053686045|53.3707115835|
|MLX active maximum bytes|11,922,148,962|11,922,148,962|
|External IO slot bytes|45,350,912|48,496,640|
|Maximum sampled process footprint, all cold/warm portions|13,553,042,608|13,556,204,696|

This is an observed0.44% lower median, a small fixture/machine-specific
difference, with a3,145,728-byte larger pool and no active-maximum improvement.
It does not establish a useful overall App acceleration, statistical portability
or a lower-memory profile. Do not add this small result to previous independent
cohorts' speedup percentages. Neither geometry is promoted to App defaults.

Both complete raw functional exports are now verified by `cmp` against the
preceding64-square norm-pack route: all157,286,400 FP16 bytes are identical, SHA
`46f31e5b44e88dc5dcd191b95641adfce11294a3220c757ec39773b77dd2e3b4`.
Each candidate separately passes the same serialized-input audited all-FP32 VAE
comparison, including all25 per-frame gates, with unchanged relative-L2
.0005252472709952212 / cosine .9999998670206866. This is a complete decoder
fixture result, not general image/video/App/device qualification.

The actual ordinary library's public release byte/link checks and App signature
verification pass, as do sixteen release/build-identity guard tests. No
experimental package or new GUI workflow is claimed; the existing running
App/ComfyUI are not replaced.

The first final-aggregation attempt incorrectly compared the signed/relocated
package library against the pre-signing build hash. That failed aggregation is
preserved separately. The corrected v2 audit binds the actual package hash
`d5a34915f0976eb6936c45c022925692dad90cc46b75610d294c6c5f896c3fac`
to the preceding package's authoritative receipt; both original unsigned builds
still have their original503b0b05... hash. No library was modified to satisfy a
check. The v2 audit freshly reruns all four native bank tests and rechecks raw
exports, independent FP32 receipts, all cohort observations/compiled identities,
120 CPU/contracts, sixteen guard tests and the actual package release/signature.

## Artifacts and reproduction

Generated banks/builds/results are ignored, not committed weights or media:

- `models/coreml/seedvr2-vae-{hotconv128,upscale256}-64x{62,94}-probe-v1/`.
- `build/seedvr2-vae-halo-tile-probe-v1/`.
- `outputs/seedvr2-halo-tile64x62-real1024-t25-abba-v1-20261015/`.
- `outputs/seedvr2-halo-tile64x94-real1024-t25-abba-v1-20261015/`.
- `outputs/seedvr2-halo-tile64x{62,94}-real1024-t25-functional-v1-20261015/`.
- `outputs/seedvr2-halo-tile64x{62,94}-real1024-t25-fp32-v1-20261015.json`.
- `outputs/seedvr2-halo-tiles-final-gates-v1-failed-20261015.json`: preserved
  aggregation-only identity mistake, not a passing gate.
- `outputs/seedvr2-halo-tiles-final-gates-v2-20261015.json`: terminal reverified
  receipts, native unit outputs and actual source/bank/probe/package identities.

Use `tools/coreml/export_seedvr2_vae_conv.py --channels 0 --rows 64 --cols 62`
or94 and the explicit original layer/checkpoint into a fresh directory. Build
with `TURBOCIDER_VAE_CAUSAL_PROBE_OUT=... bash
tools/native/build_seedvr2_vae_causal_conv_probe.sh`. Run the causal native unit
test with `norm` for the128-output layer and `upscale` for the256-output layer.

For the complete comparison, supply the existing64-wide hot/upscale manifests
to `seedvr2_vae_causal_decode_cohort.py`, select `--reference-route dual_norm
--candidate-route dual_norm`, and pass the two new banks independently through
`--candidate-manifest` / `--candidate-upscale-manifest`. Use the complete saved
1024/t25 latent/pixels and a fresh output. Bank-specific shape/strides/call counts
must pass; a single tile prediction or a shortened clip is not this gate.

Remaining full-objective work includes measured useful public configurations,
whole-request model/history residency, real low-memory hardware, broader media,
and complete-context longer videos. In particular the existing dual route's
request high water is already reached in the final DiT-block/output-head
interval: smaller VAE IO pools cannot be credited automatically as a lower
whole-request maximum.
