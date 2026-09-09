# Wan native production entry — 2026-09-09

## Repeated ABBA benchmark (832x480x5)

`tools/validation/wan/benchmark_native.py` runs eager, compiled, compiled,
eager as four fresh processes, each making three requests in one Pipeline.
All 12 completed and the observed prompt-cache flags matched first-miss then
two hits. Raw per-process stdout/stderr, commands and probe SHA are retained
beside the benchmark report. No OS cache purge or thermal control is claimed.

| Mode | First-request median (2 samples) | Warm median (4 samples) | Warm range |
| --- | ---: | ---: | ---: |
| Eager | 3.66843 s | 2.96943 s | 2.96711–2.97025 s |
| Compiled | 3.40953 s | 2.71741 s | 2.71647–2.71830 s |

Compiled warm median was about 8.5% lower for this prompt/geometry. This
compares two current native modes, not the pre-refactor Python pipeline;
it cannot establish the requested all-model no-regression result. The harness
does not claim a quality gate and retains the videos for separate comparison.
Previous same-session RGB comparisons cover repeat determinism on that fixture,
not automatic validation of every video produced by this new timing run.

## Repeated same-session performance note

The validation probe's `--repeat 2` path was run separately for eager and
compiled on 832x480x5. First/second request times were eager 4.95696/2.97247 s
and compiled 3.41377/2.71481 s; both second requests reported
`umt5_prompt_cache=1`. The second outputs for each mode were pixel-exact to
their first outputs. This confirms prompt-cache reuse and deterministic native
sampling, but it is not a formal ABBA benchmark: process order, OS/Core ML
caches and thermal state were not randomized or controlled.

The two sharded-checkpoint/Core ML LoRA test modules now run with the project
Python environment as well: 10 tests and 2 subtests passed (previous system
Python run skipped NumPy-dependent cases). The complete `make test` suite had
no failures before this note.

## Same-session prompt-cache A/B (5-frame smoke)

The probe now supports `--repeat 2`, preserving one `wan::Pipeline` instance and
using a distinct output file for the second request. Eager requests measured
4.95696 s (first request, MLX peak 12,164,001,320 bytes) and 2.97247 s (conditioning
cache hit, peak 5,855,389,622 bytes). Compiled requests measured 3.41377 s
(first request, peak 12,164,001,320 bytes) and 2.71481 s (cache hit, peak
5,855,290,314 bytes). Both second outputs were pixel-exact to their respective
first outputs (five decoded frames, zero MAE/correlation error).

The cache hit removes UMT5 work and lowers the measured per-request allocator peak; these are
single-session measurements rather than a cold/warm ABBA benchmark. The probe
is a validation executable and is not part of the release App.

## Full-geometry hybrid comparison

The same 832x480x81 fox-walking request completed with the existing INT8
4096-channel Core ML prefix artifacts (30 compiled blocks). Native hybrid
request wall was 78.5605 s and MLX peak 37,242,861,278 bytes. Compared with
the earlier one-shot eager run (75.112 s, 46,803,623,346 bytes), hybrid used
about 9.56 GB less MLX allocator memory but took longer overall. Core ML/OS
allocations are excluded from these MLX figures, so this does not establish
an equivalent reduction in total process memory. No stable speedup is proven.

All 81 frames passed the existing aligned-RGB gate against native eager:
mean correlation 0.9944991, minimum 0.9764680, mean cosine 0.9980362,
mean MAE 3.82305/255 (worst-frame MAE 7.24159), maximum motion-relative
error 0.0765909. This is visibly looser numerical agreement than compiled
GPU vs eager, not exact parity or independent reference-video qualification.
The gate uses mean MAE, so its pass does not imply every frame's MAE is below 5.
Hybrid remains explicit and approximation-opt-in; do not auto-select it based
on this measurement. Repeated runs, process-wide memory, and broader prompts
remain unqualified.

## Full-geometry 81-frame eager/compiled smoke

Both native paths completed 832x480x81 generation for the same fox-walking
prompt and probe-default seed. Eager: 75.112 s, MLX peak 46,803,623,346 bytes.
Compiled: 73.2908 s, MLX peak 46,803,148,230 bytes. The one-shot difference
is about 2.4%; no repeated timing/noise analysis was performed, and neither
result proves a stable speedup or satisfies the global no-regression goal.

All 81 decoded frames passed the existing aligned-RGB regression gate:
mean correlation 0.9992441, minimum correlation 0.9951462, mean cosine
0.9997351, mean MAE 1.73474/255, maximum motion-relative error 0.0260537.
Shape, frame count and reported FPS matched. Outputs are not pixel-exact.
This compares native eager with compiled only; it does not resolve UMT5's
reference difference or establish Python-oracle/hybrid/full prompt-suite
quality. No production defaults were changed based on this result.

## End-to-end 832x480x5 eager/compiled A/B

Using the real `FastMetal-1.3B-QAD` checkpoint and converted TAEHV decoder,
the native pipeline generated the same prompt at 832x480x5 with eager and
compiled DiT paths. Eager wall time was 3.99913 s (MLX peak
12,164,001,320 bytes); compiled wall time was 3.42904 s with the same peak,
an observed 14.3% wall-time reduction in this single run. The result is a
small smoke benchmark, not a stable performance claim.

Decoded eager/compiled videos had equal geometry, frame count and FPS. The
aligned RGB gate passed: mean correlation 0.9995169, minimum correlation
0.9993250, mean cosine 0.9999457, mean MAE 1.0055/255, maximum motion-relative
error 0.00621. This qualifies the compiled path against eager for this small
fixture only; it is not a Python/reference quality gate and does not cover
81-frame output, Core ML hybrid, cold/warm ABBA repetitions or full model-pack
distribution.

## Sequential sampling verification (small fixture)

`wan_dit_parity.py --rollout` and the native DiT probe now feed each updated
latent into the next DiT call using two identical, explicitly saved renoise
tensors. Real checkpoint (1,508,862,080 bytes), latent `[1,16,3,8,8]`, synthetic
16-token conditioning, timesteps `[1000,757,522]`: all 46 captured stages were
finite and exact against the Python MLX reference, including three eager and
three compiled rollout latents (RMSE/max absolute difference zero).

The older `dmd_0...2` outputs are still independent formula checks; the new
`eager_rollout_0...2` / `compiled_rollout_0...2` outputs check sequential updates.
This is not full video qualification: it excludes actual UMT5 conditioning,
TAEHV decode, MP4 export, full production geometry and Core ML hybrid. The
rollout invokes production DiT/sampling functions but not the complete Session.

## Implemented

- Registered model is `wan2.1-1.3b-qad`, displayed as Wan 2.1 1.3B QAD.
- `native/platform/apple/wan_session.mm` owns native `wan::Pipeline` and
  `UnigramTokenizer`. The Python process/pipe worker and its environment/profile
  discovery were removed from the shipping session and build/package targets.
- Legacy Python oracle and profile moved to `tools/validation/wan/`.
- Model package requires `vae/taew2_1.safetensors`; offline conversion lives in
  `tools/convert/wan_taehv.py`. No inference-time conversion/download occurs.
- Premerged LoRA provenance checks retained, including pinned source identity,
  adapter strength/role, merged files, SHA checks, and mapping completeness.
  New C entry is `tc_wan_lora_preflight_json`; historical ABI remains an alias.
- New Wan manifest schemas accepted, historical schemas read for explicit
  artifact compatibility only. Historical source paths are not model discovery.
- Native session rejects audio, noise overrides, unsupported dimensions,
  incompatible compile/hybrid settings and unsupported LoRA hybrid requests.

## Evidence from this migration

- Native and Swift App builds completed.
- `tests/native/test_contract.py`: 57 tests passed.
- `tests/repository`: 11 tests passed after replacing the old worker-packaging
  assertion with an absence assertion.
- Legacy development oracle protocol tests: 7 passed.
- `tests/native/test_wan_cancellation.py`, explicitly configured with real
  checkpoint/tokenizer/UMT5/native TAEHV: **5 passed in 69.28 s**. Cancellation
  covered model loading, text encoding, DiT, decoder and media export. Each case
  preserved the previous destination, left no temporary MP4, recovered on the
  same engine, reused prompt conditioning, then unloaded/reloaded successfully.
- Ad-hoc signed package generated, minimum macOS **26.2**. No FastMetal worker
  file remained in `dist`.
- Copied App's bundled CLI generated 832x480x5 video from a non-repository cwd
  with `env -i PATH=/usr/bin:/bin`: **6.6278 s**, MLX peak **12,080,115,234 bytes**.
  No Python executable/profile configuration was supplied. This test used an
  explicitly selected temporary model package with links to existing fixtures;
  it is not evidence of a fully copied model package on a clean new Mac.

## Still open

The new native entry is not a final quality/performance qualification. UMT5's
non-exact reference difference, matched full-video baseline, hybrid quality,
real premerged LoRA generation, repeated cold/warm performance, and clean-machine
GUI/distribution checks remain. Historical Python worker measurements cannot
be substituted for these checks. Legacy optional Wan VAE/quantization modes are
not silently supported by the new session; its present native contract uses
the qualified affine INT8 DiT and TAEHV decoder.

H3/LTX runtime Python LoRA preparation and Core ML export launchers have since
been removed from production; see `production-tool-boundary-2026-09-09.md`.
The overall objective remains active because numerical/performance qualification
is not complete.

The real UMT5 probe is structurally correct for embedding, relative buckets and
padding, but encoder blocks are not exact against the Transformers BF16/MPS
oracle (after the GELU boundary fix, final block RMSE about 4.98, maximum absolute
error 320, while final normalized conditioning cosine is about 0.99983). Until
the attention/FFN precision boundary is explained and a matched video comparison
is completed, Wan native output remains an engineering build rather than a
quality-qualified release.

## UMT5 GELU scalar correction

Same-input isolation found that rounding the GELU constants to BF16 before
multiplication was incorrect for the reference's eager scalar arithmetic.
FP32 scalar multiplication followed by BF16 result rounding matches all seven
isolated stages on the captured first-layer activation. The old expression
had 19,017 differing output elements on that same input.

`UMT5Encoder::activation` now implements that boundary independently of Wan
DiT's compiled FP16 GELU. `tools/validation/umt5_gelu_parity.py --probe ...`
checks the actual linked native activation, not just a duplicate Python formula.
For the 32-token real-model fixture, before/after measurements were:

| Stage | Before RMSE | After RMSE |
| --- | ---: | ---: |
| Block 0 output | 0.051426 | 0.007548 |
| Block 23 output (before final normalization) | 4.276626 | 4.975910 |
| Final normalized conditioning | 0.00093123 | 0.00081338 |

Final cosine rose from 0.99977054 to 0.99982504. Some intermediate errors
increased: the corrected local operation does not prove full-video quality,
nor that every block is closer. Projection differences remain and performance
of the corrected end-to-end route still requires matched measurements.

## First-block attention trace

The native trace and Transformers oracle now capture actual attention scores,
probabilities and the input to the output projection (not reconstructed scores).
A real 32-token, 14-valid-token run produced 40 matching stage names, all finite:

| Stage | RMSE | Maximum absolute difference |
| --- | ---: | ---: |
| Attention scores including mask/bias | 0 (exact) | 0 |
| BF16 softmax probabilities | 1.317089e-9 | 2.384186e-7 |
| Attended values, before output projection | 1.015949e-5 | 0.001953125 |
| Output projection | 0.000961926 | 0.25 |

Final normalized conditioning metrics match the prior GELU-corrected run:
RMSE 0.0008133833, cosine 0.9998250354. This narrows the first-block investigation:
masked scores match even though Q/K differ slightly; differences remain in
softmax/value aggregation and projection. It does not establish which operation
causes downstream error or prove full video quality. Next isolate those operators
using identical captured inputs before changing precision or accumulation.

Reproduce with `tools/validation/umt5_parity.py --trace-first-block`, an explicit
model directory, and the linked `build/native/umt5-probe`. This is development
tooling; normal native generation supplies no trace callback.

## Same-input attention replay

`tools/validation/umt5_projection_parity.py` now isolates operators using each
captured input set (reference and native), loading only the first output weight.
It compares Python MLX and PyTorch MPS without modifying production precision.
Both backends reproduce their own recorded projection output exactly.

| Identical input source | Softmax mismatches | Value aggregation mismatches | Output projection mismatches |
| --- | ---: | ---: | ---: |
| Reference | 2 / 65,536 | 0 / 131,072 | 30 / 131,072 |
| Native | 2 / 65,536 | 0 / 131,072 | 45 / 131,072 |

Projection same-input RMSE is 0.00009616065 for reference input and
0.0002193323 for native input (maximum 0.015625 and 0.0625 respectively).
Explicit FP32 MLX projection followed by BF16 conversion produced the same
comparison metrics, not an improvement. This does not justify promoting the
production projection to FP32.

The prior stage-level observation did not establish value-aggregation kernel
error: with identical probability/value tensors, aggregation is exact in this
fixture. Differences in its end-to-end output therefore arise from differing
inputs in this replay. Softmax and projection retain small backend differences;
their detailed rounding cause and full-video effect remain unproven. These are
Python operator experiments matched to captured native stages, not a new native
performance or quality acceptance gate.
