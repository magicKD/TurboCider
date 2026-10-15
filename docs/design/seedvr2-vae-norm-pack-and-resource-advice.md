# SeedVR2: cumulative request memory, norm packing and resource advice

## Request high-water attribution (2026-10-15)

A read-only observer uses the managed public MLX Python allocator counters in
the same process as the already-qualified native library. Native/Python request
peaks must be identical and positive; both counters return to zero after unload.
The observer creates no tensors, adds no GPU eval/fences, and never resets peak
or changes allocator/cache settings. Both original source videos and every
decoded output frame match their respective earlier nonobserved requests.
These diagnostic requests are not speed benchmarks.

Both complete1024/t25 runs have188 existing events. The following are
**cumulative high waters at observations**, not independently reset phase peaks:

| Observation | GPU shared-history | Dual Core ML convs |
| --- | ---: | ---: |
| Load complete | 7,285,550,662 | 7,285,550,662 |
| Last observed encoder event | 7,568,722,766 | 7,568,722,766 |
| DiT completion event | 12,235,735,306 | 12,235,735,306 |
| Final real exported frame | 13,014,502,498 | 12,235,735,306 |

For GPU, the final maximum is first observed at exported frame3, after the
preceding high-resolution decode event. For dual, it is already first observed
at `seedvr2_dit completed=32` and never increases during decoding. Importantly,
the code emits index31 **before** the last block, and index32 **after the final
output projection**. Existing events therefore localize this interval to the
last DiT block plus output head, not specifically one FFN or attention operator.
Do not present cumulative values carried through decode as independent decode
maxima. The4480-byte difference from the earlier unobserved dual maximum is not
a meaningful memory improvement/regression.

This establishes why further local VAE savings cannot automatically reduce the
whole-request maximum. Remaining complete-context conditioning/DiT residency
still matters to the low-memory/long-video objective. No admission estimate or
121-frame cap is lowered on the strength of this trace.

## Full-statistics affine/SiLU/halo packing prototype

The isolated hotconv candidate retains raw FP16 frame owners plus their original
per-frame32-group FP32 means/inverse standard deviations, rather than retaining
the full normalized current activation. The GPU applies the original affine,
FP16 cast, FP16 sigmoid and multiplication rounding directly while writing the
bounded Core ML tile surface. It recomputes old activations from each old frame's
own full-frame statistics; it does not use tile-local statistics. Spatial zeros
are applied **after** normalization/activation, so a padded zero is not passed
through a learned affine. Initial GPU FP32 folding remains unchanged.

Raw histories own bounded complete frames; known views with larger/offset backing
are materialized. No immutable MLX history is modified in place. Raw+statistics
history is mutually exclusive with the earlier normalized hotconv history;
the independent256-output spatial-upsample convolution can still run alongside.
This is not a full-VAE tile graph or a public request/App configuration.

Native signed-ABA tests check120,434,688 packed FP16 words against the existing
full-frame native GroupNorm result, plus11,857,664 assembled/bias words and an
original-coefficient FP32 convolution. Actual offset/strided and ragged operands,
replicated histories, retained output, wrong banks/operands, pre-cancellation,
cancel after a genuine prediction, same-runtime/VAE recovery and zero-active
teardown pass. The old128/256-output routes also pass their native regression.
Cancellation is between tile predictions, not interruption inside Core ML's
synchronous prediction call.

The first build failed because MLX eval accepts a vector/array arguments, not a
`std::array` directly. That failed output directory is preserved. The corrected
build passes; no functional/numerical gate is relabelled from the failed build.

Complete1024/t25 functional observation (not a balanced speed measurement):

| Decoder active maximum | Prior dual | Fused norm packing |
| --- | ---: | ---: |
| Bytes | 12,190,583,906 | 11,922,148,962 |

The difference is268,434,944 bytes. All157,286,400 exported FP16 bytes equal the
prior dual output exactly, SHA
`46f31e5b44e88dc5dcd191b95641adfce11294a3220c757ec39773b77dd2e3b4`.
The same-input audited all-FP32 VAE gate still passes the entire tensor and each
of25 frames, relative-L2 .0005252472710 / cosine .999999867021. The two actual
convolutions each execute6144 predictions/backing hits;24 hotconv frames use
norm packing, and the initial frame makes zero predictions.

A balanced dual-versus-dual-norm cohort is separate. Interpret only a terminal
complete receipt before claiming a latency gain. Recomputing three activations
can cost additional arithmetic; eliminating a full activation is not by itself
proof of faster execution. The prototype is not promoted to default/public or
full-request selection, even if its component gate passes.

The corrected complete1024/t25 ABBA has now completed four independent
processes/twelve freshly recomputed clips, with four warm observations per
route. Every frame passes, original latent bytes are unchanged, initial GPU
pixels/zero new predictions pass, and both layers report their full6144-call
counts per iteration (24 norm-packed frames only on the candidate):

| Warm complete decoder component | Dual | Dual + norm packing |
| --- | ---: | ---: |
| Median seconds including host pixel gate | 54.005329271 | 53.600636521 |
| Active maximum bytes | 12,190,583,906 | 11,922,148,962 |
| External surface bytes | 45,350,912 | 45,350,912 |
| Maximum sampled process footprint bytes, whole cold/warm processes | 13,687,686,344 | 13,586,711,776 |

Measured decoder median is about0.75% lower, a small fixture/machine-specific
difference; the active reduction is268,434,944 bytes. Sampled footprint differs
less and includes cache/cold portions, not just warm active allocations. Neither
metric is a hard RAM bound. This does not reduce the already dominant DiT request
high water or establish a new overall App speed gain. Norm packing remains a
compile-isolated probe choice, not a public/default route or qualified device
profile.

## App recommendations and scratch-space checks

The App's row-file quota text now shows the actual configured quota, not an
unconditional8-GiB literal. A utility task reads the current system temporary
volume's free-space snapshot outside SwiftUI rendering; cancelled/stale queries
do not publish to another model/configuration. Unknown observations stay unknown,
not a valid zero. A manual refresh is available; free space can change after any
snapshot, so advice is not an admission or disk reservation.

Pure resource advice uses the actual imported source frame count. It warns on
source/draft mismatch and >121 complete source frames, unsupported >1024 row-file
targets, unqualified <=16-GiB devices and remaining complete DiT/conditioning
state. It never truncates source frames, resizes output, enables unqualified ANE,
deletes artifacts or relaxes native admission. User-applied streaming/staged/
cache advice preserves explicit row/workspace choices.

For the measured1024-square/64-row video case (>=5 frames), the decoder reference
is6.5 GiB live files and the encoder-head reference2.25 GiB. Separate phase
lifetimes mean the references are **not added** when both are selected. A quota
below the relevant reference and free space below it plus0.5-GiB margin produce
warnings. Other dimensions/row heights use an explicit unmeasured-reference
notice, not an extrapolated guarantee. Single images do not inherit video
history-file requirements. Current local free space is below the known decoder
reference; no new multi-GiB spilling test is launched merely to fill the volume.

## Records / remaining gates

Generated under ignored `outputs/` and `build/`:

- `seedvr2-request-memory-trace-{gpu,dual}1024-t25-v1-20261015/report.json`.
- `seedvr2-norm-pack-real1024-t25-functional-v1-20261015/report.json`.
- `seedvr2-norm-pack-real1024-t25-word-equality-v1-20261015.json`.
- `seedvr2-norm-pack-real1024-t25-fp32-v1-20261015.json`.
- `seedvr2-resource-advice-app-controls-v1-20261015/`: weight-free App source/
  draft/advice checks, not an inference or UI-click qualification.
- `seedvr2-norm-pack-decode-real1024-t25-abba-v1-20261015/`: require final status.
- `build/seedvr2-vae-norm-pack-probe-v1/`: preserved build failure.
- `build/seedvr2-vae-norm-pack-probe-v{2,3}/`: corrected unit/complete decoder tools.

Fresh ordinary library/App regression, final resource-advice workflow checks and
the balanced norm-pack result are now completed for their stated scopes. The
ordinary build passes116 SeedVR2 CPU/contract checks plus sixteen build-identity/
release tests. Its source-staged App completes seven real requests (512/1024
images/complete25-frame videos and residency switches); all pixels/PNG alpha
equal the preceding ordinary App. Weight-free App tests also exercise source
count mismatch/limit, 16-GiB warning, quota/free-space mismatch, unknown space,
unmeasured geometry, single images and independent encoder/decoder file peaks.

Independent ad-hoc signed ordinary App:
`dist/seedvr2-resource-advice-20261015/TurboCider.app` (macOS26.2+).
The actual packaged library passes14 public contracts and release flag/byte/link
checks. The real embedded packaged CLI restores all512/t25 frames at24 FPS;
every decoded RGB frame equals the tested App output. Signing verification
passes. The main App, test frontend and native library used for packaging are
byte-identical to the tested build before relocation/signing; actual relocated
library/runtime identity is verified again. No GUI-click qualification or
16-GiB/1024-packaged-video performance claim follows from this smoke.

The full build script continued into other models' unrelated Swift test targets
after generating all required App/CLI/helper artifacts. Its owned isolated
process group was validated and intentionally stopped (143); no user App or
ComfyUI process was stopped. This is not a claim that the entire global build/
all-model test target completed. Native source/catalog freshness checks had
already passed, and required artifacts/package were independently checked.

Additional receipts:

- `seedvr2-resource-advice-app-e2e-v1-20261015/generation-report.json`.
- `seedvr2-resource-advice-app-export-equality-v1-20261015.json`.
- `seedvr2-resource-advice-packaged-cli-smoke-v1-20261015/report.json`.
- `seedvr2-norm-pack-resource-advice-final-gates-v1-20261015.json`: nine terminal
  receipts, current native-source/library IDs, identical tested/full-build
  artifacts and actual signed-package identity. Queries use separate processes;
  the controlled143 stop is recorded, not presented as a successful all-model build.

The overall objective still includes useful public configurations,
further VAE/model residency and speed improvements, real low-memory hardware,
whole-VAE/ANE feasibility and complete-context video support beyond121; this
component/observer/advice work does not complete it.

### Next whole-request residency work, not implemented

The request trace makes DiT residency a necessary next target for the original
low-memory/long-video objective, not a reason to advertise more VAE-local savings
as overall savings. Reading only the original checkpoint header shows:

- All DiT tensor bytes:6,784,202,176 (excludes safetensors header).
- Largest block, including the separate early video/text branches:317,266,986.
- Last block:158,633,514; nonblock/global tensors:121,595,008.

A source-verified block-weight loader could retain global tensors (including
block0's shared RoPE frequency) plus one current block, without changing the
existing full-context forward, casts or weights. Synchronize/evaluate the actual
block outputs before retiring its parameter owners; never evict a weight still
referenced by a lazy graph or active GPU submission. Initial serial IO is a
valid selectable memory/time tradeoff; bounded next-block prefetch would be a
separately measured option. Header arithmetic is not a measured allocation,
performance result or16-GiB-device qualification. Admission and source-frame caps
must remain unchanged until actual whole-request evidence supports a new mode.

For truly longer clips, inspect the global coupling before choosing a schedule:
each attention window attends to the same text state, and its text output is
reduced across all windows for the next layer. Independently restoring temporal
crops is therefore not an exact full-context replacement. A bounded per-layer
window schedule must preserve all video windows/shifted indices, shared text
state, original aggregation order/precision and causal VAE histories.

Original FP32 video residual state geometry alone gives1,059,061,760 bytes per
state bank for401 source frames at512-square, or4,236,247,040 at1024-square
(`Tlatent=101`, patch16, width2560). Two alternating state files would be about
2.118/8.472 GB before metadata, windows, model weights and other storage. These
are derived logical sizes, not total RAM/scratch bounds. The latter does not fit
the current local free volume; no such large-state experiment is launched and
no existing results are deleted to make it fit.
