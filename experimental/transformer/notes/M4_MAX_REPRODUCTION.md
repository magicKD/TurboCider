# M4 Max Transformer GPU/ANE reproduction — ongoing

## Current actionable result: smaller VAE tiles preserve denoiser performance

On this M4 Max, the existing 256-pixel spatial VAE tiling with 64-pixel overlap
avoided the post-decode parent-process compression observed with untiled and
512-pixel decoding in the tested 704x448/97-frame case. This is an **opt-in
tested configuration**, not a new default or a general resolution policy.
Use `config/ltx_m4max_vae256.env` explicitly; it does not enable ANE or change
attention. It changes decoder output pixels, so quality must be evaluated.

Two complete resident video+audio requests per route have now finished:

| Route, VAE256 | First Stage-2 s | Second Stage-2 s | First request s | Second request s |
|---|---:|---:|---:|---:|
| GPU | 30.169566 | 30.248053 | 64.496460 | 60.312767 |
| GPU + Stage-2 INT8 ANE MLP | 26.942670 | 27.008621 | 71.998984 | 61.979567 |

The ANE Stage-2 ratios are 1.120x/1.120x, but complete-request ratios are
0.896x/0.973x: **ANE did not beat the tiled GPU-only route end to end in this
pair.** ANE load was 13.431 s vs GPU 4.480 s on the first requests. The warm
ANE request's VAE/audio decode cost was also higher (8.564/1.294 s vs
5.094/0.491 s), consuming its Stage-2 saving. Thus the demonstrated model-level
improvement is the VAE memory/scheduling configuration, not an ANE E2E win.

For context only, earlier untiled warm resident GPU/ANE requests were
83.718/94.878 s versus current 60.313/61.980 s (about 1.39x/1.53x). These are
different sequential campaigns, not a contemporaneous randomized speed test;
do not attribute all that difference solely to tiling. The stronger mechanism
evidence is the paired replay intervention and observed compression changes.
New complete reports: `ltx-vae256-gpu-001.json` and `ltx-vae256-ane-001.json`
under `outputs/transformer-m4max`. Each has only one warm request; more prompts,
resolutions, repetitions and stronger external-workload control are outstanding.

### Decoder quality check

Compared each of the two tiled videos to its corresponding **same-route**
untiled video. Every Stage-1/Stage-2 video/audio latent and Stage-2 input hash
matches, isolating the decoder change. All 97 full-resolution frames were
compared, with matched dimensions and fps:

| Route | Mean RGB correlation | Mean absolute error /255 | Max motion-energy relative error |
|---|---:|---:|---:|
| GPU | 0.992876 | 4.9491 | 3.15% |
| GPU/ANE | 0.992838 | 4.9806 | 2.89% |

Full-resolution LPIPS Alex v0.1 was also evaluated for all 97 frames, with no
resize or crop:

| Route | Mean LPIPS | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| GPU | 0.041240 | 0.041145 | 0.051314 | 0.061581 |
| GPU/ANE | 0.041232 | 0.042125 | 0.051814 | 0.058741 |

Metrics, including per-frame LPIPS, were identical across the two request
indices within each route because the repeated fixed prompt/seed runs produced
the same corresponding videos. The LPIPS identity check was 0.0 in all four
comparisons. These descriptive scores support a relatively small decoder-only
perceptual difference in this workload, but there is no universal LPIPS
acceptance threshold and this is not an independent multi-seed sample.
At frames 0, 48 and 96, the inspected GPU/ANE contact sheet retains the fox,
composition and coarse detail with no obvious hard tile boundary; some local
texture/detail differences are visible. This is a sampled visual inspection,
not full-motion playback or proof of artifact-free output. No temporal
perceptual score, lip-sync check or multi-prompt acceptance is claimed.
The user's visual-quality goal is not replaced by exact pixel parity: tiling
is not pixel-exact, and downstream acceptance must consider actual appearance.

Artifacts: `vae256-quality-002/report.json` and
`vae256-gpu-quality-001/report.json` under the campaign output root, with
`request-0-comparison.png` and `request-1-comparison.png`. The first quality
attempt failed at contact-sheet extraction because local ffmpeg does not accept
`-vsync`; it is not counted. The evaluator now uses `-fps_mode passthrough` and
refuses to overwrite output directories. Its reports do not claim timing
comparability with the older references. All-frame perceptual results are in
`vae256-lpips-001.json`. They use LPIPS commit
`082bb24f84c091ea94de2867d34c4544f68e0963` and model-state SHA256
`cd2943aa26b6c7e7d100ab7e09a6c0975653ff57be42499de2ab48991fff82de`,
matching the earlier Stage-2 LPIPS evaluation. The LPIPS tool now accepts a
completed VAE-tiling quality report only when every recorded denoising tensor
pair is byte-exact; this keeps the comparison isolated to decoder behavior.

### VAE tile-size screening evidence

Both modes use the stat-only helper and retain the real denoiser:

| Tile | Stage-1 first/second s | Stage-2 first/second s | Decode first/second s | Compressed bytes after decode |
|---|---:|---:|---:|---:|
| 512, overlap 64 | 23.679 / 47.352 | 26.857 / 35.445 | 14.329 / 14.780 | about 26.63 GB |
| 256, overlap 64 | 23.620 / 23.957 | 26.802 / 26.348 | 5.260 / 5.148 | 0 in both rounds |

In the 256-pixel case, post-decode resident memory remains about 26.7 GB. This
is not a decoder peak-memory measurement: snapshots only cover the parent
before/after the child, not the child's peak working set. Raw observations:
`vae-tiled-replay-001.json` and `vae-tiled-replay-256-001.json`. The replay
binary was `4657ca661c69a44c235bcec5756ffe1ff4b1aacd25a9ba83f16020bd1cd8562c`;
the rebuild-denoiser experiment was explicitly disabled. No new graph-cache
API was retained: the existing graph-reset negative result already covers
that hypothesis at a similar stage boundary.

## No-decoder native replay: actual isolation result (2026-09-14)

The existing connected-conditioning cache contains all video/audio text and
mask tensors. The earlier conversational claim that these inputs were missing
was incorrect. The native library exports `ltx_native_create/run/free` and
supports reuse without request-layer decoding. A new optional diagnostic,
`src/ltx_stage2_replay.c`, now exercises these entry points with **real** cached
conditioning and reference tensor dumps. It is fixed to 704x448, 97 frames,
seed 42, text rows 1024, and validates exact input sizes. Stage-1 audio is the
Stage-2 audio input; the final Stage-2 audio dump is never reused as input.

With the same Stage-2-only INT8 MLP profile as `worker-off-001`, worker off:

| Replay scope | Run | Stage-1 s | Upsample s | Stage-2 s | Reference parity |
|---|---:|---:|---:|---:|---|
| Stage-2 only | 0 | — | — | 26.951066 | video/audio byte-exact |
| Stage-2 only | 1 | — | — | 26.191359 | video/audio byte-exact |
| Stage-2 only | 2 | — | — | 26.259019 | video/audio byte-exact |
| Both stages, no decode | 0 | 23.775536 | 0.363296 | 26.918356 | all checkpoints byte-exact |
| Both stages, no decode | 1 | 23.825779 | 0.219634 | 26.260982 | all checkpoints byte-exact |

These runs preserve one native denoiser instance, reset identical real inputs
before each replay, and execute no video/audio decode or export. Both-stage
mode regenerates the native seeded noise, runs Stage-1, verifies both Stage-1
latents, executes the actual checkpoint upsampler, verifies the Stage-2 input,
and then verifies final video/audio latents. The tool fails with status 3 on
any reference mismatch. A GPU-only attempt against the ANE reference correctly
failed parity and is not a qualified matched GPU speed baseline.

**Finding:** neither repeated Stage-2 alone nor repeated Stage-1→Stage-2
reproduces the large resident regression. In the both-stage replay, cumulative
Stage-2 GPU partial time is 6858.000/6857.671 ms, while ANE prediction is
8069.216/7500.087 ms. Contrast the full resident requests' Stage-1
23.602→44.422 s and Stage-2 26.554→35.265 s. This narrows investigation to paths
present in the complete request but absent here: decode/export and request-layer
resource/lifecycle management. It does **not** identify VAE alone as the cause;
request-level cleanup and external-state differences remain confounders.

The observations are in `outputs/transformer-m4max/no-decoder-replay-001.json`.
They are transcribed from captured tool output, not an automatically collected
campaign. Same native library SHA256 as the worker pair:
`3ac1557d9953a54ebf2597c01395b2110133c1c851bfa95d251646c68a69437e`.
The extended replay executable hash is
`00eec8b5f4300c2e9d85713f5588e86638d78c323e722eaf82f59d0f188a01ea`.
Core ML load times (18.741 s before Stage-2-only and 2.462 s before both-stage)
are not comparable cold measurements; cross-process specialization-cache state
was uncontrolled. No end-to-end acceleration or new visual-quality claim is
made from replay latency.

Build after the native engine:

```sh
make -C experimental/transformer build/ltx-stage2-replay
experimental/transformer/build/ltx-stage2-replay \
  CHECKPOINT ABSOLUTE_SHADER CONDITIONING_DIR DUMP_DIR ANE_DIR RUNS
```

Append `UPSAMPLER VIDEO_VAE` to include Stage-1 and upsampling without decoding.
`DUMP_DIR` is the exact request tensor directory, not its parent. All artifact
arguments should be absolute paths. Use references produced by the same ANE
profile, prompt, seed and geometry. This tool calls an internal native ABI;
rebuild whenever `ltx_native.h` or the engine changes. The optional target uses
the relative loader path to the existing native library and can override
`LTX_DEPLOYMENT_TARGET` for other local builds. Production defaults are unchanged.

### Replay autorelease-pool ablation (2026-09-14)

The complete C API wraps each request in an Objective-C autorelease pool,
whereas the original replay did not. The replay now supports the default-off
`TURBOCIDER_LTX_REPLAY_DRAIN_POOL=1`: a pool surrounds each replay round and
drains even on a parity failure. Native model ownership is unchanged. This
diagnostic is compiled as Objective-C without ARC solely to manage that pool.

| Pool mode | Stage-2 run 0 s | Stage-2 run 1 s | Reference parity |
|---|---:|---:|---|
| disabled | 26.911260 | 25.959067 | video/audio byte-exact |
| enabled | 26.840282 | 26.138276 | video/audio byte-exact |

Both processes use the same replay executable, real inputs, Stage-2-only ANE
MLP profile and native library. The pool-enabled run follows the pool-disabled
run. Neither shows the large full-request hot Stage-2 regression. Thus a
per-round autorelease boundary alone is not sufficient to reproduce that
regression in this test; this is not a statistically powered equivalence test.
It does not cover autoreleased objects created during original model loading,
decoder allocations or request export. Pool drain cost is outside the reported
Stage-2 timer. No production pool behavior was changed.

Replay executable SHA256:
`225781df8efbf57c5d66d1b1489d589f32169e340cf6811bc8222357fb548cdc`.
Native library remains
`3ac1557d9953a54ebf2597c01395b2110133c1c851bfa95d251646c68a69437e`.
All four video/audio exact-reference checks succeeded. These are screening
observations, not a new warmup optimization or complete-request speedup.

### Video-helper intervention: regression reproduced (2026-09-14)

The full request already uses `decode_ltx_video_isolated` when the bundled
helper is available. Earlier conversational claims that a child decoder would
necessarily be an invalid comparison were incorrect. The replay now optionally
launches that **same helper**, preserving the parent denoiser/ANE objects.
It writes the exact generated latent, waits for decoding, reads and size-checks
the 97x448x704x3 BF16 pixels, and releases the temporary output before the next
round. This covers helper launch, actual Video VAE, parent readback and cleanup;
it does not include RGB conversion, audio decode, video export or mux.

One same-executable pair completed: helper enabled first, then a separate
no-helper process; both use two full Stage-1→upsample→Stage-2 rounds, the same
Stage-2-only INT8 MLP artifacts, and `TURBOCIDER_LTX_REPLAY_DRAIN_POOL=1`.

| Intervention | Round | Stage-1 s | Upsample s | Stage-2 s | Helper + readback s |
|---|---:|---:|---:|---:|---:|
| Video helper | 0 | 23.746225 | 0.230480 | 26.774096 | 12.191524 |
| Video helper | 1 | 57.854761 | 0.355430 | 36.906059 | 14.936364 |
| No helper | 0 | 23.676876 | 0.366605 | 26.759250 | — |
| No helper | 1 | 23.685644 | 0.216460 | 26.027103 | — |

All Stage-1 video/audio, upsampled video, and final Stage-2 video/audio latents
are byte-exact against the original request references at every round. Both
helper invocations succeed. **The video-helper intervention reproduces the
regression while the no-helper control remains stable.** This is stronger
localization than the earlier full-request/replay comparison: neither audio
decoding nor export is required for the effect observed here. It does not
establish the internal mechanism or prove that VAE convolution alone causes it.
Possible driver residency, memory pressure and process interaction still need
timelines or additional interventions. No external-process exclusion or
counterbalanced repetitions were performed, so this remains a single paired
screening result, not production qualification.

This result directs optimization toward retaining effective Transformer/ANE
device residency across Video VAE use, or scheduling denoising and decoding
in separately managed workers. Simply keeping host model objects or changing
pthread lifetime is insufficient in the observed complete-request workflow.
These are next hypotheses, not implemented speedups.

Raw observations and complete argv:
`outputs/transformer-m4max/video-helper-intervention-001.json`. They are
transcribed from tool stdout; two malformed-path attempts failed and are
excluded. Startup file validation now rejects such attempts before loading
the Transformer. Temporary latent/pixel files are created in a unique replay
directory and removed by the helper wrapper; no input artifacts are removed.

Replay executable SHA256:
`fd19b447c381035aaf88b9d1268fce34ec92a105b29d68eb5bb93db7f5459a2e`.
Helper SHA256:
`6fd736a12297f5dfd311a2d4192cdc92fa2f0bc12bab03ddd3a6c2be13e06a5a`.
Native library is unchanged from the earlier replay/worker tests.

To enable this intervention, append `DECODER DECODER_CHECKPOINT` after the
existing `UPSAMPLER VIDEO_VAE` pair. The two VAE checkpoint arguments point to
the same video VAE file. The decoder runs after **each** Stage-2, including the
last round. Its time is reported separately and is not a complete video-request
latency. Production code/defaults have not changed in this intervention.

### Decode modes: readback and process switching are not necessary (2026-09-14)

Two further interventions completed with the same newly rebuilt replay:

- `TURBOCIDER_LTX_REPLAY_VIDEO_MODE=stat-only` runs the original helper, but
  validates the output's file size without reading the pixels into the parent.
  The child still computes and writes pixels; this is not GPU-only/no-output
  VAE decoding.
- `TURBOCIDER_LTX_REPLAY_VIDEO_MODE=in-process` invokes the exported native MLX
  VAE in the parent. Each round creates the VAE, decodes into BF16 pixels,
  frees pixels and the VAE, then clears the VAE MLX cache. No child process or
  decoded-pixel file is used. This also changes VAE/context lifetime relative
  to a permanently resident decoder, so it is not an exact service simulation.

| Intervention | Stage-1 first/second s | Stage-2 first/second s | Decode first/second s |
|---|---:|---:|---:|
| Helper, no parent pixel read | 23.723517 / 43.590467 | 26.821139 / 34.311645 | 11.978159 / 11.757251 |
| In-process VAE, free + clear | 23.697018 / 44.859151 | 26.855886 / 34.359600 | 13.352747 / 12.913549 |

All Stage-1 video/audio, upsampled video, and Stage-2 video/audio reference
comparisons passed byte-exactly in both rounds of both modes. Both decoders
reported success, but decoded pixels were **not** compared between modes:
the tested parity is the denoising trajectory, not a new image-quality gate.
Timings exclude audio decode, RGB conversion and export. Decoder timings have
different scopes (helper startup/file output vs in-process create/free/cache
clear), so they are not pure VAE-kernel speed comparisons.

**Finding:** parent pixel readback and cross-process switching are each
unnecessary for the observed slowdown. In-process VAE execution and subsequent
cleanup still precede the same large Stage-1/Stage-2 regression. This further
narrows investigation toward shared GPU/MLX memory/device state during and
after VAE use. It does not prove weight eviction, paging, thermal throttling
or any particular driver mechanism. Moving the VAE into the parent or merely
removing parent copies is therefore not supported as a remedy by these runs.

Mode order was stat-only then in-process, two rounds each; the no-decode
control from the preceding intervention was not rerun on this new executable.
These are screening results with no external-process exclusion or statistical
confidence claim. Observations were transcribed from tool output into
`outputs/transformer-m4max/video-decode-modes-001.json`.
Executable SHA256:
`43b8e93e8b2e17aafa5d618aeef9e6661693368a46c7cdc818d643632bb8d955`;
native library/helper unchanged. The mode flag is rejected on unknown values
before any GPU initialization; default is `readback`. No production decode
mode has been changed.

## Machine and synthetic workload details

### Rebuild-denoiser after Video VAE (2026-09-14)

As a default-off mitigation probe, after each Video VAE helper the replay freed
the native denoiser and immediately rebuilt it before the next round. The first
round remained normal (Stage-1 23.670 s, Stage-2 26.813 s); Video VAE took
11.049 s and rebuild took 12.738 s. The next round restored Stage-1 to 23.698 s
and Stage-2 to 26.880 s, rather than the 43.6/34.3 s hot values without a
rebuild. All available Stage-1/Stage-2 reference checks were exact.

This apparent latency recovery is not a usable optimization yet. Before
rebuild, the process had only 3.5 MB resident and 26.6 GB compressed. Rebuild
raised physical footprint to 42.9 GB with 19.6 GB still compressed; after the
next Stage-2 it reached 47.3 GB footprint. The process exited 137 before it
could complete the second decode. Exit 137 is consistent with a memory kill,
but no unified system log was captured, so it is not proof of the exact killer.
The result demonstrates a trade-off: rebuilding restored denoising timing in
the observed round, while process footprint increased substantially and the
complete experiment failed. Ownership of the remaining compressed memory has
not been established; it cannot be labeled duplicate Transformer weights from
these counters alone. Do not enable this candidate in production.

Raw selected observations: `outputs/transformer-m4max/rebuild-after-video-001.json`.
This is one sequential screening run with stat-only helper, no audio/RGB/export,
and external Python processes present. A safe future version would need a
two-phase teardown that explicitly synchronizes and clears all old Metal/MLX
allocations before reconstruction, then verifies footprint below a hard budget.
The opt-in replay flag is `TURBOCIDER_LTX_REPLAY_REBUILD_DENOISER=1`; release and
create costs are combined in `denoiser_rebuild.seconds`. This runs only between
rounds, not after the last one. The current tool performs no extra cache purge
or memory-budget gate, and the failing mode is not a service recommendation.
A process-name check after failure found no remaining replay or video-decoder
helper. Replay CLI and existing validation tests pass (`29 passed, 24 subtests
passed`), but that does not turn this failed GPU/ANE experiment into a pass.

### Metal residency audit (2026-09-14)

The current LTX native GPU context allocates Transformer weights, workspaces
and ANE shared I/O with `newBufferWithLength:options:MTLResourceStorageModeShared`.
The audited path has no `MTLResidencySet` or `requestResidency` calls, and
`ltx_gpu_clear_graph_cache` only drops Objective-C graph dictionaries.
Consequently, “resident” in the LTX reports means retained host/model
objects, **not** an assertion that the GPU driver keeps all weights physically
resident.

This matches the replay evidence: the native denoiser alone remains stable,
whereas VAE use triggers a subsequent slowdown. A hypothetical explicit
Metal residency set would need to include the Transformer graph/weight buffers
without pinning the VAE working set, and must be measured for unified-memory
pressure; adding a blanket pin could make the 64-GB M4 Max result worse. No
residency change is implemented or enabled by default. The subsequent memory
snapshot experiment below measures process/host VM state first; it does not
inventory or pin individual GPU allocations. The installed SDK's public
`MTLResidencySet.h` documents `requestResidency`, `endResidency`, and allocation
commit operations. An explicit residency request is not proof of hard pinning.

New campaign on Mac Studio `Mac16,9`, Apple M4 Max, 16 CPU cores (12P+4E),
40 GPU cores, 64 GB UMA, macOS 26.6.2 (`25G83`). Historical `REPORT.md` and
`measurements.csv` describe M4 Pro and are not overwritten or pooled here.
Exact capture time, hardware query, compiler, package versions, source/binary
hashes, and commands are stored in each campaign manifest. This run uses
coremltools 9.0 and NumPy 2.5.2; the exporter emits a warning that installed
Torch 2.14.0 is outside its tested range. It exports MIL directly, not Torch.

This is a synthetic full causal RMSNorm/SwiGLU Transformer block, not an LTX
block: no RoPE, cross-attention, audio stream or pretrained weights. The first
campaigns use MPSGraph SDPA, not the historical report's Steel baseline.
Differences from M4 Pro therefore cannot be attributed to chip alone.

### Memory-pressure snapshot around Video VAE (2026-09-14)

An opt-in replay snapshot (`TURBOCIDER_LTX_REPLAY_MEMORY=1`) records task VM
and host VM counters before/after denoising and Video VAE. In the stat-only
helper intervention, before round 0 the parent had 23.28 GB resident and
23.28 GB physical footprint with no compressed bytes. After Stage-2 it had
26.71 GB resident and 27.77 GB footprint. After Video VAE returned, parent
resident size fell to 44.84 MB while **26.60 GB was compressed**. Before round
1 it remained at 47.76 MB resident with 26.50 GB compressed. After the hot
round's Stage-2 it was 3.49 GB resident with 23.21 GB compressed; after the
second decode it was 4.34 MB resident with 26.63 GB compressed.

During the first decode interval, host swap-outs increased from 466,229,189
to 467,611,289 and host compressions from 1,971,927,098 to 1,976,754,102.
These are cumulative host-wide counters, not an attribution of those changes
to this process or a measured paging rate. The direct process-level observation
is the increase in `task_vm_info.compressed`, with status zero (successful)
for every task/host query.

This provides direct evidence that the Transformer process is largely
compressed after the Video VAE intervention, before the next denoise. It
supports memory-pressure investigation and shows that host-object retention
does not guarantee uncompressed process memory. It still does not prove whether
compression/decompression, GPU page migration, MLX allocator behavior or
Metal resource eviction dominates. A future controlled run should record
`task_info` continuously and use a dedicated machine with other GPU clients
excluded. Explicit Metal residency requests remain untested; their memory
cost and effect on decoder execution must be measured, not assumed beneficial.
Process compressed accounting does not identify which allocations are
Transformer Metal buffers versus Core ML or other memory. The kernel may also
account GPU-accessible allocations differently from CPU-resident pages.

The corresponding run's Stage-1 was 23.662096/44.781622 s and Stage-2 was
26.887253/34.109769 s; all reference latent comparisons passed. Video decode
was 11.567270/11.921806 s. Raw selected snapshots are preserved in
`outputs/transformer-m4max/video-memory-snapshots-001.json`; observations were
transcribed from tool output. These are sampling points, not a continuous
memory trace, and they do not establish a per-allocation cause.

## First public-Core-ML measurements

QKV and attention stay on GPU. GPU and CPU_AND_NE each compute a complete
paired-channel FFN shard including partial down projection; GPU joins the
hidden-width partials and residual. Attention and FFN within a block have a
dependency and are **not** executed as independent parallel branches.

| Shape [M,H,F] | ANE intermediate share | GPU p50 ms | Heterogeneous p50 ms | speedup |
|---|---:|---:|---:|---:|
| [64,256,1024] | 50% | 0.359 | 0.576 | 0.623x |
| [256,1024,4096] | 50% | 1.103 | 1.086 | 1.016x |
| [1024,1024,4096] | 25% | 3.720 | 3.554 | 1.047x |
| [1024,1024,4096] | 50% | 3.724 | 3.071 | 1.213x |
| [256,2048,8192] | 25% | 3.152 | 3.038 | 1.038x |
| [256,2048,8192] | 50% | 3.150 | 2.838 | 1.110x |
| [1024,2048,8192] | 25% | 11.513 | 9.978 | 1.154x |
| [1024,2048,8192] | 50% | 11.518 | 9.712 | 1.186x |

First two rows: one process, 8 warmups, 20 timed pairs (`campaign-001`).
Remaining rows: 3 fresh processes each, 10 warmups, 40 timed pairs per process
(`campaign-002`). Table shows median of process p50s and median of process
speed ratios separately. Internal order alternates GPU/heterogeneous and
heterogeneous/GPU. Shape/share order is fixed; this is screening, not a
randomized independent confirmation. No confidence intervals are asserted.
No power/thermal trace or external-workload exclusion gate was recorded for
these first runs; they must not be treated as dedicated-machine qualification.

Independent FP32-reference heterogeneous NRMSE ranges from 0.003066 to
0.005745; each reported cosine is at least 0.999983. Output-backing use is
100% for measured FFN ANE calls. This proves caller-owned output reuse, not
the absence of internal Core ML copies, physical ANE placement, or video quality.
Per-op device placement and finite-output handling still need an independent
runtime audit before these are considered qualified results.

At [1024,2048,8192], 50% share, the second process reports a 5.431 ms ANE
branch versus 2.291 ms GPU branch. The ANE branch is the longer branch in this
configuration. A share between 25% and 50% and alternative lowering merit
measurement; increasing ANE share is not automatically beneficial.

## Reproduction

From the TurboCider root, use a Python environment containing NumPy and
coremltools. No old-machine absolute path is required for this public test.

```sh
make -C experimental/transformer build/transformer
Python/bin/python3 experimental/transformer/scripts/campaign.py \
  --output outputs/transformer-m4max/new-campaign \
  --shapes 1024x1024,256x2048,1024x2048 --shares 0.25,0.5 \
  --layers 1 --rounds 3 --warmup 10 --iterations 40
```

Use a new output directory and run hardware campaigns serially. Models,
weights and raw measurements remain local, are not committed, and are not
automatically removed. `manifest.json` records initial setup; `results.json`
is emitted only after all requested runs finish. Compiler/ANE setup time is
outside warm block latency, but raw process duration and preparation timing
are retained. Fresh process does not mean cold OS compiler cache.

## Reference audit and remaining hypotheses

- `gpu_ane/mac_transformer/native/mac_transformer.mm`: complete FFN partial
  down versus activation-join; persistent worker and shared public bindings.
  Historical M4 Max MLP results of 1.7–2.0x are not full Transformer results.
- `gpu_ane/mac_local_ai/docs/COREML_ANE_OVERHEAD_2026-08-25.md`: stable compiled
  paths and resident model pools reduce repeated setup; tiny wrapper savings
  do not imply faster prediction. Its M4 Pro numbers require new reproduction.
- `src/private_ffn.inc` and `scripts/add_private_gpu.py`: private MIL/IOSurface
  backend with CPU or GPU transpose. No-copy binding still leaves physical
  transpose passes and synchronization. Audit ownership of its temporary
  compilation directory before running: the inherited destructor deletes a
  content-derived directory without checking whether this process created it.
- `gpu_ane/mac_transformer/notes/transformer-heterogeneous-report-2026-09-04.md`:
  historical QKV splitting had local improvements but session switching and
  shared-resource contention erased full-model gains. Replicate rather than
  assume direct head-slot writes alone solve the bottleneck.

An existing TurboCider LTX validation summary is
`docs/design/validation/ltx-ane-and-720p-tiled-2026-09-13.json`. Its 704x448,
97-frame GPU+ANE INT8 both-stage run reports Stage-2 29.765 -> 26.131 seconds
(about 1.139x), but it changes the Stage-1 trajectory too. This is not an
isolated Stage-2 comparison. Its FP16 candidate was slower in total denoising.
Original binary/model provenance and timing scope must be checked before
comparison to current 121-frame sparse or GPU baselines.

Next work: audit timer and numerical checks; add campaign exclusion and result
validation; compare multi-layer stacks and stronger GPU attention; vary FFN
split/lowering; implement matched QKV partition; measure public/private bridge,
resident multi-model switching and warmup separately; finally use real LTX
weights/activations and matched Stage-2 requests with decoded visual evaluation.
Production kernels and dense defaults are unchanged in this campaign.

## Native packed-QKV projection probe (M4 Max, synthetic)

The existing native probe compares three separate INT8 ConvRot projections
against one packed-QKV dispatch. Inputs and weights are synthetic, packing is
precomputed and excluded from timing, and this is not an LTX quality test.

| Video rows | Separate s | Packed s | Packed speedup | Relative L2 |
|---:|---:|---:|---:|---:|
| 64 | 0.002660 | 0.002185 | 1.217x | 0.002647 |
| 256 | 0.003638 | 0.003611 | 1.007x | 0.002655 |
| 1024 | 0.008170 | 0.008034 | 1.017x | 0.002654 |

The benefit is strongly shape-dependent: the small 64-row case shows a
dispatch/setup win, while realistic larger token counts are effectively flat.
This is not evidence for a substantial GPU/ANE Transformer speedup. A real LTX QKV ANE
artifact and decoded-video comparison are still required before enabling that
route.

## FFN split sweep (M4 Max, screening)

For `[M,H,F]=[1024,2048,8192]`, a one-round six-point sweep varied the ANE
intermediate share. This isolates the load-balance question; it is not a
multi-round qualification.

| ANE channels | share | GPU p50 ms | heterogeneous p50 ms | speedup |
|---:|---:|---:|---:|---:|
| 1024 | 12.5% | 11.503 | 10.903 | 1.055x |
| 2048 | 25% | 11.514 | 9.978 | 1.154x |
| 3072 | 37.5% | 11.509 | 9.052 | **1.271x** |
| 4096 | 50% | 11.518 | 9.708 | 1.187x |
| 5120 | 62.5% | 11.520 | 12.564 | 0.917x |
| 6144 | 75% | 11.520 | 16.113 | 0.715x |

The local maximum is near a 37.5% ANE share, not 50%. At 37.5%, GPU and ANE
reported projection and ANE branches are 2.83 ms and 3.85 ms respectively; at 75%, ANE is 11.26
ms and becomes the clear straggler. This confirms that topology must be tuned
to measured device throughput and shape; “half the channels per device” is not
a general optimum. The sweep's output NRMSE remains below 0.007 and cosine
above 0.99997 against the independent reference, but no video-quality claim
follows from this synthetic block.

### Timer interpretation correction

Code inspection of `Projection::runHeterogeneous` shows that `up_gpu_p50_ms`
times only the first GPU command buffer. The supergraph GPU continuation
(SwiGLU plus partial down) is a second command buffer, waited for by the
whole projection wall timer but not included in `up_gpu_p50_ms`. The ANE
timer includes its complete FFN shard. These two fields therefore cannot
be directly compared as full-branch times. Earlier statements calling their
difference proof of a GPU/ANE full-branch imbalance were too strong. The
full-block wall speedups and observed 37.5% optimum remain valid, but the
cause needs continuation timing and a serialized-branch ablation.

## Private ANE / IOSurface smoke

`campaign-004-private` uses `[256,1024,4096]`, 50% FFN share, one block,
8 warmups and 20 paired measurements. The isolated private executable
successfully compiled/loaded the MIL model (41.31/20.69 ms reported setup).
Private FFN uses IOSurface shared-memory bindings with GPU transpose both
before and after ANE evaluation. Its complete block was **0.883x** GPU:
1.185 ms GPU versus 1.341 ms heterogeneous. Reference NRMSE was 0.005437,
cosine 0.999985. This establishes a working private path, not a speed win.

No-copy binding does not remove the two physical transpose passes or their
waits. The inherited `output_backing` field is also not evidence of a public
Core ML prediction in this private branch. Public/private lowering differs
(matmul versus convolution MIL); this is not a pure API-wrapper comparison.
The previous public smoke used a different executable/process and baseline
latency, so it is not a controlled latency comparison against private.

Private compilation now refuses preexisting content-derived temporary paths
and preserves its generated files instead of deleting them implicitly. The
campaign supplies a fresh per-round TMPDIR. No system setting or production
runtime was modified. Further experiments must compare matched public/private
executables, multi-layer stacks and full GPU-continuation timing.

## Semantic QKV split pilot

The runner now supports a matched semantic split: GPU computes Q (`H` output
channels) while ANE computes K/V (`2H` channels for equal-head geometry), with
head slots consumed by fused SDPA. FFN channels remain at 37.5% ANE. This is a
one-round, 20-pair pilot, not yet a multi-process qualification.

| Shape [M,H,F] | QKV+FFN split speedup | GPU p50 ms | heterogeneous p50 ms |
|---|---:|---:|---:|
| [256,1024,4096] | 0.876x | 1.160 | 1.324 |
| [1024,2048,8192] | **1.343x** | 11.520 | 8.577 |

At the large shape, QKV projection reports 1.798 ms heterogeneous wall (GPU
portion 0.715 ms, ANE portion 1.763 ms), and attention starts only after the
split Q/K/V are ready. The combined result improves on FFN-only at that shape
(1.271x), but the small shape loses to dispatch and ANE overhead. Large-shape
heterogeneous reference NRMSE is 0.0050 and cosine 0.999987. This shows that
semantic head-slot writes can be numerically valid and useful at sufficiently
large shapes; it is not yet evidence of LTX Stage-2 benefit. Production LTX
QKV remains unchanged.

### Three-layer follow-up

`campaign-006-qkv-stack` repeats `[1024,2048,8192]`, 37.5% FFN ANE share and
semantic Q/KV split with **three independent layers**. Each layer has different
seeded weights and consumes the preceding layer's actual output. Three fresh
processes each use 10 warmups and 40 internally alternating paired samples:

| Process | GPU stack p50 ms | heterogeneous stack p50 ms | speedup |
|---:|---:|---:|---:|
| 0 | 34.625 | 26.011 | 1.331x |
| 1 | 34.623 | 26.038 | 1.330x |
| 2 | 34.611 | 26.027 | 1.330x |

All three report reference NRMSE 0.007592 and cosine 0.999971. The observed
gain survives three resident layer instances, but this does not establish
48-layer session residency, LTX weights or production INT8 GPU competitiveness.
The sweep-selected configuration needs further independent workload coverage;
these three trials quantify repeatability at this one configuration only.

## Real LTX Stage-2 MLP pilot

The paired harness now accepts `--ane-manifest` as an alternative to sparse
attention. It uses a distinct `ltx-ane-stage2-paired-v1` schema so these results
cannot silently enter the sparse-attention summary. Admission requires MLP
stage mask 2 and no ANE QKV/KV/V2A route; native preflight additionally checks
model geometry and checkpoint provenance. Final completion requires byte-exact
Stage-1 video and Stage-2 video input, not merely a matching prompt and seed.

The existing 48-block INT8 ANE artifact supports only 1001/4004 video rows.
The initial real-model experiment therefore uses 704x448, 97 frames, seed 42,
11 steps, audio, component-staged residency and dense attention. This is NOT
the previous 768x448/121-frame sparse workload. The ANE FFN partition is 6912
of 16384 intermediate channels; the synthetic 37.5% setting is not transplanted
without re-exporting real weights.

Local profile: `experimental/transformer/ltx_stage2_mlp_int8.json` (ignored
experiment artifact). Report: `outputs/transformer-m4max/ltx-stage2-mlp-int8-97-pilot/report.json`.
Both variants use the current same native library, with one warmup each before
one measured AB pair. Measured inference and full-frame RGB evaluation have
completed. Do not quote warmups as measured speedups. Production defaults and native kernel source are
unchanged by this experiment.

### Stage-1 graph-reset ablation

`TURBOCIDER_LTX_RESET_GRAPHS_STAGE1=1` clears the Metal graph dictionaries at
the beginning of Stage-1 while retaining Transformer weights and ANE sessions.
On the same 704x448/97-frame resident ANE setup, request 0/1 denoise was
50.445/77.375 s and request wall was 74.581/92.028 s. Stage-1 was
23.712/42.570 s and Stage-2 26.733/34.805 s. The second request's Stage-2
video and audio tensors were byte-exact to request 0, and Stage-1 video output
and Stage-2 video input were also exact. Clearing graph caches therefore did not remove hot-request
slowdown; it is retained as a diagnostic switch only. Report:
`outputs/transformer-m4max/ltx-resident-graph-reset-97.json`.

| Measured component | GPU seconds | GPU+ANE seconds | speedup |
|---|---:|---:|---:|
| Stage-2 | 30.289 | 27.108 | **1.117x** |
| video FFN (Stage-2 profile) | 11.936 | 8.730 | **1.367x** |
| model load | 4.690 | 18.043 | 0.260x |
| complete request | 96.954 | 108.691 | **0.892x** |

Stage-1 video and Stage-2 video input are byte-exact. Final video latent is
finite, relative L2 0.10658 and cosine 0.99432. All 97 frames decode; mean RGB
correlation 0.97410, MAE 8.264/255, maximum motion-energy relative difference
1.836%. These are reference-difference diagnostics, not perceptual acceptance
or evidence of visible degradation. Actual visual inspection is still needed.
Full-resolution LPIPS AlexNet v0.1 across all 97 frames is **0.08095**;
`ltx-stage2-mlp-int8-97-lpips.json` stores model/source and video hashes.
This metric has no calibrated acceptance threshold here.
The separate evaluator also verifies Stage-1 audio isolation. No all-frame
identity or lip-sync qualification is claimed.

This pilot reproduces a real Stage-2 gain while demonstrating **negative total
request benefit** under component-staged loading. The model-load difference
(13.35 s) exceeds the Stage-2 saving (3.18 s). Retaining the engine object does
not make every model resident across these component-staged requests. Resident
session/cache or load-scheduling experiments are necessary before integration
can be called a user-visible acceleration. Stage-1 timing also differs despite
identical output, so this single AB pair cannot establish timing confidence.

Source audit: `ltx_ane_mlp.m` already binds caller-owned row-major FP16 buffers,
checks returned output identity/pointer and shares I/O by device/shape. It casts
BF16 input before submission, runs the GPU INT8 partial concurrently with ANE,
and joins BF16/FP16 partials afterward. `ane_start` currently creates a pthread
for each prediction and `ane_wait` joins it. A persistent worker is a candidate
ablation, not an assumed speed win. The per-stage logger currently omits its
already-collected ANE pack/overlap/GPU/prediction/join fields; detailed attribution
will need an instrumented follow-up rather than inferring all costs from FFN.

## Resident-session follow-up (2026-09-14)

The first resident experiment uses two consecutive same-configuration requests
in one engine with explicit text-conditioning cache. Separate GPU and GPU+ANE
processes are run serially, not as an interleaved statistical comparison.

| Route/request | Denoiser cache hit | Stage-1 s | Stage-2 s | request s |
|---|---|---:|---:|---:|
| GPU first | no | 23.466 | 30.190 | 70.190 |
| GPU second | yes | 42.530 | 30.230 | 83.718 |
| ANE first | no | 25.124 | 26.960 | 107.012 |
| ANE second | yes | 69.874 | 37.384 | 122.164 |

Both second requests hit text and denoiser caches; model-load time is below
one microsecond. ANE first request populated the shared text cache, while GPU
first request used that cache, so first-request totals are not matched cold
comparisons. Second-request denoising nevertheless regresses within each route.
This is a counterexample to treating cache hits as proof of a speedup, not
proof that residency itself is always harmful. Reports:
`ltx-resident-gpu-97.json` and `ltx-resident-mlp-int8-97.json` in the campaign
output root. ANE process peak RSS is 31.08 GB decimal, excluding the separate
video decoder; it is not a total-system peak memory measurement.

Source inspection shows component-staged explicitly frees the Transformer
after denoising. Resident mode keeps it but also retains audio decoder objects;
a new uncached prompt frees the Transformer before Gemma encoding. Thus merely
retaining the engine is insufficient, and keeping every component may create
resource interference. The cause of the observed timing regression is still
unresolved: no device timeline, paging deltas or controlled thermal traces
have yet isolated it.

A default-off ablation, `TURBOCIDER_LTX_RESIDENT_RELEASE_DECODERS=1`, now frees
decoder objects and clears their MLX caches before a resident request while
keeping the Transformer/ANE sessions. Decoder reload cost stays in request
wall time. The native build succeeds. Its two-request ANE test has completed;
do not interpret this implementation as a verified optimization yet. Native
source defaults and model math are unchanged; only the opt-in lifecycle
experiment changes behavior.

With decoder release enabled, request 0/1 denoising is 50.467/74.894 s and
complete request wall is 78.072/87.401 s. The second request has both caches
hit and zero model-load time, yet Stage-1 is 41.987 s and Stage-2 32.908 s
(versus 23.635/26.832 s on request 0). Releasing these decoder objects did not
eliminate the hot-request regression. Its absolute timings improve over the
earlier resident ANE process, but different binary/process order and system
state prevent causal attribution of that improvement to the flag. A matched
same-binary on/off test is required before retaining it as an optimization.
Report: `ltx-resident-mlp-release-decoders-97.json`.

### Persistent ANE-worker ablation (2026-09-14)

The native LTX ANE MLP path now has an opt-in persistent prediction worker,
enabled only with `TURBOCIDER_LTX_ANE_PERSISTENT_WORKER=1`. The default remains
the existing pthread-per-prediction path, so production behavior and dense
defaults are unchanged. The worker is lazy, serializes one prediction at a
time, preserves caller-owned Core ML output backing checks, and is shut down
before model resources are released. This is a scheduling ablation, not yet a
claimed optimization: every MLP block can own a worker and the resulting
thread/resource footprint may offset any pthread creation savings.

The native build completed with the managed MLX SDK. CPU pthread lifecycle,
campaign/sparse evidence and LPIPS input tests pass (`25 passed, 19 subtests
passed`). The lifecycle test compiles the actual scheduler with a mock CPU
prediction, exercises success/failure, duplicate submission, repeated reuse,
and shutdown with an outstanding prediction. It is not a Core ML benchmark.

A same-binary real LTX on/off screening pair has now completed on the machine
listed above: 704x448, 97 frames, 11 steps, seed 42, audio enabled, dense
attention (`quality`), the Stage-2-only INT8 MLP profile, and the cached fox
prompt. Each process retains one resident engine for two requests. Off runs
first, then on; no release-decoder or graph-reset flag is set in either launch.
`TURBOCIDER_LTX_C_PROFILE=1` is set in both, and only
`TURBOCIDER_LTX_ANE_PERSISTENT_WORKER=0/1` differs. Both use the existing shared
conditioning cache, not a cold text-encoder comparison.

| Request | Stage-1 off/on s | Stage-2 off/on s | Stage-2 off/on ratio | Complete request off/on s | Request ratio |
|---|---:|---:|---:|---:|---:|
| first | 23.602 / 23.660 | 26.554 / 26.836 | 0.989x | 73.840 / 72.610 | 1.017x |
| second | 44.422 / 42.710 | 35.265 / 33.278 | 1.060x | 94.878 / 90.234 | 1.051x |

Stage-2 ANE overlap/GPU partial/prediction cumulative timings (ms):

| Worker | Request | Overlap | GPU partial | ANE prediction |
|---|---:|---:|---:|---:|
| off | first | 7862.567 | 6853.624 | 7856.245 |
| on | first | 8021.314 | 6857.023 | 8015.675 |
| off | second | 16504.393 | 15828.821 | 13088.197 |
| on | second | 14444.347 | 13621.250 | 12461.551 |

Interpretation: persistent threads do **not** eliminate hot-request regression.
The first Stage-2 is slightly slower; the second is modestly faster, while its
GPU partial also becomes faster. These observations do not isolate pthread
overhead from scheduling or system-state differences. Keep the flag off by
default. Counterbalanced repetitions and device timelines remain necessary;
no stable 1.06x benefit is claimed from this single pair.

SHA256 checks confirm all five tensor dumps (Stage-1 video/audio, Stage-2 input
video, final Stage-2 video/audio) are byte-exact across off/on at both request
indices. They are also unchanged across request indices. This is evidence of
scheduling parity, not a new dense-relative perceptual-quality qualification;
decoded RGB has not been independently checked for this ablation.

Provenance:

- Raw reports: `outputs/transformer-m4max/worker-off-001.json` and
  `outputs/transformer-m4max/worker-on-001.json`; corresponding directories
  contain the videos and tensor dumps.
- Library SHA256 before and after the pair:
  `3ac1557d9953a54ebf2597c01395b2110133c1c851bfa95d251646c68a69437e`.
- ANE MLP source SHA256:
  `3b82a7b38aff74a252f91f6edd9edd61d6542745364b3094b62143aedcfeaeb0`.
- `scripts/compare_workers.py` checks matching workloads, ANE identities, run
  counts, positive finite timing and tensor hashes; it prints reproducible
  comparison JSON without changing input artifacts. Launch provenance remains
  separate because the original resident benchmark does not record this env flag.
- A runtime snapshot confirms `Mac16,9`, M4 Max, 68719476736 bytes UMA, macOS
  26.6.2/25G83. Other Python/ComfyUI processes were present; their device activity
  was not measured. This is **not** a dedicated-machine qualification. `pmset`
  reported no recorded thermal/performance warnings; a single snapshot is not
  a continuous thermal trace. `vm_stat` is cumulative since boot and cannot by
  itself attribute paging to these requests.

### Attribution follow-up (earlier resident profile)

The native logger now exposes already-collected per-stage ANE MLP pack,
overlap, GPU partial, prediction and join timings. They accumulate across
blocks and are not independent portions to sum into request wall. No compute
operation was changed by this instrumentation. Native build succeeded.
The `ltx-resident-profile-97` two-request run collected these fields in
`ltx-resident-profile-97.stderr.log`; both requests and all four stage records
exist. Shape restoration and ANE stage-slot
selection were inspected; no definite state-restoration defect was identified.

The profile completed with request-0/request-1 denoise **50.515/77.034 s**.
Stage-2 ANE MLP cumulative fields changed from pack/overlap/GPU/compute/join
`117.6/8035.0/6847.0/8027.8/161.3 ms` to
`172.2/16182.7/15053.6/13184.7/150.2 ms`. The ANE prediction and GPU partial
both become substantially slower on the second request, while pack/join remain
small. This points toward shared GPU/ANE resource pressure or runtime state,
not an output-copy bottleneck. Stage-1 also slows sharply, so a Stage-2-only
optimization cannot be validated from resident warm requests until this
cross-stage effect is isolated. Report: `outputs/transformer-m4max/ltx-resident-profile-97.json`.
