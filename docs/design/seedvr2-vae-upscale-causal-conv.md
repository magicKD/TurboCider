# SeedVR2: spatial-upsample causal convolution on fixed Core ML tiles

## Scope (2026-10-15)

The new isolated boundary is
`decoder.up_blocks.2.upsamplers.0.conv`, original FP16 OCTHW
`[256,256,3,3,3]`. It follows the existing spatial projection/bias/shuffle; those
operations, global normalization, attention and temporal phases remain native.
It is not the earlier1x1 projection experiment or the128-output residual bank.
Only subsequent frames are offloaded. Both candidate layers preserve native
GPU FP32 initial-frame folding and the exact first output frame.

The existing direct-frame executor now explicitly supports two independent
full-output signatures. Each has its own model, two reusable IOSurface pairs,
history owners, counters and validation. The GPU packs two old/current FP16
frames directly with the exact spatial halo; no complete temporal concat is
created. The256-output consumer materializes all channels, applies the original
separate FP16 bias rounding and assembles every tile, including ragged borders.
No normalization statistics or hidden states are approximated by independent
crops. All APIs remain public; CPU+NE configuration is not an ANE-only execution
or physical occupancy claim.

The new64-square bank remains `qualified=false`. Manifest SHA:
`877f50e7d97399b3eae2e2c90b6832b8d99bb9d547e699683d37631d2060598b`.
The original128-output bank and original checkpoint are unchanged. No public
request default, model admission estimate, App package or121-frame cap is changed.

## Completed component correctness

Both old/new factory unit tests execute signed ABA original-coefficient FP32
convolutions, ragged17x19/65x71/64-square operands, actual strided/offset inputs,
replicated history aliases, retained output safety and independent full IO word
witnesses. Each checks120,434,688 packed half words; old/new consumers check
11,857,664/23,715,328 words. Wrong-layer banks, incomplete output/bias shapes,
pre-cancelled calls, active cancellation after a real prediction, same-runtime
recovery, actual VAE sink failure/complete9 recovery and zero-active teardown pass.

Complete restored1024/t25 screens compute every output value against the same
saved original latent/pixels. The independent all-FP32 oracle uses the same
serialized input SHA, original coefficients promoted to FP32 and audited FP32
parameters/output; it is not a DiT-FP32 report with FP16 VAE. Every one of25
frames passes the unchanged finite/.05-relative-L2/.999-cosine gate:

| Complete output against original-FP32 VAE | Relative L2 | Cosine |
| --- | ---: | ---: |
| New256-output convolution alone | .0005225678468 | .999999868455 |
| Both128-output residual and256-output upscale convolutions | .0005252472710 | .999999867021 |

The new hook executes6144 predictions/backing hits over24 subsequent frames;
the dual route independently executes that count for **each** layer. Both
initial-frame counts are zero, and initial raw pixels are byte-identical to
native. Results are numerically qualified for this fixture, not generally byte
identical or broad image/video/device qualification.

## Complete-decoder observations

The same v1 probe, original complete25 fixture, frame history, existing upscale
sharing and1-GiB allocator cache produce the following **single observations**.
Timing includes host pixel checks/export, not model load:

| Complete1024/t25 route | Seconds | MLX active maximum bytes | Sampled process footprint maximum bytes |
| --- | ---: | ---: | ---: |
| GPU shared-history control | 55.547361375 | 12,996,152,418 | 14,494,778,448 |
| New upscale convolution only | 54.366604417 | 12,996,152,418 | 14,462,977,272 |
| Both causal convolutions | 54.131275208 | 12,190,583,906 | 13,679,576,288 |

Moving just one region can leave another region as the maximum: upscale-only
does not lower the decoder high water. The dual observation is805,568,512 bytes
below GPU and402,390,772 bytes below the previous hotconv-only active maximum.
This is not a whole-request or hard-RAM bound. The old/new external surface
pools are21,626,880/23,724,032 bytes, excluded from MLX active. Internal Core ML
allocations and allocator/OS caches must not be declared free or added as if
these metrics were disjoint memory categories.

A new complete-decoder ABBA compares the **already optimized hotconv-only**
route with the dual candidate, not merely unoptimized Conv3D. Each process
recomputes all25 frames three times, one cold/two warm, and checks every raw
value, every frame's quality, original first-frame pixels/zero new calls, all
per-layer calls/backings and unchanged latent bytes. Final performance acceptance
requires the terminal complete receipt; in-progress medians or the128/t9 harness
smoke are not1024 speed qualification. Full-request/App timing and memory remain
separate gates.

The complete1024/t25 ABBA has now terminated successfully: four independent
processes, twelve complete fresh decodes, four warm observations per route.
All per-frame gates, native first-frame words/zero calls, original latent bytes
and per-layer call/backing counts pass. The measured component median gate is:

| Warm complete decoder | Prior hotconv-only CPU+NE | Dual CPU+NE |
| --- | ---: | ---: |
| Warm median seconds, host pixel gate included | 55.6823873745 | 54.1915953545 |
| Active maximum bytes | 12,592,974,678 | 12,190,583,906 |
| Maximum sampled process footprint bytes, whole process cold/warm | 14,382,122,064 | 13,767,410,816 |
| External surface bytes | 21,626,880 | 45,350,912 |

This is about2.68% less decoder latency and402,390,772 fewer active bytes versus
the already optimized reference. It is a machine/fixture/component result, not
a portable improvement or whole-request/App speed qualification. Process maxima
include nonwarm portions and caching differences; they are not disjoint from
active/surface memory or enforceable bounds. The small128/t9 harness's4-byte
active difference is not a meaningful memory gain.

## Complete-request raw quality

Fresh full-request libraries execute paired GPU/dual1024/t25 requests with
bounded diagnostics, original source/seed42 and the same library for both modes.
Both original bank hashes and complete compiled trees are checked. Actual
condition/velocity/latent hashes (including dtype/shape/bytes) are identical.
Every source frame is emitted and checked; both first-frame counters are zero.

Against native, raw FP16 relative-L2 is .0002308082539, cosine .999999973364
and maximum absolute difference .0048828125. Actual preencoder RGB24 maximum
difference is1, MAE .007144241 and PSNR69.591242 dB. Concatenated captured bytes
match each request's real encoder-input hash. Every raw and exported frame passes
the unchanged gate. H264 decoded maximum difference is95 and PSNR39.456661 dB;
do not attribute that entire lossy-codec difference to VAE arithmetic or infer
that95 versus the earlier138 demonstrates a perceptual improvement.

These captured requests include diagnostic file/hash work and are NOT speed or
peak-memory benchmark evidence. Broader media/temporal/low-memory-machine quality
and a qualified public artifact remain separate requirements.

## Request integration

The compile-isolated session has an independent
`TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL_MANIFEST` selector and
`TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL_COMPUTE=ne|cpu` control. The manifest SHA
above and every compiled-tree file/size are pinned, with shared rejection of
symlinks/unregistered files. Old-only, upscale-only and dual selection have
independent counters/recipes/resident components. Ordinary libraries reject the
new selector; no public VAE field/picker or automatic routing is promoted.

The framework observer requires explicit
`--experimental-upscale-causal-manifest`; inherited unrecorded selectors fail.
Existing raw diagnostics bind each selected bank separately, all actual
preencoder bytes and the condition/velocity/latent hashes. The App test frontend
has per-layer expectations. Fresh source-verified experimental/ordinary libraries
now pass fourteen actual App requests using the same SHA-bound Swift frontend.
Both512/1024 images and complete25-frame/24-FPS videos, job persistence and
resident/staged/resident switches pass. Each video executes1536/6144 predictions
per selected layer; images execute none and retain the original GPU recipe.
Ordinary exports match the prior ordinary App on every pixel. Dual-versus-
ordinary images and PNG alpha are exact; every exported video frame passes the
unchanged gate. This is an App-state/job-store workflow, not GUI clicking or
stable packaging.

Same-source, capture-disabled single1024/t25 App native request observations.
Both routes explicitly use staged weights,1-GiB cache, source/sink streaming,
frame history and existing upscale-history sharing; these are not all-default
request settings:

| Whole request | Ordinary GPU | Dual CPU+NE |
| --- | ---: | ---: |
| Native request seconds | 90.747781542 | 89.481431125 |
| MLX active maximum bytes | 13,014,502,498 | 12,235,730,826 |
| Sampled process footprint maximum bytes | 14,280,377,880 | 13,570,311,728 |

These include import/encode/DiT/decode/export, not model constructor/readiness
or all Swift UI/history overhead. The778,771,672-byte active reduction is smaller
than the805,568,512-byte decoder-only reduction: other request phases also bound
the maximum. This does not by itself identify the exact limiting phase. The
time observations are not paired warm full-request speed qualification; do not
turn the2.68% decoder result into an overall App speed claim.

Seven actual same-engine lifecycle cases reject a tampered owned copy of the
new compiled bank, cancel after genuine subsequent-frame predictions, recover,
turn both selectors off, enable upscale-only, enable hot-only, then reenable
both across resident/staged switches. All successful cases preserve25 frames/
24 FPS and exact per-layer counts. Original sources/banks and an existing
private destination are unchanged, staging drains and explicit unload reports
zero MLX active allocations. This is cancellation after completed predictions.
The separate native component test covers cancellation between tile predictions
and same-runtime recovery; neither test interrupts Core ML's synchronous call.

The ordinary library actually rejects each of the old selector, capture selector
and new upscale selector, then prepares on the same engine without inference
and unloads to zero active bytes. It passes112 SeedVR2 CPU/contract tests,
sixteen additional build-identity/release tests and actual public release flag/
byte/link checks. Experimental packaging remains rejected; no public/default
routing is promoted. No16-GiB admission or longer-model-clip support is inferred
from these64-GiB-machine results.

## Records and reproduction

Generated artifacts stay under ignored `models/`, `build/` and `outputs/`:

- `models/coreml/seedvr2-vae-upscale-causal256-64x64-probe-v1/`.
- `outputs/seedvr2-upscale-causal-real1024-t25-{gpu,upscale,dual}-v1-20261015/`.
- `outputs/seedvr2-upscale-causal-real1024-t25-{upscale,dual}-fp32-v1-20261015.json`.
- `outputs/seedvr2-upscale-causal-decode-128-t9-abba-v1-20261015/`:
  complete harness smoke, not1024 speed/memory qualification.
- `outputs/seedvr2-upscale-causal-decode-real1024-t25-abba-v1-20261015/`:
  completed balanced cohort; complete original25 frames on every iteration.
- `outputs/seedvr2-upscale-causal-raw-request-1024-{gpu,dual}-v1-20261015/`:
  paired complete request and actual preencoder captures, not speed benchmarks.
- `outputs/seedvr2-upscale-causal-raw-request-1024-quality-v1-20261015.json`.
- `outputs/seedvr2-upscale-causal-complete128-t9-cpu-v1-20261015/`:
  CPU-only graph control with8 subsequent frames/32 predictions/backing hits;
  the rest of the VAE and pack/assembly remain GPU, not an all-CPU VAE test.
- `outputs/seedvr2-upscale-causal-app-{dual,ordinary}-e2e-v1-20261015/generation-report.json`:
  fourteen capture-disabled App requests on current experimental/ordinary libraries.
- `outputs/seedvr2-upscale-causal-app-export-gates-v1-20261015.json`.
- `outputs/seedvr2-upscale-causal-full-request-lifecycle-v1-20261015/report.json`:
  seven cases, including all four selections and typed cancel/recovery.
- `outputs/seedvr2-upscale-causal-ordinary-rejection-v1-20261015/report.json`:
  three actual selector rejections, then prepare recovery, no inference.
- `outputs/seedvr2-upscale-causal-final-gates-v1-20261015.json`:
  twelve terminal receipts, rehashed actual raw/exported frames and both compiled
  trees, current native source snapshots, actual library/runtime IDs and the
  identical Swift frontend hash. Identity queries run in separate processes;
  this index is not whole-objective or public-bank qualification.

Export with `tools/coreml/export_seedvr2_vae_conv.py --channels 0 --rows 64
--cols 64 --layer decoder.up_blocks.2.upsamplers.0.conv` and the original
checkpoint/fresh output directory. Build probes into a fresh
`TURBOCIDER_VAE_CAUSAL_PROBE_OUT` with
`tools/native/build_seedvr2_vae_causal_conv_probe.sh`. Spatial screens accept
`--upscale-causal-manifest` independently or together with
`--causal-conv-manifest`. Balanced cohorts accept `--reference-route ne
--candidate-route dual --upscale-manifest ...` with the original hotconv bank.

The objective remains broader than these two operators: reusable native
workspaces, fused normalization/packing with original full-frame statistics,
broader VAE stages/encoder scheduling, qualified public choices, measured useful
GPU/ANE cooperation, actual lower-memory hardware and complete-context long
videos beyond121 remain open. This experiment does not complete that objective.
