# SeedVR2 integration and acceptance

Task started October 13, 2026. This document is a work log, not a completion claim.

## Required outcome

- Download `Comfy-Org/SeedVR2`'s original `seedvr2_3b_fp16.safetensors`
  from ModelScope, with the matching SeedVR2 VAE and verified SHA-256s.
- Integrate real image and video super-resolution into TurboCider's native
  runtime. No Python/ComfyUI subprocess masquerading as a native executor.
- Exercise real images and videos, including 512×512 output and 512×512 input
  enlarged to 1024×1024. Reuse verified Z-Image/LTX media where practical.
- Compare complete pure-GPU requests to the pinned local ComfyUI implementation
  on the same Mac, using the same original weights, inputs, scale, sampler,
  steps and color-processing policy. Report cold and repeated warm results
  separately; require pure GPU to be no slower in the matched warm cohort.
- Attempt a real GPU+ANE partition and validate output quality and actual
  partition execution. Do not claim ANE occupancy from a compute-unit setting.
- Preserve image detail and video temporal behavior. Validate intermediates
  against the independent reference and visually inspect complete outputs.

## Source identity

ModelScope revision: `8d30a39cd60f07cfff11306406f97021a50ff0eb`.
Upstream Apache-2.0 architecture reference:
`ByteDance-Seed/SeedVR` commit `e4de8c24441a67e1b7df56abea10645059bb1185`.
The local pinned ComfyUI v0.39 installation is an external validation oracle,
not a production dependency. Its specific commit and package versions must be
captured in benchmark records.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| diffusion_models/seedvr2_3b_fp16.safetensors | 6784268336 | 98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660 |
| vae/seedvr2_ema_vae_fp16.safetensors | 501324814 | 20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1 |

`tools/models/download_seedvr2.py` publishes final filenames only after size
and SHA checks; existing downloads can be finalized with `--verify-only`.

## Current evidence

The clean initial worktree had no SeedVR2 module. Both ModelScope downloads
finished and final size/SHA-256 verification passed. The checkpoint
header confirms 32 transformer blocks, hidden width 2560, 20 attention heads,
SwiGLU width 6912, 10 dual video/text blocks followed by shared blocks, and
embedded positive/negative conditioning. The VAE has 16 latent channels,
8× spatial and 4× temporal downsampling.

Native geometry, unquantized DiT, causal VAE and one-step restoration pipeline
are implemented and now registered as `seedvr2-3b` with `image.upscale` and
`video.upscale` through the public CLI/C ABI/Swift SDK. The App includes source
image/video import and operation-dependent PNG/MP4 output handling. The model
library has a matching ModelScope/Hugging Face recipe and file inspection rule.
The initial public route was GPU-only. The later manual qualified-bank
GPU/CPU+NE integration, complete-request gains and App release audit are
recorded chronologically below; auto selection remains GPU-only.

Verified component evidence (raw records under ignored `outputs/`):

| Check | Result | Raw record |
| --- | --- | --- |
| Download safety | Three tests passed; both full files hashed and published | `models/Comfy-Org-SeedVR2/source.json` |
| Adaptive/shifted windows | 18 geometry combinations match independent formulas; each token covered exactly once | `tests/native/test_seedvr2_geometry.py` |
| 128² one-frame VAE | Latent cosine 0.999999666; decoded cosine 0.999999892 | `seedvr2-vae-128-t1-parity-20261013.json` |
| 128² nine-frame causal VAE | Latent cosine 0.999999659; decoded cosine 0.999999910 | `seedvr2-vae-128-t9-parity-20261013.json` |
| FP32 complete DiT, original FP16 weights | Captured embeddings, blocks 0/1/9/10/31, velocity and sampling all pass cosine ≥0.99999 / relative-L2 ≤0.005; velocity cosine 0.999999825 | `seedvr2-dit-128-t1-fp32-parity-20261013.json` |
| BF16 DiT | Runs all 32 blocks, but the current intermediate gate fails; do not call it qualified | `seedvr2-dit-128-t1-v2-parity-20261013.json` |
| Real image, 256→512 | Complete native restoration is finite and visually restores fur/snow detail. Decoded pixel cosine to the independent Comfy component pipeline is 0.999884845, but latent relative-L2 0.0564 still fails the generic 0.05 gate | `seedvr2-restoration-output-256-to512-t1-parity-20261013.json` |
| Real image, 512→1024 | Complete native restoration generated and rendered; independent 1024-output comparison remains pending | `seedvr2-native-restore-512-to1024-t1-20261013/` |

The independent oracle is pinned ComfyUI commit
`b0b743566f65daafc423b4fea8a2fbda94b3384a`, running PyTorch 2.14.0 on MPS
with automatic CPU fallback forbidden. The VAE uses FP16; default DiT
projection math uses BF16 with original FP16 checkpoint weights, matching
the reference's manual-cast route. FP32 is a diagnostic, not the proposed
production speed route. Native biased projections now use a fused bias
epilogue instead of separately rounding matmul and addition. BF16 rounding
differences still accumulate in late blocks; this needs either stronger
numerical alignment or a broader real-media quality qualification, not a
claim that the failing intermediate gate passed.

Host hardware was verified with `sysctl`: Apple M4 Max,
68,719,476,736 bytes (64 GiB) unified memory.

The nine-frame fixture is a rolling image for causal codec testing, not real
video acceptance. Subsequent real LTX video restoration and frame/FPS checks
passed as recorded below; matched speed and broader quality remain pending.

### Public integration and real video — subsequent progress

- Native-only build completed after the public module/session and media
  integration. The dedicated public contract suite passes all five tests,
  including schema-2 dynamic image/video output kinds and rejection of
  unsupported routes. Metadata correctly reports no text encoder instead of
  inheriting Qwen encoder labels.
- Full Swift/App build completed. The dedicated SeedVR2 App test passed source
  image/video staging, empty native prompt, source input roles, 25-frame/24-FPS
  binding, mismatched-frame rejection and legacy asset/draft persistence.
- The public CLI generated `seedvr2-cli-image-1024-v1-20261013.png` from the
  existing 512-square Z-Image sample. Complete first request including model
  loading/import/export: 6.739 s. This is a single observation, not a median.
- A real LTX parent record was checked: `ltx-base-current-v2-three-20261011/0-gpu/0.mp4`,
  model `ltx-2.5-distilled`, 704×448, 97 frames, 24 FPS. A deliberately prepared
  512-square, 25-frame / 24-FPS silent crop is
  `seedvr2-ltx-source-512-t25-v1-20261013.mp4`. This is derived from an existing
  verified LTX generation, not newly generated footage or a rolling-image test.
- Public native super-resolution restored every frame to 1024-square and
  exported MP4 with independently verified 25 frames, 24 FPS and 1.041667 s
  duration. First unbounded-VAE observation: 100.404 s, MLX peak 52,104,617,014
  bytes. The output was regenerated during the memory optimization; retain
  this as an observed timing, not an available before/after full-video pair.
- The encoder full-resolution head and decoder spatial-only tail now run
  frame-by-frame while carrying causal histories. The nine-frame VAE comparison
  still passes: latent cosine 0.999999659, decoded cosine 0.999999910,
  relative-L2 0.000831 / 0.000427. Public real-video observation after this
  change: 95.864 s, MLX peak 37,378,420,246 bytes. The available full video is
  `seedvr2-cli-video-1024-t25-frame-bound-v1-20261013.mp4`. These cold, unpaired
  observations demonstrate reduced peak memory, not a qualified speed gain.

The maintained public usage guide is `docs/public/SEEDVR2.md`.

### Complete framework prechecks

`tools/validation/seedvr2_framework_screen.py` observes real native C API or an
owned loopback ComfyUI server. It does not connect to a user's server. Input
filenames are fresh but source bytes are identical, forcing real media/resize/
VAE encode/sampler/VAE decode/export on every repeat. Model loader caching is
allowed; encoded-latent caching is not. Original model hashes are rechecked.

Image 512→1024, one Euler/simple step, CFG1, no color correction:

| Route | Cold request | Two warm requests | Warm median |
| --- | ---: | --- | ---: |
| Pinned stock Comfy v0.39, MPS SDPA/GPU-only, original FP16 DiT with BF16 manual cast and FP16 VAE | 18.482 s | 15.633 / 15.645 s | 15.639 s |
| Public native GPU | 6.734 s | 5.917 / 5.908 s | 5.913 s |

This is about 2.65× in this precheck. Both reports explicitly retain
`qualification_passed=false`: no reverse-order/ABBA paired cohort, full 512
output coverage or completed video speed acceptance exists yet. The complete
ComfyUI process exited and its loader-cache-only execution was checked.
Raw directories: `seedvr2-comfy-sdpa-image1024-precheck-v1-20261013` and
`seedvr2-native-image1024-precheck-v1-20261013`. This stock reference uses the
existing full dependency environment (PyTorch 2.12.1, torchvision 0.27.1),
whereas earlier independent component probes used PyTorch 2.14.0.

The stock full-video reference failed in VAEEncode with
`Tried to instantiate dummy base class Event`: `CausalMemoryCache.park()` calls
`torch.cuda.Event()` when a channels-last MPS frame sequence is parked. The
failure is recorded in `seedvr2-comfy-sdpa-video1024-t25-precheck-v1-20261013`;
the owned server terminated. It is not a speed observation.

A task-owned copy, `.deps/comfy-seedvr2-mps-cache-20261013`, changes only
`_offload_caches_for` to return false on non-CUDA devices. This keeps original
precision cache tensors on MPS; model math, steps, nodes and weights are
unchanged. The portable patch is
`tools/validation/seedvr2_comfy_mps_cache.patch`, and the observer records the
entire local diff plus VAE source SHA. The original pinned checkout and user
installation/server remain untouched. The first launch request was not
executed because the approval-review service returned a temporary 503; the
normal approval retry succeeded. Its full-video precheck completed under
exec session `82656`, raw directory
`seedvr2-comfy-sdpa-video1024-t25-mps-cache-precheck-v1-20261013`: 298.375 s for
the first complete request, including genuine media import, VAE encode,
sampler, VAE decode and MP4 export. All 25 frames, 24 FPS and 1024-square
dimensions passed independent checks; the owned server terminated. This is
one cold compatible-reference observation, not a warm median or paired speed
gate. Do not compare it to the 95.864-s native observation as a qualified
universal 3.11× result. No process from this precheck remains to wait on.

General regression suite observation: 116 tests, five failures and one fixture
skip. The failures involve pre-existing/stale Qwen encoder capability
expectations, source-string LoRA/Qwen assertions and a hardware-sensitive
profile-error expectation. The tested backend/Qwen source files and this
general suite are unchanged from HEAD; do not silently amend them to claim a
green repository. The dedicated SeedVR2 contract/App suites pass separately.

Next: complete matched image/video reverse-order warm cohorts at 512 and 1024,
strengthen real-media quality and shared-noise checks, and qualify an
identity-bound GPU+ANE FFN partition. The compatible single cold video
reference already ran. Silent video and raw/no-color-transfer remain explicit
boundaries.

Single component-pipeline observations including intermediate dumps:
256→512 encode/DiT/decode = 0.363/0.492/0.790 s;
512→1024 = 1.237/1.781/3.159 s. These omit final media export and are **not**
paired warm medians or proof of the requested ComfyUI speed gate. The complete
owned workflow prechecks above ran; final paired qualification has not.

### VAE optimization and tested GPU baseline — October 13, 2026

The new default is `singleton2d`: a replicated initial singleton frame folds
its temporal filter into a spatial convolution; temporal-kernel-one operators
also use a spatial convolution. Other temporal convolutions remain 3D and
carry their original causal histories. FP32 convolution accumulation avoids
nonfinite results observed in the first experimental FP16 2D implementation
at 512/1024. Those failed runs are not valid speed/quality evidence.

The following memory and precision changes are now in the public route:

- A single image no longer retains histories that no future slice will read.
- Typed VAE scaling constants keep encoded latents and decoder activations
  in FP16. Earlier C++ scalar expressions inadvertently promoted the decoder
  to FP32. Thus the old `fp16_vae` metadata did not describe every decoder
  activation; preserve earlier reports as observations, not proof of a fully
  precision-matched codec benchmark.
- VAE attention uses FP32 scores with 512 query rows and every spatial key,
  one frame at a time. This is exact full spatial attention, not spatial
  tiling/truncated context, and bounds its score workspace.
- Request-local allocator retention defaults to 4 GiB and is cleared on exit.
  The previous cache limit is restored. A 512 MiB cap was measured but slowed
  the 1024 codec substantially; 4 GiB balances the measured 2/4/8 GiB sweep.
  Resident model tensors remain available between requests.

All updated VAE comparisons use cosine >=0.99999 and relative-L2 <=0.005.
128-square t1/t9 pass against the independent Comfy FP16 VAE. Independent
512/1024 single-frame checks pass too: latent relative-L2 0.001398/0.001810,
decoded cosine 0.999999887/0.999999883. The 1024 t9 comparison between updated
3D and singleton2D codecs passes with decoded cosine 0.999999960. This last
check is native recipe parity, not an independent real-video oracle.

Before/after native image observations, same 512-square source to 1024:

| Native route observation | Warm median, full request | MLX active peak |
| --- | ---: | ---: |
| Earlier frame-bounded GPU precheck | 5.913 s | 29,871,577,598 bytes |
| Updated GPU precheck, default singleton2D / 4 GiB cache | 3.913 s | 17,799,614,850 bytes |

Updated cold request is 4.982 s; warm requests are 3.901/3.926 s. Raw record:
`seedvr2-native-image1024-vae-opt-v1-20261013`. The observer now preserves the
library SHA-256 and runtime build manifest. These remain non-paired prechecks
with `qualification_passed=false`, not a final ComfyUI speed gate.

The App's first real generation test caught a missing `warmup=false` field in
the native result: valid inference/PNG export was rejected by the App's strict
image transaction. The result contract was corrected without weakening image
validation. New App source selection also refreshes frame/FPS binding; retained
video assets cannot leak into image-only operations.

The complete App workflow test subsequently passed with real imported media,
native inference, validated PNG publication, video import/export checks,
persistent history and final unload. It exercises the actual App
`StudioState`/`NativeJobStore` chain, not manual UI clicks. Separately, the
rebuilt graphical App opened an on-screen window under the isolated test
directory with the SeedVR2 video draft, 1024-square canvas, 25 frames and 24 FPS.

| App full request | Request observation | MLX active peak |
| --- | ---: | ---: |
| Image 512 output | 1.520 s, initial model load | 9,937,447,930 bytes |
| Image 512->1024 | 3.883 s, reused model | 17,801,004,842 bytes |
| Video 512 output, t25 / 24 FPS | 21.471 s, reused model | 11,671,672,546 bytes |
| Video 512->1024, t25 / 24 FPS | 91.161 s, reused model | 24,814,400,554 bytes |

Raw record: `seedvr2-app-real-e2e-vae-opt-v2-20261013/generation-report.json`.
Both videos independently decode to 25 frames, 24 FPS and 1.041667 s via
ffprobe. Updated image plus first/middle/last video frames were visually
inspected against the source. The 91.161-s reused-model observation is not a
paired speed gain over the earlier 95.864-s cold video. The observed video
active peak fell from 37,378,420,246 to 24,814,400,554 bytes; this is not a
whole-process/whole-Mac peak measurement. MLX active peaks exclude allocator
cache and non-MLX/Core ML allocations; results now state that scope explicitly
and reset the active peak for each request.

The updated shared-noise 256->512 restoration remains finite and improves
decoded cosine slightly to 0.999889351. Latent relative-L2 is 0.054731, so the
existing generic 0.05 intermediate gate still fails. Do not mark complete
DiT/real-media quality qualification as passed. Raw record:
`seedvr2-restoration-512-vae-opt-v2-parity-20261013.json`.

The latest host-access regression run is 116 tests, four pre-existing failures
and one fixture skip; the earlier hardware-dependent profile failure does not
reproduce with host access. The remaining names are
`test_gpu_ane_capabilities_are_optional_and_encoder_scoped`,
`test_in_memory_lora_failure_discards_partially_modified_weights`,
`test_native_inference_time_lora_keeps_packed_weights`, and
`test_qwen3_hybrid_quality_gate_is_explicit_and_telemetry_is_recorded`.
Their Qwen/LoRA capability/source expectations and relevant implementation
files are unchanged from HEAD. Dedicated SeedVR2 download/geometry/public
contract checks, App real generation and Studio behavior regressions pass;
installation inspection regressions pass. A bare Studio-model test invocation
was rejected for missing its required model/output arguments and is not a pass.

GPU+ANE groundwork is intentionally not enabled or included in this GPU
baseline commit. A block-0 prefix compiled and predicted once with no output
copy. Its compute plan prefers Neural Engine for the three convolutions,
SiLU and multiplication; this is planned assignment, not physical runtime
occupancy or end-to-end benefit. Source-weight parity, identity-safe manifest
integration, wider block coverage and full-media qualification remain pending.

Next optimization priority: the new real 1024 video spends 22.389 s in encode
and 55.697 s in decode versus 12.885 s in DiT. Investigate bounded convolution
workspaces/fused normalization first, then measure GPU/ANE module allocation
and parallel FFN joins without sacrificing the validated GPU fallback.

### Fused normalization and bounded-convolution experiments — subsequent progress

The default normalization now uses a native Metal kernel: per-frame 32-group
FP32 mean and centered variance, followed by affine/activation without
full-frame FP32 transposes or affine tensors. Small chunk-major statistics
preserve full-frame context. Independent operator tests cover ten convolution
geometries and ten normalization cases, including constant/near-constant,
large-amplitude and non-contiguous inputs. All pass cosine >=0.99999 and
relative-L2 <=0.003. Convolution maximum relative-L2 is 0.000033214; norm
maximum is 0.000584063. Record: `seedvr2-conv-norm-unit-v3-20261013.json`.

The full VAE with fused normalization passes independent 128 t1/t9 and 1024
t1 Comfy comparisons at cosine >=0.99999 / relative-L2 <=0.005. The 1024 t1
decoded cosine is 0.999999886; latent relative-L2 is 0.0017565. The 1024 t9
native before/after comparison passes with decoded cosine 0.999999952 and
latent relative-L2 0.0012620. Same-input, 4-GiB-cache component observations:

| 1024-square / 9-frame codec | Warm encode/decode sum | MLX active peak |
| --- | ---: | ---: |
| singleton2D + original MLX norm | about 28.030 s | 17,419,572,050 bytes |
| singleton2D + fused Metal norm | about 25.442 s | 14,332,562,250 bytes |
| bounded direct convolution + fused norm, two asynchronous tiles | about 31.367 s | 13,309,152,090 bytes |

The bounded direct convolution is diagnostic, not the default: it preserves
causal history and original stencil operands, with up to two 256-MiB column
workspaces in flight. It is not spatial VAE tiling and does not truncate
normalization/attention context. The initial synchronous version was slower;
double-buffered submission reduced that overhead but remains slower than the
default. For 1024 t1 with original norm, it lowers active peak from
10,993,750,870 to 5,149,659,986 bytes but increases codec time from about 1.79
to 2.55 s. Keep this explicit tradeoff; do not label it a universal speed win.

Updated public full-request image precheck, source 512 -> output 1024:
4.539 s initial request, 3.341/3.359/3.369 s warm, median 3.358822 s; active
peak 16,489,268,410 bytes. Source/library/build identities are preserved in
`seedvr2-native-image1024-metalnorm-precheck-v1-20261013`. This remains a
single-route precheck, not paired ComfyUI qualification.

Real App generation again passes all four 512/1024 image/video requests,
strict PNG publication, source frame/FPS binding, history and final unload.
Both MP4s independently decode to 25 frames / 24 FPS / 1.041667 s via ffprobe.
Record: `seedvr2-app-metalnorm-real-e2e-v1-20261013/generation-report.json`.

| App request | Request observation | Active MLX peak | Sampled process footprint peak |
| --- | ---: | ---: | ---: |
| Image 512 | 1.918 s, first model load/JIT | 9,622,477,706 bytes | 12,703,324,208 bytes |
| Image 1024 | 3.472 s, reused model | 16,490,574,330 bytes | 19,881,154,656 bytes |
| Video 512 t25 | 20.677 s, reused model | 10,899,936,146 bytes | 14,979,274,872 bytes |
| Video 1024 t25 | 84.163 s, reused model | 21,727,425,594 bytes | 25,320,560,808 bytes |

A request-local read-only observer samples `task_vm_info.phys_footprint` and
`MTLDevice.currentAllocatedSize`. Nominal wait interval is 20 ms; samples also
take time, and short-lived peaks can be missed. The two figures overlap and
must not be added. These are observed lower bounds, not enforced budgets or
whole-Mac peaks. The 1024 video has 2905 observations, no observer failures,
and sampled Metal peak 25,012,125,696 bytes. Prior reports did not measure this
process-footprint scope, so do not compare their active-only numbers to it.

The new image and first/middle/last video frames were visually inspected.
Before/after raw exported RGB comparisons have image cosine
0.999990499/0.999991761 at 512/1024; three sampled video frame cosines are
0.999924724/0.999884720/0.999829442. Small encoder rounding changes can
propagate through BF16 DiT; these observations do not prove full independent
video quality acceptance or eliminate the previously failing strict latent
gate. Preserve that pending requirement.

`tools/validation/seedvr2_paired_cohort.py` now orchestrates native/Comfy/
Comfy/native cohorts, one cold and at least three full warm requests each,
with immutable source, model, binary/build and Comfy-version/patch identities.
It records a workload-specific median speed gate, not an overall completion
claim. Full image/video paired runs and GPU+ANE qualification remain next.

### Balanced image speed cohorts and hybrid component progress

The initial paired runner stopped before any Comfy server launch because its
native Python environment lacks aiohttp. That failed record is preserved at
`seedvr2-image512-paired-sdpa-v1-20261013`; it is not paired speed evidence.
The runner now preflights the observer dependencies and uses the explicit
Comfy dependency interpreter for Comfy observation, leaving native inference
and all workflow/timing settings unchanged.

Both fresh ABBA cohorts completed, each with two independent processes per
route, one cold and three genuinely executed warm requests per process:

| Full image workload | Native warm median (6 requests) | Stock Comfy MPS SDPA/GPU-only warm median (6 requests) | Ratio |
| --- | ---: | ---: | ---: |
| Real Z-Image-derived 256->512 | 0.819821 s | 3.974071 s | 4.8475x |
| Real Z-Image 512->1024 | 3.368968 s | 15.631517 s | 4.6399x |

Raw records: `seedvr2-image512-paired-sdpa-v2-20261013` and
`seedvr2-image1024-paired-sdpa-v2-20261013`. Source/model/library/build and
Comfy version/local-code identities remain unchanged; Comfy cached only model
loaders on warm runs, not encode/sampling/decode/export. Both owned servers
exited. The workload-specific speed gates pass. Overall qualification remains
false: default-Comfy controls, full video paired results, broad shared-noise
quality and GPU+ANE are separate outstanding requirements. Do not generalize
these ratios to arbitrary resolutions/frame counts/hardware.

The identity-bound hybrid manifest reader now verifies the original checkpoint
identity, SwiGLU/operand/compute-unit recipe, row/channel/block coverage, safe
relative paths, exact compiled file sets, every size/SHA and canonical tree
digests. This code remains experimental and is not in the public session.
All 32 static 512-row / 2048-channel prefixes were exported successfully into
the ignored artifact bank. No original/user Comfy installation was changed.

Block-0 synthetic FFN at 4096 rows executes eight CPU+NE predictions per call.
Three warm GPU observations are 0.031137/0.031140/0.031129 s; hybrid observations
are 0.026782/0.026117/0.026389 s. Relative-L2 is 0.00424344, cosine 0.999991.
Record: `seedvr2-hybrid-ffn-block0-4096-v1-20261013`. This is a one-block
microbenchmark, not whole-model speed or proof of physical ANE occupancy.
The available developer tools do not contain xctrace; compute-plan preference
and successful CPU+NE prediction must not be labeled hardware occupancy.

Complete component restoration at 512 executes all 32 video FFN prefixes,
64 Core ML calls and zero output-copy bytes, with GPU text FFNs. Output is
finite; the VAE condition exactly matches the fixed-input/noise GPU control.
Decoded cosine is 0.999906551, pixel relative-L2 0.0136724. Velocity relative-L2
is 0.0342650. Latent relative-L2 0.0458219 passes its 0.05 limit, but latent
cosine 0.998949745 fails the existing 0.999 gate. Preserve that failure; do
not enable/advertise the public hybrid route as qualified yet. Records:
`seedvr2-hybrid-restore-512-v1-20261013`,
`seedvr2-gpu-restore-512-hybrid-control-v1-20261013`, and
`seedvr2-hybrid-restore-512-parity-v1-20261013.json`. The hybrid's 0.489-s
component denoise observation includes neither full-request import/export nor
balanced warm runs. All original precision/quality gates remain intact.

A meaningful 256-square / 25-frame / 24-FPS source for 512-output video cohorts
was derived with Lanczos scaling and explicit H.264 CRF18 from the earlier
LTX-derived 512 source. `seedvr2-ltx-source-256-t25-cohort-v1-20261013.mp4`
independently decodes to all 25 frames, 24 FPS and 1.041667 s. The complete
video ABBA observer is running separately; no GPU/ANE experiment overlaps its
timed requests.

### Complete 1024-output video paired speed gate

The immutable 512->1024 / t25 / 24-FPS native/Comfy/Comfy/native cohort
completed with six full warm requests per route. Native warm median is
84.079419 s; MPS-compatible Comfy SDPA/GPU-only is 298.961029 s, ratio 3.5557x.
All outputs independently retain 1024-square geometry, every one of the 25
frames, 24 FPS and 1.041667 s duration. Required import/resize/encode/sampling/
decode/export stages execute on every warm request, with only model loaders
cached by Comfy. Both task-owned servers exited and all bound identities remain
unchanged. Record: `seedvr2-video1024-t25-paired-sdpa-v1-20261013`.

The SDPA/GPU-only workload-specific speed gates now pass for images and real
videos at both 512 and 1024 outputs. The non-CUDA compatibility guard is still
disclosed for video. This does not complete default-Comfy controls, independent
shared-noise full-media quality, or GPU/ANE execution/quality qualification.
The public path remains GPU-only and the overall goal is not marked complete.

### Complete 512-output video paired speed gate

The native/Comfy/Comfy/native 256->512 video cohort completed. Each route has
six real warm requests across two independent processes plus separate cold
observations. Native median is 20.437427 s, the MPS-compatible Comfy SDPA/
GPU-only median is 71.331294 s, ratio 3.4902x. Workload: all 25 LTX-derived
frames, 24 FPS, one Euler/simple step, original weights, no color correction.
Every output independently decodes to 512-square, 25 frames, 24 FPS and
1.041667 s. All required stages genuinely execute on every warm request;
reference caches only the model loaders. Source, binary/build, model and
reference/patch identities are unchanged; both owned Comfy servers exited.
Record: `seedvr2-video512-t25-paired-sdpa-v1-20261013`.

This workload-specific speed gate passes. Its compatibility patch remains the
three-line non-CUDA cache-transfer guard described above; it is not a stock
unpatched MPS video result. Shared-noise quality/default-Comfy controls remain
incomplete. The full 1024/t25 paired speed cohort subsequently completed with
the same immutable native library and isolated reference; see the complete
1024-output video section above. Never overlap GPU/ANE benchmarks during timed
requests.

The earlier fixture generator enlarged before source downsampling; its saved
input tensors remain immutable valid parity fixtures. The generator now
downsamples the original directly and saves the actual source PNG/provenance.
Use fresh v2 fixtures for final real-media acceptance and benchmarks.

### Compact VAE histories, pointwise projections and selective hybrid progress

The new speed-priority defaults keep the singleton2D/Metal-normalization
recipe, but retain causal tails directly instead of copying them out of a
full halo concatenation. The initial replicated two-frame tail exposes a
zero-stride view of one owned frame. Normal convolutions explicitly preserve
the old materialization/lifetime boundary before replacing history; delaying
it initially increased rather than reduced memory. That failed screen remains
in `seedvr2-vae-compact-pointwise-1024-t9-v1-20261013` and is not the final
default. Shortcuts with a 1x1x1 filter flatten positions into one FP16/FP32-
accumulating GEMM, avoiding full-resolution FP32 conv2D operands.

`seedvr2-conv-history-pointwise-unit-v2-20261013.json` passes ten causal
convolution, ten norm, sixteen exact-history and four pointwise cases. Tests
cover noncontiguous inputs, distinct/replicated tails, strides and FP32 folded
coefficients. The 1024/t9 final-default VAE latent and decoded pixels are
bitwise identical to the pre-change control. Independent 128/t9 Comfy codec
parity also passes; this does not repair the pre-existing BF16 DiT quality
failure or constitute independent real-video full-model qualification.

The optional `mixed3d` recipe bounds only noninitial temporal convolutions
with at least 1024x1024 positions and 256 input channels. It does not change
spatial context, normalization, attention or causal phase. Its 1024/t9 codec
screen observes 13,309,152,090 active bytes and about 26.5 s, versus the earlier
full-tiled route's roughly 31.4 s at a similar peak. Decoder relative-L2 to
the pre-change control is 0.00012580, cosine 0.999999992; latent is identical.
These are component screens, not balanced whole-request speed proofs. Keep
this explicit memory/speed tradeoff optional, not the default.

Actual result telemetry declares `vae_history_recipe` and
`vae_pointwise_recipe`. The old control uses `concat`/`conv`; the new default
uses `compact`/`matmul`. Environment settings are bound by the full-request
observer as well. `seedvr2_vae_cohort.py` adds immutable-input/probe/VAE ABBA
codec observations with one cold and three warm repeats per process, actual
recipe checks and host-side tensor gates. The native base Python lacks
safetensors; use the existing validation interpreter. Its CPU gate tests pass
there (seven cases, including refusal to finalize live/incomplete cohorts);
the dependency-free environment explicitly skips them.

The final-default ABBA codec cohort completes with six warm observations per
route: pre-change median 25.495785 s, new median 25.203935 s (1.01158x), active
peaks 14,332,562,250 and 14,198,344,530 bytes, a 134,217,720-byte reduction.
Every candidate latent and decoded tensor is bitwise equal to the original
control. Input, probe and original VAE SHA-256s remain unchanged. Record:
`seedvr2-vae-compact-pointwise-abba-1024-t9-v1-20261013`. All four GPU cohorts
completed before the base interpreter failed its host-side safetensors import;
`host-finalization-failure.json` preserves that failure. The explicit
`--finalize-only` path revalidated all completed raw identities/order/settings/
samples and computed parity in the existing validation interpreter. No GPU
measurement was rerun, replaced or counted from an incomplete cohort. The
observer now preflights the reader before future GPU runs.

The App's real StudioState/NativeJobStore chain passes again at both sizes for
images and complete 25-frame/24-FPS LTX-derived video, including source/FPS
binding, strict image transactions, result recipes, persistent job history and
unload. Both PNG SHA-256s match the previous App outputs exactly. Record:
`seedvr2-app-compact-pointwise-real-e2e-v1-20261013/generation-report.json`.
The 1024 video observes 83.3219 s and 21,593,207,870 active MLX bytes, versus
the previous 21,727,425,594-byte observation. Sampled process footprint is
24,613,001,384 bytes, sampled Metal 25,013,895,168 bytes, 2862 observations,
zero failures. Sampling is a lower bound and overlaps; do not add those
figures or infer a system-wide/hard peak budget. Image peaks are unchanged;
this is not a universal memory-reduction claim or a new paired Comfy gate.

The optional mixed3D App chain also passes all four real-media generation,
frame/FPS/history/unload checks, and telemetry confirms `mixed3d` for each
request. Its 1024/t25 video observes 86.5521 s / 20,704,016,710 active bytes,
versus the default's 83.3219 s / 21,593,207,870 bytes: roughly 0.89 GB less
active allocation for roughly 3.9% more time in these App observations. Its
sampled footprint is 24,720,038,200 bytes, not lower than the default sample;
do not promise reduced whole-process physical peak from the active-memory
change. Sampled Metal peak is 24,468,635,648 bytes. Complete decoded H.264 RGB
comparisons (all 25 frames) give cosine 0.999897811 / relative-L2 0.0142962,
maximum 8-bit channel difference 106; outputs are not bitwise equal. First,
middle and last contact-sheet frames were inspected. This compressed-media
before/after witness is not independent shared-noise model parity. Record:
`seedvr2-app-mixed3d-real-e2e-v1-20261013/generation-report.json`. The mode
remains optional; the faster singleton2D route stays default.

Selecting even-numbered video FFNs (16/32) in the experimental CPU+NE prefix
probe now passes the existing 512 fixed-input/noise tensor gate against a
matching pure-GPU probe: latent cosine 0.999185125 / relative-L2 0.0403741,
pixels cosine 0.999927184 / relative-L2 0.0120730. VAE condition is identical.
Each restoration executes 32 Core ML predictions, with no output-copy bytes;
four repeats execute 128 predictions in total. The all-32 failure above still
stands. Records: `seedvr2-hybrid-even16-restore-512-v1-20261013`,
`seedvr2-gpu-restore-512-hybrid-control-v2-20261013`, and
`seedvr2-hybrid-even16-restore-512-parity-v2-20261013.json`.
The observed warm denoise is about 0.466 s versus 0.481 s, but those component
screens are not balanced full-media cohorts. Physical ANE occupancy and
real-video quality/speed are still pending. The public route remains GPU-only;
experimental hybrid source is not part of this tested VAE stage's commit.

The matching 1024 single-image component check also passes the unchanged
0.999-cosine / 0.05-relative-L2 gates: latent cosine 0.999316683 / relative-L2
0.0369639, pixels cosine 0.999966378 / relative-L2 0.0082030, identical VAE
condition. Each hybrid restoration makes 128 Core ML calls (512 across four
repeats), with zero output-copy bytes. Warm component denoise medians are
1.69639 s hybrid and 1.73573 s GPU. These are separate route screens, not ABBA
full-media speed proof or physical ANE occupancy evidence. All-32 coverage
remains unqualified; only even-16 coverage passes these two image fixtures.
Records: `seedvr2-hybrid-even16-restore-1024-v1-20261013`,
`seedvr2-gpu-restore-1024-hybrid-control-v1-20261013`, and
`seedvr2-hybrid-even16-restore-1024-parity-v1-20261013.json`. Real-video
shared-noise quality, matched complete-request speed/memory and App hybrid
integration remain required before public GPU+ANE enablement. The pre-existing
independent BF16 DiT qualification failure also remains unresolved.

### October 14 precision/reference audit (not a public-default change)

The independent component oracle previously set `sys.argv` without enabling
Comfy's argument parser. Stock `comfy.options.args_parsing` is false by default,
so those runs used MPS's default `attention_sub_quad`, not the intended SDPA.
Their numerical failures remain valid default-attention observations and are
not relabeled as passes. The full-request speed cohorts launched real Comfy
servers through `main.py`; this standalone-oracle bug does not invalidate those
server measurements. A fresh explicitly parsed default run is bitwise equal
to the old 128 fixture: `seedvr2-oracle-default-equivalence-v1-20261014.json`.

`configure_comfy` now enables parsing before model/attention imports, rejects
pre-imported CLI state and verifies requested CPU/GPU-only/SDPA settings.
Reports disclose requested and actual attention, parsed policy, Torch source
commit and tracked Comfy patch SHA. Eight CPU regression cases pass in the
validation interpreter: four policy cases and four real-video fixture cases
covering full frames, truncation, VFR timestamps and incompatible temporal
phase. The base environment passes policy tests and explicitly skips the four
Torch/safetensors-dependent fixture tests. Diagnostics use non-mutating
time-embedding/first-block hooks; hook nonmutation was verified against the
existing reference outputs. First-block details use the actual adaptive
window function, not its `(4,3,3)` partition-count parameters as extents.

Operator isolation identifies two real precision mismatches: the MLX single-row
BF16 `addmm` fallback rounds before bias (FP32 accumulate/bias then one cast
matches all three time projections exactly), and checkpoint RoPE frequencies
are FP16, with reference angles rounded to FP16 before FP32 trigonometry.
Correcting the latter matches isolated Q/K rotary tensors bitwise. The Torch
2.14 weighted RMS affine product is FP32; the older 2.12 MPS kernel/MLX 0.32
half kernel rounds before multiplying its weight. Time/RMS/RoPE corrections,
last-attention plain-Ada handling and output-shift grouping alone do not pass
the BF16 full-model gate. Explicit all-FP32 GEMM accumulation with BF16 output
boundaries does not resolve it either. Those failed diagnostics are preserved;
do not claim the old BF16 path qualified or a relaxed gate.

A separate candidate retains the checkpoint's original FP16 coefficients and
FP16 projections, while residuals, normalization, time embedding, FFN nonlinear
arithmetic and final velocity remain FP32. Its optional FP16 SDPA keeps every
key/query/window and is not spatial or temporal approximation. This changes
the low-precision recipe; it is not claimed bitwise equivalent to Comfy BF16.
Against the same original checkpoint's explicitly configured FP32 reference,
all 128 intermediate/time/velocity/Euler gates pass unchanged (cosine >=0.999,
relative-L2 <=0.05). FP16-attention candidate latent relative-L2 is 0.0045180,
cosine 0.999989795. Its 128 latent comparison to the BF16 reference is 0.109071
relative-L2 / 0.994034610 cosine and fails; that comparison is also retained.
The stronger original-parameter FP32 witness is not a repair of BF16 equivalence.

Complete fixed-input/noise restoration with independent FP16 VAE and FP32 DiT
reference passes every saved tensor at both real-image sizes and both complete
512/1024-output LTX-derived 25-frame/24-FPS videos. These use actual directly
downsampled Z-Image input and genuine video frames, not rolling-image video.
Input/probe/model coefficients remain unmodified. The video reference uses
the separately owned three-line non-CUDA cache-transfer guard; patch SHA is
recorded and the pinned original checkout is untouched.

| Candidate component workload | Latent relative-L2 | Decoded pixel relative-L2 | Pixel cosine |
| --- | ---: | ---: | ---: |
| Real image 256->512 | 0.00824473 | 0.00176499 | 0.999998457 |
| Real image 512->1024 | 0.0120328 | 0.00151095 | 0.999998867 |
| Real video 256->512, all 25 frames | 0.00326721 | 0.00101590 | 0.999999484 |
| Real video 512->1024, all 25 frames | 0.00941184 | 0.00123735 | 0.999999237 |

Records: `seedvr2-restore-fp16-fp32residual-512-all-parity-v1-20261014.json`,
`seedvr2-restore-fp16-fp32residual-1024-all-parity-v1-20261014.json`, and
`seedvr2-restore-fp16-fp32residual-video512-t25-all-parity-v1-20261014.json`, and
`seedvr2-restore-fp16-fp32residual-video1024-t25-all-parity-v1-20261014.json`.
Both video witnesses include every decoded frame, original dimensions and
matching original FP32 noise. The preparation tool now checks actual frame
timestamps as well as nominal FPS; it rejects truncation and VFR rather than
publishing a reduced fixture. These component times include dumps and exclude media
import/export, so they are not paired full-request speed proof. No public
default or App/CLI binary had been replaced at the end of that component-only
audit. The subsequent local App promotion is recorded below. Its 1024/t25
component observation is encode 20.1335 s / denoise-with-dumps 14.9533 s /
decode 50.0064 s, active MLX peak 21,275,343,558 bytes. This is not an App
measurement or a before/after physical-memory proof. Full-request
speed/memory, App regression and hybrid qualification are still required.

### Fixed stable profile and real App regression, October 14

The local build now defaults public SeedVR2 sessions to the quality-checked
original FP16-coefficient/projection and FP32-state profile. It uses FP32 time
embedding, normalization/affine, FFN nonlinear arithmetic and final velocity;
FP16 SDPA retains all keys, queries and original windows. The legacy BF16
diagnostics are not declared equivalent or qualified. No checkpoint content,
sampler schedule, source frame count or spatial context was changed.

Shared precision constants bind planning to actual results. Every generation
reports `numerical_recipe=seedvr2-stable-fp16-fp32-v1`, projection/residual/RMS/
time/attention/velocity and RoPE-angle precision. Public load rejects other
unqualified DiT recipes, clears a failed session and never exports their
output. An explicit `DIT_RESIDUAL_RECIPE=native` rejection exited 1 with the
expected stable-profile error and no `rejected.png`. Record:
`seedvr2-stable-profile-rejection-v1-20261014`. Offline probes still allow
precision diagnostics. The no-environment default's entire 128 trace is
bitwise identical to the previously qualified candidate:
`seedvr2-stable-default-equivalence-128-v1-20261014.json`.

Native/Swift App builds completed and source fingerprints match the compiled
runtime manifest. Dedicated contract (5), geometry (1), downloader (3) and
oracle policy/real-video fixture (8, validation interpreter) tests pass. This
is not a claim that unrelated pre-existing repository failures disappeared.

The real StudioState/NativeJobStore App chain passes all four generations,
original-source SHA checks, executed/plan precision agreement, strict image
transactions, complete MP4 frames/FPS/dimensions, retained job history and
final unload. The 1024 image and video first/middle/last frames were visually
inspected; neither clip is truncated or blank. Both MP4s independently decode
to 25 frames / 24 FPS / 1.041667 s. Record:
`seedvr2-app-stable-fp16-fp32-real-e2e-v1-20261014/generation-report.json`.

| App native request | Seconds | Active MLX peak (bytes) | Sampled physical footprint peak (bytes) |
| --- | ---: | ---: | ---: |
| Image 512, first load/JIT | 1.445831 | 9,622,623,114 | 13,066,573,944 |
| Image 1024, reused | 3.522688 | 16,491,014,650 | 19,904,846,064 |
| Video 512, full t25, reused | 20.668313 | 10,867,197,374 | 14,789,433,608 |
| Video 1024, full t25, reused | 84.686611 | 21,596,673,206 | 24,869,574,968 |

The 1024 video has 2905 memory samples and zero observer failures; sampled
Metal allocation peak is 25,017,483,264 bytes. Sampled peaks are lower bounds,
the footprint/Metal scopes overlap and must not be added, and these App runs
are not balanced speed or physical-memory reduction proof. The profile's
complete-request Comfy speed cohorts are being re-run separately; old BF16
speed ratios remain historical. Default-reference controls and the new
profile's GPU/CPU+NE quality/speed/App integration remain outstanding.

The fixed profile's first new complete ABBA speed gate passes against the
default-attention Comfy observer for real 256->512 images: native warm median
0.827034 s, reference 3.985151 s, ratio 4.81861x, six warm requests per route.
Every import/resize/encode/sample/decode/export executes, and only Comfy model
loader nodes are cached. Both owned servers exited; source/model/native build
and reference identities remain unchanged. Record:
`seedvr2-stable-image512-paired-default-v1-20261014`. The default observer's
explicit FP16-VAE setting and ordinary residency are bound in its launch
record; this is not an unlabeled GPU-only/SDPA reference or a universal speed
claim. The 512->1024 default-image ABBA cohort also completes and passes:
native median 3.506320 s versus 15.535424 s, ratio 4.43069x, again six full
warm requests per route with immutable identities and terminal owned servers.
Record: `seedvr2-stable-image1024-paired-default-v1-20261014`. New-profile
SDPA/GPU-only and complete video paired gates remain separate pending checks.

Both new-profile complete image SDPA/GPU-only ABBA cohorts subsequently pass,
six warm requests per route and workload. At 256->512, native median is
0.827852 s versus 3.977832 s (4.80500x); at 512->1024, 3.503769 s versus
15.636062 s (4.46264x). Required media/encode/sampler/decode/export nodes execute
every time; only Comfy model loaders are cached. Source, checkpoints, native
library/build and pinned stock Comfy identities remain unchanged; all owned
image reference servers exited. Records:
`seedvr2-stable-image512-paired-sdpa-v1-20261014` and
`seedvr2-stable-image1024-paired-sdpa-v1-20261014`. Default controls above are
also retained rather than assuming SDPA is always faster. These four image
cohorts satisfy image workload-specific speed gates, not video or ANE gates.

The stable 256->512 / all 25-frame / 24-FPS video SDPA/GPU-only cohort is now
running with the owned non-CUDA cache-transfer compatibility copy. No Core ML
export, GPU/NE probe or native rebuild overlaps the timed requests. The
experimental exporter has a separately tested `original-fp16` operand option,
preserving source bits and distinguishing old BF16-rounded artifacts; five
CPU cases pass. The isolated experiments after this cohort are recorded below.

The complete stable 256->512 / t25 / 24-FPS SDPA/GPU-only video ABBA cohort
now passes: native warm median 20.598051 s versus compatible Comfy 71.337080 s,
ratio 3.46329x, six complete warm requests per route. All MP4s retain every
frame, 512-square dimensions and 24 FPS. Required stages execute on every
request; only reference model loaders are cached. Both owned servers exited,
and source/checkpoint/library/build/patch identities stayed immutable. Record:
`seedvr2-stable-video512-t25-paired-sdpa-v1-20261014`. The three-line non-CUDA
cache-transfer compatibility guard remains disclosed; this is not an
unpatched stock MPS video claim. Stable 1024-video and default-video controls
still require complete paired observations. The stable 1024-video cohort
subsequently completes as recorded below.

After the timed cohort ended, an isolated detached stable-profile worktree
prepares the new hybrid without changing stable source or compiled identity.
It aligns GPU-suffix SwiGLU nonlinear precision with the stable full FFN,
binds original/legacy coefficient recipes to loaded DiT dtype and rejects
incompatible operands before Core ML construction. The original-FP16 512-row /
2048-channel bank's block-0 synthetic 4096-row FFN screen executes eight
CPU+NE predictions per call: GPU warm 31.869/31.882/31.889 ms, hybrid
27.541/27.431/27.520 ms, relative-L2 0.00311428 / cosine 0.999995. Record:
`seedvr2-original-fp16-hybrid-block0-4096-v1-20261014`. This is one-block
evidence, not whole-model/App speed or physical ANE occupancy. The Core ML
plan prefers Neural Engine for three convolutions, SiLU and multiply; that
branch excludes GPU. The full 32-bank export subsequently completed. Whole-
model quality, real-media speed/memory and App hybrid integration remain
pending. No public GPU+ANE capability has been enabled.

### Original-FP16 all-block hybrid image gates and scheduling audit

The original-FP16 static 512-row / 2048-channel prefix bank now contains all
32 blocks. Manifest SHA-256:
`109dde47f82b7acb3a6d8cbba16b9d42e9f15fddae539d64141a4ec0ac8a8f76`.
Coefficients are the checkpoint's original FP16 values, not the earlier
BF16-rounded bank. The isolated GPU suffix uses the stable profile's FP32
SiLU/product arithmetic before the FP16 down projection. Public native
source/library/build identities are unchanged by these experiments.

Complete shared-input/noise component restoration executes all 32 video FFN
prefixes while text FFNs remain GPU. The original gates (cosine >=0.999 and
relative-L2 <=0.05) pass for both real-image sizes against the stable GPU:

| Hybrid image output | Latent relative-L2 / cosine | Pixel relative-L2 / cosine | Core ML predictions per restoration |
| --- | ---: | ---: | ---: |
| 512-square | 0.02942364 / 0.99956703 | 0.00831129 / 0.99996546 | 64 |
| 1024-square | 0.02554959 / 0.99967360 | 0.00540864 / 0.99998537 | 256 |

The VAE condition is identical to the matching GPU control. The 512 witness
also passes against the original checkpoint's independent FP32 reference:
latent relative-L2 0.03177506 / cosine 0.99949505; pixels 0.00856942 /
0.99996329. A subsequent explicit 1024 independent FP32 comparison also
passes every saved tensor: latent relative-L2 0.02727620 / cosine 0.99962797;
pixels 0.00545219 / 0.99998514. This checks the saved tensors directly rather
than inferring reference parity from the GPU comparison. Records:
`seedvr2-original-fp16-hybrid-all32-512-parity-v1-20261014.json`,
`seedvr2-original-fp16-hybrid-all32-512-original-reference-v1-20261014.json`,
`seedvr2-original-fp16-hybrid-all32-1024-matched-parity-v1-20261014.json`, and
`seedvr2-original-fp16-hybrid-all32-1024-original-reference-v1-20261014.json`.

These quality passes are not full-request speed proof. In the separately
observed 1024 component screens, row-chunk hybrid warm denoise is about
2.06 s versus the matched GPU control's about 1.84 s: a regression. The
1024 hybrid makes 1024 Core ML predictions over four restorations, with zero
Core ML fallback output-copy bytes. Neither that fact nor the preferred
compute plan establishes physical ANE occupancy or an end-to-end speed win.

A whole-GPU-suffix schedule exposed a mutable-output lifetime bug. A logical
`mx::copy` did not freeze Core ML's shared backing: synthetic block-0
FFN relative-L2 0.461709 / cosine 0.888908 fails. Its apparent roughly 23.6-ms
timing is invalid performance evidence. A subsequent physical Metal snapshot
and auxiliary-stream fence restores the block-0 gate (relative-L2 0.00311428,
cosine 0.999995), but warm median is about 33.60 ms versus GPU 31.88 ms.
It additionally copies 83,886,080 snapshot bytes over four calls. Records:
`seedvr2-original-fp16-hybrid-wholesuffix-block0-4096-v1-20261014` (failed) and
`seedvr2-original-fp16-hybrid-wholesuffix-block0-4096-v2-20261014` (correct,
slower). Keep row chunks as the experimental default and preserve both runs.

Next: investigate independently owned output backings and fewer scheduling
fences, measure profitable shape/block partitions, and gate real 512/1024
videos as well as images. No public GPU+ANE capability or App route has been
enabled on the strength of these component-only observations.

### Stable 1024-video speed gate and independent-buffer ABI diagnosis

The stable-profile complete 512->1024 / all 25-frame / 24-FPS SDPA/GPU-only
ABBA cohort passes: native warm median **84.878555 s**, MPS-compatible Comfy
**298.901318 s**, ratio **3.52152x**, six genuinely executed warm requests per
route across two processes each. Both owned reference servers exit; all
outputs retain every frame, 1024-square geometry, 24 FPS and 1.041667 s.
Every processing stage runs on each request. Cached reference nodes, where
present, are model loaders only; the first warm request in the second
reference cohort re-executes both loaders rather than reusing them. This is
not an encoded-latent cache shortcut. Original input/checkpoint/library/build
and compatible-reference identities remain unchanged. Library SHA remains
`de18ca6024dc87ae802660acbce81f5ef0a0fad3d3736d3c41e9635e4c978be7`.
Record: `seedvr2-stable-video1024-t25-paired-sdpa-v2-20261014`.
An earlier v1 invocation stops on a mistyped library path before hardware
execution; its empty output directory is preserved and is not speed evidence.

The stable-profile SDPA/GPU-only image and complete-video speed gates now
pass at both output sizes. Default-video controls, complete hybrid execution/
quality/speed/memory and App promotion remain separate requirements. Do not
replace those checks with these pure-GPU speed results.

After that cohort ends, the deferred independent-output-backing laboratory
build succeeds but its first 512-row probe rejects the feature ABI before
predicting. Record: `seedvr2-original-fp16-buffered-schedule-screen-v1-20261014`.
The failed run yields no qualified speed or numerical result. A CPU-only,
no-prediction description inspection shows that both features have exactly
one enumerated shape `[1,2560,1,512]`, with singleton ranges in every dimension.
The initial check incorrectly expected Core ML's `Unspecified` enum. Apple's
SDK defines that enum as unconstrained, not fixed; it is unsuitable for a
bounded caller-owned output backing.

The laboratory now accepts only a single exact enumerated shape or exact
singleton ranges, and rejects unspecified, multiple/different enumerated
shapes and non-singleton ranges. CPU constraint tests and strict compilation
pass. The failure record is retained. This is a safety-preserving metadata
interpretation correction, not relaxation to flexible output backings. Fresh
hardware probes, full image/video quality and whole-request memory/speed
remain required before any public GPU+ANE enablement.

A fresh 4096-row block-0 screen with the corrected independent backing pool
passes all four global and per-row-chunk gates (relative-L2 <=0.01, cosine
>=0.99999). Inputs change between every repeat (seeds 42...45), so stale
previous-call outputs cannot masquerade as correct cached results. Warm
medians are GPU **31.9203 ms** and GPU/CPU+NE **23.6038 ms**. Eight independent
output slots occupy 20,971,520 MLX bytes; snapshot-copy and Core ML fallback
output-copy counters are both zero. Record:
`seedvr2-original-fp16-buffered-ffn-4096-v2-20261014`.
Unlike the failed logical-copy schedule, these timings have passing numerical
witnesses. They remain a one-block screen, not balanced full-request/App
speed proof or physical ANE occupancy. The fresh v2 orchestration completes
512/1088/4096-row micro gates and a matched 512-image GPU control, then stops
because its caller uses exporter-style `0-31` rather than the restoration
probe's explicit CSV block-list ABI. No hybrid image request executes in that
failed attempt. The argument adapter is corrected with a CPU regression case;
eight deferred-screen CPU checks pass. The failed v2 record is retained, and
the complete four-workload screen runs in a fresh v3 directory, with hardware
jobs still serialized.

### Independent-backing all-block full-media component qualification

The corrected v3 screen completes all three micro geometries and all four
real-media restorations. Every saved condition, velocity, latent and pixel
tensor passes the unchanged cosine >=0.999 / relative-L2 <=0.05 gates against
both the matching GPU and the original checkpoint's independent FP32
reference. The source/noise fixtures, reference tensors, original weights,
stable library and compiled bank remain unchanged. Record:
`seedvr2-original-fp16-buffered-schedule-screen-v3-20261014`.

| CPU+NE-prefix component witness vs original FP32 | Latent relative-L2 / cosine | Pixel relative-L2 / cosine |
| --- | ---: | ---: |
| Image 512 | 0.03177506 / 0.99949505 | 0.00856942 / 0.99996329 |
| Image 1024 | 0.02727620 / 0.99962797 | 0.00545219 / 0.99998514 |
| Real video 512, all 25 frames | 0.01328465 / 0.99991176 | 0.00433528 / 0.99999062 |
| Real video 1024, all 25 frames | 0.01513084 / 0.99988552 | 0.00367425 / 0.99999327 |

Both image screens have one cold and three warm component restorations per
route. Warm denoise medians are GPU/prefix 0.492346/0.434514 s at 512 and
1.838220/1.727250 s at 1024. These separate-route component observations are
not balanced complete-request/App speed gates.

The complete 512/t25 prefix restoration executes 448 Core ML predictions;
1024/t25 executes 1792. Each uses every one of the 32 video FFN prefixes;
text FFNs remain GPU. Snapshot-copy and Core ML fallback output-copy bytes
are zero. Independent slots retain 36,700,160 and 146,800,640 MLX bytes,
respectively; this is a deliberate ownership/scheduling buffer cost, not a
claim of zero additional memory. Active peaks in these single observations
are 10,883,055,558 / 21,422,175,806 bytes, versus matching GPU controls'
10,785,535,350 / 21,275,343,574 bytes. Core ML allocations are outside that
MLX-active scope; no physical-memory reduction is demonstrated by this pool.

Single component restoration observations are GPU/prefix 20.7536/20.1695 s
at 512 and 84.6629/83.0873 s at 1024. Their denoise portions are
3.22196/2.89352 and 14.2925/12.7929 s. They exclude media import/export and
model loading and are not repeated video warm medians, balanced speed proof,
or physical ANE occupancy evidence. Complete-request speed/memory, runtime
execution evidence and real App regression still precede public promotion.

The laboratory also adds an explicit CPU-only prefix control (not a hidden
environment override), and an optional offline DiT staging experiment.
Staging releases DiT parameters during both VAE phases and reloads the same
original checkpoint before sampling, without changing codec context/math.
Source size/mtime checks guard each reload. Clear/synchronize/reload cost is
included in `staging_seconds` and the reported total `restore_seconds`;
never count only the three compute timers as the staged pipeline's cost.
This is an experiment, not an enabled public residency policy. Raw GPU-only
image/video memory screens are running separately after the hybrid screen
ends; numerical equality, whole-process peaks and App integration remain
to be verified before any staged-memory claim.

### DiT staging component screens: exact output and active-peak tradeoff

All three initial GPU-only staged screens complete. Each saved condition,
velocity, latent and decoded-pixel tensor is bitwise identical to its matching
resident GPU witness; even the entire restoration file SHA-256 matches.
The two real-video witnesses include every one of the 25 frames. Original
DiT/VAE hashes are rechecked afterward. Record:
`seedvr2-staged-dit-component-exact-v1-20261014.json`.

| Workload screen | Resident active peak (bytes) | Staged active peak (bytes) | Staged weight-management seconds |
| --- | ---: | ---: | ---: |
| Image 1024 | about 16.49 GB | 9,686,993,754 | about 0.506 warm |
| Video 512, t25 | 10,785,535,350 | 9,112,797,702 | 0.541682 |
| Video 1024, t25 | 21,275,343,574 | 14,486,178,654 | 0.533852 |

These are MLX-active allocation peaks only, not physical-process/system or
enforced-budget measurements. In the 1024/t25 single component observations,
total restoration including all staged management is 84.9039 s versus the
resident's 84.6629 s; this is not a paired speed claim or proof of no tradeoff.
The image's three warm staged totals are 3.79461/3.80532/3.79087 s, with about
half a second of explicit management each. At request end only the VAE's
501,296,454 parameter bytes remain resident, rather than 7,285,499,974 bytes
for VAE plus DiT. This does not mean all other allocations are zero.

`seedvr2_staged_cohort.py` in the isolated lab now serializes resident/staged/
staged/resident full component restorations, cold plus three warm repeats per
process. It binds source/noise/probe/bank/model identities, requires identical
restoration files, checks actual residency/Core ML exclusion, and refuses
timings that omit reload/synchronization cost. Five CPU guard tests pass.
Balanced 1024 image and complete 1024/t25 video cohorts are running in that
order. Public `component_staged` residency is still rejected: plan/SDK/App
integration, physical-memory sampling, broader frame/geometry coverage and
App regression are required before promoting this offline-only policy.

### Balanced staged-DiT image/video component cohorts

Both four-process resident/staged/staged/resident cohorts complete with six
warm samples per route. Source/noise/probe/bank/original-model identities
remain unchanged, every request restores its input anew, and all four saved
final restoration files have identical SHA-256s within each workload. Each
intermediate repeat is finite; final-file equality is not presented as a
separately saved tensor comparison for every repeat.

| Component ABBA workload | Resident / staged warm median | Resident / staged active peak (bytes) |
| --- | ---: | ---: |
| Image 512->1024 | 3.397610 / 3.786635 s | 16,472,933,578 / 9,686,993,754 |
| Real video 512->1024, all t25 | 84.499350 / 84.971450 s | 21,275,343,574 / 14,486,178,654 |

The image memory reduction costs about 11.45% component time; the full-video
component reduction costs about 0.56%. All clear/synchronize/reload work is
included. The active-peak gates pass; those scopes exclude allocator cache
and non-MLX allocations, and are not whole-request/App physical-peak or hard
budget qualification. The faster resident route remains the default.
Records: `seedvr2-staged-dit-image1024-abba-v1-20261014` and
`seedvr2-staged-dit-video1024-t25-abba-v1-20261014`.

The optional public `component_staged` candidate source is now prepared for
planning/CLI/Swift/App, with explicit per-request policy selection, source
stamp/stable-profile reload checks, cost/resident-component telemetry, and
safe resident/staged/resident transitions. Swift schema-2 conversion retains
the SeedVR2 policy while continuing to omit private residency from unrelated
public streaming intents. The conservative memory estimate is deliberately
unchanged until broader geometry/physical-memory qualification; no lower hard
budget is promised. Dedicated tests cover both request schemas and an actual
App policy-switch PNG witness. Strict C++ syntax, Swift parsing and Swift SDK
typechecking pass; an early bare SDK invocation lacked target/include flags
and is not counted as a pass. An isolated native candidate build is running
after the benchmark finishes. No stable App/library has been overwritten,
and real App/default/staged physical sampling still precedes code commit.

### Tested optional staged residency in isolated native/Swift/App build

The isolated candidate library and graphical App build complete. Dedicated
SeedVR2 public contract (6), downloader (3), geometry (1) tests pass. SDK
streaming transaction regressions pass, including explicit SeedVR2 schema-2
residency preservation and exclusion of private residency from public
streaming intents. An extra test initially links with a lower deployment
target than the pinned MLX/native library and emits a linker warning; it runs
successfully on this host, not proof of support for older macOS versions.
Unrelated pre-existing repository failures are not relabeled as fixed.

Candidate library SHA:
`45caa42aba570667c45ba9b6f8eb4c1bcaaf20386e2746dd5929651a1c9c71d6`.
The earlier stable library remains intact at SHA
`de18ca6024dc87ae802660acbce81f5ef0a0fad3d3736d3c41e9635e4c978be7`.
Its established speed cohorts remain attributable to that binary; do not
silently relabel them as candidate-binary paired speed records.

Both actual StudioState/NativeJobStore chains terminate successfully: each
executes real 512/1024 images and full 25-frame/24-FPS videos, then an explicit
resident/staged/resident image sequence within the same workflow. All seven
requests per policy pass precision/plan/residency/component/timing checks,
strict PNG transactions, retained history and final unload. All policy-switch
PNGs match exactly. Original-model SHA checks pass for every request. Records:
`seedvr2-app-staging-candidate-resident-e2e-v1-20261014/generation-report.json`
and `seedvr2-app-staging-candidate-staged-e2e-v1-20261014/generation-report.json`.

| Actual App request observation | Resident / staged seconds | Resident / staged active MLX peak (bytes) | Resident / staged sampled footprint (bytes) |
| --- | ---: | ---: | ---: |
| Image 512, first load/JIT | 1.841292 / 1.925124 | 9,622,623,114 / 7,784,157,906 | 13,066,426,344 / 11,011,201,216 |
| Image 1024 | 3.639583 / 3.880221 | 16,491,014,650 / 9,703,770,954 | 19,904,714,800 / 13,307,020,704 |
| Video 512, all t25 | 20.798663 / 21.098368 | 10,867,192,086 / 9,191,435,598 | 14,796,822,648 / 13,091,543,208 |
| Video 1024, all t25 | 84.798178 / 85.661877 | 21,596,667,278 / 14,800,751,438 | 24,732,817,576 / 18,470,209,024 |

The staged 1024 video has 2959 memory observations, no observer failures,
sampled Metal peak 18,221,400,064 bytes. The scopes overlap and must not be
added; sampled peaks remain lower bounds and these App observations are not
balanced physical-memory or universal speed guarantees. Actual staged output
retains only 501,296,454 VAE parameter bytes; its DiT reuse flag is false.

Independent host verification confirms PNG file hashes match across policies,
and every decoded frame of both MP4s matches bitwise while retaining all 25
frames, correct dimensions and 24 FPS. MP4 container file hashes differ; do
not call their complete files identical. Record:
`seedvr2-staged-app-media-verification-v1-20261014.json`.

The graphical candidate opens a separate on-screen window using the test
state directory, without overwriting the current App's state. System Events
accessibility and window-image capture are denied by macOS permissions;
therefore no manual control-click or screenshot/visual UI pass is claimed.
The tested window is closed after inspection, and the existing App is left
alone. Actual App generation and typed draft/selector integration are tested
above; the visual-click limitation is retained, not bypassed.

### New staged-capable library: complete default-reference 512-video gate

The staged-capable candidate's default resident pure-GPU 256->512 / all t25 /
24-FPS full-request ABBA cohort completes and passes. Six warm requests per
route across two independent processes give native median **20.585554 s**
versus compatible Comfy default-attention/ordinary-residency **71.189599 s**,
ratio **3.45823x**. All processing stages execute each request; only model
loader nodes are cached. Every MP4 independently retains all 25 frames,
512-square dimensions and 24 FPS. Original source/checkpoint/candidate-library/
runtime-build/reference identities remain immutable and both owned reference
servers exit. Record:
`seedvr2-staged-candidate-video512-t25-paired-default-v1-20261014`.
The three-line MPS cache-transfer compatibility guard and explicit FP16 VAE
configuration remain disclosed; it is not an unpatched stock MPS video claim.
This evidence binds the new library SHA `45caa42...`, not the older resident
binary's ratios. New-library 1024 default-reference and hybrid full-request/
App qualification remain separate checks.

After the timed job, the main-tree hybrid backend is being aligned with the
quality-checked original-FP16 independent-backing implementation. It remains
outside the public runtime/source list and does not turn on `supports_gpu_ane`.
Its typed schedule has no environment fallback to the failing old copy paths.
A 64-slot / at-most-256-MiB pool is reused only after a completed, separately
owned join. Row groups partition long inputs exactly instead of expanding the
pool beyond its bound; the validated 1024/t25 geometry still uses one group.
CPU coverage includes tails, larger videos and the signed-int limit, and
refuses incomplete/double-written leases or flexible output shapes. Failed
allocation/prediction latches the partition unavailable; fallback output
handling rejects aliases into other live slots. These source/CPU checks do
not replace fresh numerical/performance checks or grant public qualification.

The main-tree single-block screens now pass global and every row-chunk gate
for all four varying seeds. At 4096 rows, warm GPU median is 31.9137 ms,
GPU/CPU+NE 23.6330 ms; an explicit GPU/CPU-only prefix control is 27.1534 ms
and independently passes its gates. The CPU-only control itself improves on
GPU-only in this screen, so do not attribute the whole hybrid gain to ANE.
The CPU+NE configuration is additionally faster than that CPU-only control,
but these are separate-route micro screens, not balanced device-occupancy
proof. Records: `seedvr2-main-buffered-ffn4096-v1-20261014` and
`seedvr2-main-buffered-cpu-only-ffn4096-v1-20261014`.

The larger 126,976-row screen (the patched-latent row count of 1024/t121)
uses 248 fixed-row predictions per call in four bounded GPU/prefix groups.
All four seeds and all 248 row-chunk gates pass (global relative-L2 about
0.003098, cosine 0.999995), with no snapshot or Core ML fallback output-copy
bytes. Only 64 backing slots / 167,772,160 MLX bytes are retained, rather than
one slot per input chunk. Warm GPU/prefix medians are 1.01416/0.789548 s.
Record: `seedvr2-main-buffered-ffn126976-v1-20261014`.
This is synthetic one-block numerical/performance evidence, not a restored
121-frame real video or a whole-model/App acceptance claim. Complete-media
main-tree regressions and public interface/qualification are next.

### Qualified-bank manual hybrid candidate and actual App regressions

The main-tree four-media GPU/FP32 tensor screen completes and passes at both
image sizes and both complete t25-video sizes:
`seedvr2-main-buffered-full-media-screen-v1-20261014`. A separate candidate
then integrates the typed bounded prefix into the native request/session,
CLI, Swift/App plan and executed precision. It accepts only the exact
original-FP16 all-32 manifest digest, requires approximation consent, verifies
the full compiled tree before model construction, and checks file/directory
stamps before/after generation. Planning rejects orphan/incomplete banks,
unqualified manifests and GPU/auto requests carrying a hidden prefix bank.
Auto remains GPU-only. No CPU-only or failed legacy scheduling environment
override is present in this candidate.

Candidate library SHA:
`2fdc400244fa4163465c2bdb38ad04ee8d495d4fd30062e0d4c501db58e2a4d7`.
Seven public contract and five exporter tests pass. Actual App GPU, hybrid
resident and hybrid staged runs all terminate successfully, each with seven
real requests (four media/geometry cases plus resident/staged/resident PNG
transitions), retained history, strict PNG publication and final unload.
Mixed precision is explicitly separate from the GPU recipe: FP16 Core ML
prefix activation/partial down output plus FP32 joins, with unchanged
original coefficients and FP32 residual state. Runtime asserts all 32 video
FFN prefixes executed, and reports actual 64/256 image and 448/1792 video
predictions at 512/1024. Text FFN, attention and VAE stay GPU. Core ML CPU+NE
configuration is not claimed as physical ANE occupancy.

App records: `seedvr2-app-hybrid-candidate-{gpu,resident,staged}-e2e-v1-20261014`.
Single reused-model 1024/t25 request observations are GPU 84.715817 s,
hybrid resident 83.106090 s and hybrid staged 83.532414 s. Their active peaks
are 21,596,667,598 / 21,743,501,998 / 14,947,552,078 bytes, sampled footprints
24,933,898,552 / 25,055,156,728 / 18,789,992,296 bytes. These are not paired
speed proofs or enforced physical-memory budgets. Staged hybrid retains its
Core ML prefix bank; reported resident parameter bytes explicitly exclude
Core ML, and `resident_components` includes that bank rather than falsely
claiming VAE-only residency. Each mixed-policy PNG transition is identical;
GPU-vs-hybrid outputs are not declared bitwise equivalent.

A new complete-request native/hybrid ABBA observer forces import/resize/VAE
encode/sample/decode/export on every request, validates executed recipe,
residency and full prefix counts, and binds original model/source/library/
build/bank identities. Four CPU guard tests reject cache shortcuts, wrong
route/precision/counters/bank and unexpected Core ML use on GPU. Both image
cohorts complete with six warm samples per route: at 512, GPU/hybrid medians
0.828580/0.772848 s (1.07211x); at 1024, 3.500113/3.385148 s (1.03396x).
Records: `seedvr2-public-hybrid-image{512,1024}-request-abba-v1-20261014`.
These are matched whole-request wins for the stated bank/hardware/workloads,
not extrapolated single-block gains. Video warm cohorts, physical execution
evidence, broader exported-media quality and release/commit audit remain.

The complete-request 256->512 / full t25 / 24-FPS GPU/hybrid ABBA cohort also
passes with six warm samples per route: GPU median **20.576623 s**, hybrid
**20.282359 s**, ratio **1.01451x**. All actual processing/export stages run,
the bank/precision/all-block counters are checked, and library/build/model/
input/bank identities remain unchanged. Record:
`seedvr2-public-hybrid-video512-t25-request-abba-v1-20261014`.
This modest full-request win is not replaced by the larger FFN-only ratios.
The subsequent 1024/t25 complete-request cohort and release audit are
recorded below. Hardware placement remains a separate acceptance boundary.

### Completed 1024 hybrid request cohort and corrected-metadata release audit

The fourth public hybrid workload completes and passes. Six warm requests
per route give GPU **84.951658 s**, GPU/CPU+NE **83.104532 s**, ratio
**1.02222654x** for real 512->1024 / all t25 / 24-FPS video. The full import,
encode, one-step sample, decode and export execute on every repeat; all bank,
source, build, executed precision and 32-block counters remain bound to the
candidate library SHA `2fdc400...`. Both reference-route processes terminate.
Record: `seedvr2-public-hybrid-video1024-t25-request-abba-v1-20261014`.
All four stated GPU/hybrid complete-request cohorts now pass. These are
CPU+NE-configuration results, not physical NE occupancy proof or evidence of
universal acceleration.

The generic plan initially labeled the all-block SeedVR2 route as a
single-block MLP approximation. A metadata-only correction reports
`seedvr2_original_fp16_video_ffn_prefix_fp16_partial_down_fp32_join`, without
changing inference math/weights. The separate v2 native library SHA is
`e47afeb57050e1cf040f8bbcab6389e0b1ea2bc795d5d704f53ddcd1e7c57ab0`.
The measured v1 binary and build identity remain intact; its ABBA timings
are not relabeled as v2-binary measurements. An initial Swift build found
absolute paths inside the copied v1 module cache. That task-owned cache was
retained under `module-cache-copied-v1`, and a fresh-cache App build passes.
No user cache/App was removed or replaced.

Fresh v2 native contracts (7), exporter tests (5), observer tests (4),
downloader tests (3), geometry test (1), bounded coverage/lease/static-shape
C++ test, and exported-media statistics/decoder guards (4) pass. Three
negative compiled-bank tests reject absent files, same-size corrupted files
and symlink entries before constructing models; they write only to fresh
scratch directories and leave the real bank unchanged. These are dedicated
tests, not a claim that unrelated repository suites are entirely green.

All three actual v2 StudioState/NativeJobStore chains also terminate
successfully: GPU, hybrid resident, hybrid staged, seven real requests each.
They retain all media frames/FPS, correct recipes/residency/counters, strict
PNG publication, resident/staged/resident transitions, history and unload.
Every one of the 14 hybrid requests carries the corrected all-block plan
label and lacks the erroneous single-block label. Records:
`seedvr2-app-hybrid-candidate-{gpu,resident,staged}-e2e-v2-20261014`.
The v2 GPU, hybrid resident and hybrid staged outputs each match their v1
counterparts in PNG file bytes and every decoded video frame; the v2
hybrid resident/staged pair is also exact. MP4 container identities are
recorded separately. All four required-equality comparisons pass:
`seedvr2-public-hybrid-app-v2-equivalence-20261014.json`.

The bounded host export observer checks every PNG RGBA value and all 25
decoded RGB video frames, without retaining an entire float64 video. The
v1 hybrid resident/staged pair matches exactly at both sizes; GPU/hybrid
outputs deliberately do not. Diagnostic GPU/hybrid exported PSNR is
52.4212/53.8134 dB for image RGBA and 33.4344/38.6809 dB for video RGB at
512/1024. Video includes codec effects; these statistics do not replace or
weaken the shared-noise original-FP32 component gates. The 512-video middle
frame pair and 1024 hybrid image were visually inspected; this is selected
output inspection, not every-frame visual QA or a manual-click UI test.
Record: `seedvr2-public-hybrid-app-export-screen-v1-20261014.json`.

Local read-only `sudo -n powermetrics` sampling cannot run because an
administrator password is required. No credential prompt/bypass or system
configuration change was attempted. Metadata stays
`physical_ane_occupancy_measured=false`; device configuration/planned
assignment cannot stand in for runtime occupancy. The independent CPU-only
micro control also improves on GPU, so all gain is not attributed to NE.
The newer resident library's 1024 default-attention Comfy video control and
broader hardware/long-video qualification remain pending. Default GPU and
optional staged residency remain available without the prefix bank.

### Input-buffer residency experiment and speed-first image revision

After `ed502cb`, an isolated native/Swift/App candidate consumes normalized
source pixels instead of holding a resized NHWC RGB/RGBA source and its RGB
view through restoration. Video drops the redundant RGB source; the first
candidate also copies image alpha into independently owned FP32 storage,
without arithmetic changes. The model/VAE/resize/sampler recipes remain
unchanged. Candidate-v1 library SHA:
`6cdb3bb1f5cb5e2b15dfe3c4dedf5b976a62960fe0c0a26c2427f7d5b67c7926`.

Standalone Metal tests pass bitwise RGB/alpha normalization, disjoint buffer
ownership, source release and shape guards for images and t1/t5/t25 inputs.
The reusable build script matches the linked MLX deployment target. An early
ad-hoc build used 15.0 against MLX's 26.2 minimum and produced a linker warning;
it is not proof of older-system support. Oracle-policy/fixture tests (8) and
VAE-observer tests (7) pass using the full validation interpreter; the smaller
venv lacks optional readers and skips those cases rather than passing them.
Native contracts (7), native/Swift/App builds pass independently.

All three candidate-v1 App chains pass seven requests each, including real
512/1024 image/video generations, residency switches, history and unload.
All four exported-media comparisons require equality and pass: GPU vs its
previous build, hybrid resident vs previous build, hybrid staged vs previous
build, and candidate hybrid resident/staged. PNG bytes and all decoded video
frames match; MP4 container identities are reported separately. Records:
`seedvr2-app-media-lifetime-{gpu,resident,staged}-e2e-v1-20261014` and
`seedvr2-media-lifetime-app-exact-screen-v1-20261014.json`.

The v1 hybrid 1024/t25 active peaks drop from 21,743,499,350 to
21,428,931,206 bytes (resident), and 14,947,552,078 to 14,632,979,286 (staged).
These are App observations, not balanced physical-memory guarantees. Sampled
footprint does not consistently fall: staged observes 18,104,108,808 ->
18,474,583,936 bytes. Do not call source release a demonstrated whole-process
peak improvement. GPU's observed active reduction is larger than the logical
released buffer due to differing allocation/workspace peaks; do not attribute
every observed byte to the source buffer alone.

A new immutable-library complete-request revision ABBA observer tests before/
after/after/before, with six genuine warm requests per revision. The 1024 GPU
image cohort **fails** the non-regression speed gate: 3.503239 -> 3.521613 s,
about 0.52% slower, while active peak falls only 12,582,908 bytes. This result
is retained and is not relabeled as a speed improvement. Record:
`seedvr2-media-lifetime-image1024-revision-abba-v1-20261014`.

Accordingly, v1 is not promoted as the default image path. The revised source
keeps the original RGBA-backed alpha view for resident/speed-first images;
only explicitly staged images copy alpha, while both video policies release
redundant RGB. Telemetry reports the chosen retention recipe, logical alpha
bytes and retained source-RGBA bytes separately. The new standalone Metal
test additionally verifies the fast view policy. Fresh v2 native/App checks
and image non-regression measurements precede promotion; conservative memory
budget estimates remain unchanged.

### Revised input-residency candidate: actual App equivalence

The separately built v2 native library SHA is
`24441cacb407992573809bba5274b713c2de7d1f9440486a30c9188b72ddd553`.
Native contracts pass, including rejection of unrelated runtime FFN/QKV
mode overrides; native and Swift/App builds pass. The new Metal test verifies
both independently owned alpha and the original source-backed view, with
bitwise equal normalization/alpha and explicit source-release behavior.

All three actual v2 App chains complete successfully (seven requests each).
Every initial request reports the correct media-input retention policy and
logical alpha/source-RGBA bytes, plus the all-block hybrid approximation label
where applicable. Resident/staged/resident PNG transitions, complete video
frames/FPS, history persistence and unload pass. Records:
`seedvr2-app-media-lifetime-{gpu,resident,staged}-e2e-v2-20261014`.
All four required-equality exported-media pairs pass against the preceding
hybrid-v2 release and between the new hybrid policies; PNG bytes and every
decoded frame at both sizes are exact. Record:
`seedvr2-media-lifetime-app-exact-screen-v2-20261014.json`.

| Revised App 1024/t25 observation | Request seconds | Active MLX peak (bytes) | Sampled process footprint (bytes) |
| --- | ---: | ---: | ---: |
| GPU resident | 84.772071 | 21,282,100,094 | 24,953,821,472 |
| GPU/CPU+NE resident | 83.248088 | 21,428,934,494 | 24,704,080,616 |
| GPU/CPU+NE staged | 83.494277 | 14,632,979,286 | 18,475,140,944 |

These are single actual App observations, not matched speed or physical-peak
guarantees. Compared to the preceding hybrid-v2 App observations, 1024/t25
hybrid active memory is lower by about 314.6 MB in both policies; residency
staging additionally releases about 6.8 GB of GPU DiT parameters. Resident
image active peaks remain unchanged apart from one scalar allocation;
staged images retain the optional independently owned-alpha reduction.
Logical alpha/source-RGBA sizes overlap for view-backed images and exclude
resize graph/allocator/cache effects. They must not be added as a physical
peak estimate. Original quality recipes/gates, default GPU selection and
conservative memory planning remain unchanged.

The revised v2 resident image ABBA still fails the exact non-regression gate:
**3.512202 -> 3.525093 s**, about 0.37% slower, with one extra scalar and no
image active-memory benefit. Record:
`seedvr2-media-lifetime-image1024-revision-abba-v2-20261014`.
The valid output/App checks above are not substituted for a speed pass.
The next isolated v3 restores the original source/RGB owners through
restoration and creates resident image alpha only at export, instead of
adding an alpha consumer before normalization evaluation. A graph-fusion/
lifetime effect is a hypothesis, not an instrumented diagnosis. Video and
explicitly staged images still use the tested consuming helper. A fresh
v3 cohort and actual App checks must pass before code promotion.

The v3 image ABBA records **3.500405 -> 3.500845 s**, a 0.440-ms / 0.0126%
median difference and identical active peaks. Its exact `speed_gate_passed`
Boolean remains **false**; it is not relabeled as a passing speed gain.
The six before/after warm values have sample standard deviations of about
21.64/18.43 ms, much larger than this median difference. That context does
not prove equivalence or permit deleting any measurements. Record:
`seedvr2-media-lifetime-image1024-revision-abba-v3-20261014`.
This revision restores the original resident-image implementation/ownership
and is assessed as a video/staged-image memory feature, not an image-speed
promotion. No timing threshold is loosened. Any subsequent local code commit
must retain these latency limitations and require the actual App/output
correctness gates separately, rather than claim that the strict speed test
passed.

### Final input-residency correctness/commit audit (v3)

The final isolated native library SHA is
`596c160aabcc7ec248c6e842e86da1b2b27a0338bab41749014dff866a35d7e4`.
Native contracts and native/Swift/App builds pass. The new revision-runner
CPU guards (3) reject short cohorts, mismatched image frames and existing
output directories without reading models or overwriting artifacts.
The engine-owned cancellation atomic was audited: cached hybrid objects
reference stable engine storage, not a request-local stack variable.

All three v3 App workflows terminate successfully, seven requests each,
including all four complete media/size cases, three residency transitions,
precision/retention/counter checks, history and unload. All four required
export equality pairs pass against the preceding hybrid-v2 build and between
v3 hybrid policies. PNG files and all decoded video frames are exact; MP4
container identity remains separate. Records:
`seedvr2-app-media-lifetime-{gpu,resident,staged}-e2e-v3-20261014` and
`seedvr2-media-lifetime-app-exact-screen-v3-20261014.json`.

The v3 resident image active peaks now exactly match the preceding release
(9,622,623,434 / 16,491,014,970 bytes for GPU at 512/1024). The 1024/t25 App
observations are GPU **84.882445 s / 21,282,100,094 active bytes**, hybrid
resident **83.290870 s / 21,428,931,206**, and hybrid staged
**83.721415 s / 14,632,979,286**. The latter samples process footprint
17,788,192,712 bytes; this is a lower bound, not an enforced/paired physical
peak. Other preceding samples vary substantially and are retained.

The code is assessed for actual correctness and reduced video/staged-image
active-buffer residency, not promoted as an image speed win. All three
strict image revision speed Booleans remain false, including the final
0.440-ms difference, without changed thresholds or deleted failed history.
Existing all-four-workload GPU/CPU+NE speed evidence remains bound to its
earlier binary; the new library still requires its own reference speed
qualification. Stable/current user App artifacts remain unmodified.

### Follow-up audit during the immutable latest-library default-video cohort

The complete latest-library 1024/t25 default-reference ABBA runs under the
same live process handle throughout; it is not restarted on observation
timeouts. Its first native cohort completes at 85.076744 s warm median.
The first compatible Comfy default-attention/ordinary-residency cohort
completes at 295.439931 s warm median, with only UNET/VAE loaders cached.
The second owned Comfy cohort is still running. These partial results are
not a finalized speed gate. Record:
`seedvr2-media-lifetime-video1024-t25-paired-default-v1-20261014`.
All observed outputs retain 25 frames, 1024-square dimensions and 24 FPS.

The local SDK's `MLComputePlan.h` explicitly calls device usage anticipated,
not measured occupancy. The current SDK has CPU-only, CPU+GPU, all and CPU+NE
compute-unit choices, not a public NE-only choice. `/usr/bin/xctrace` exists
as a wrapper, but `xcrun xctrace list templates` fails because the installed
Command Line Tools do not provide the actual utility; Xcode.app is absent.
A narrowly filtered read-only IOKit query can read the H11ANE driver's power
state (0 in the observed GPU/reference-only snapshot). Driver power state or
registry service “active” status does not prove model execution/occupancy.
No permission bypass or system configuration change is attempted. A separate
controlled hardware experiment, if useful, must follow the terminal timed
cohort; public `physical_ane_occupancy_measured` remains false.

Existing 1024/t9 codec component observations show about 7.70 GB encoder and
14.20 GB decoder active peaks, motivating a further ownership experiment.
An explicit offline-only `Pipeline::restore_owned` and `restore-owned` probe
mode are prepared: materialized FP16 conditioning must have matching shape
and independent storage before normalized/padded FP32 source handles drop.
Frame geometry is captured before dropping ownership; model math, alpha,
noise, condition and non-owning/default request paths are unchanged.
Logical released bytes are not asserted as physical/peak reduction.

CPU primitive tests cover t1/t2/t5/t6/t25/t26, correct 4n+1 padding, unchanged
conditioning and rejection of unready/wrong-shaped/wrong-dtype results.
The sandboxed test initially fails while initializing MLX's Metal allocator,
even with CPU operations selected. The scoped host rerun passes without
loading models or running GPU inference. Strict C++ syntax checks pass.
This candidate is not enabled in the public session, built into the timed
library, committed, or credited with a measured memory/speed improvement;
fresh GPU numerical/App verification follows the completed benchmark.

### Finalized latest-library 1024-video default-reference speed gate

The same observed long-running handle terminates with exit 0. All four ABBA
cohorts complete, six genuine warm requests per route. Native GPU median is
**85.010773396 s**, compatible Comfy default-attention/ordinary-residency
**295.438371146 s**, ratio **3.475305062x**, speed gate **true**. All 16
MP4 outputs independently retain 25 frames, 1024-square dimensions and 24 FPS;
both task-owned Comfy servers report terminal exit. Only UNET/VAE loader
nodes are cached; import, resize, VAE encode, sampling, decode and export all
execute each request. Original model/source/native-library/runtime-build/
reference identities remain unchanged across the complete cohort. Record:
`seedvr2-media-lifetime-video1024-t25-paired-default-v1-20261014`.

The result binds the committed input-lifetime library SHA
`596c160aabcc7ec248c6e842e86da1b2b27a0338bab41749014dff866a35d7e4`.
The non-CUDA cache-transfer compatibility guard remains explicitly disclosed;
this is not an unpatched-stock MPS video result or BF16 numerical-equivalence
claim. No new experimental binary/source is substituted into these timings.
The owned-input experiment can now be built in a separate output directory
and screened; previous tested artifacts and failed histories remain intact.

### Four-media post-encoder input release exact screen

After the live preceding full-request cohort terminates, the new probe builds
in `build/seedvr2-owned-input-probes-v1` without overwriting tested binaries.
Probe SHA: `f1dfc753f14538ef5d5de153c8426f7587cefab72c68f6d21fb956edd75ce5fc`.
The host runner refuses unfinished/failed/partial or identity-mismatched
preceding cohorts, live reference servers and wrong native child identities.
Four CPU guard tests pass, including exact-byte signed-zero differences and
nonfinite-output rejection. Original model/probe/input identities are checked.
`no-dumps` suppresses per-block tensor dumps to respect the remaining disk
space; final condition, velocity, latent and pixels are still saved and
compared. This does not omit model computation.

All four real-media component cases pass exact byte equality and finiteness
for all four saved tensors. Record:
`seedvr2-owned-input-full-media-screen-v1-20261014`.

| Component observation | Borrowed / owned active peak (bytes) | Owned logical RGB bytes released |
| --- | ---: | ---: |
| Image 512 | 9,618,429,130 / 9,615,283,410 | 3,145,728 |
| Image 1024 | 16,472,933,562 / 16,460,350,658 | 12,582,912 |
| Real video 512, all t25 | 10,785,540,622 / 10,706,897,430 | 78,643,200 |
| Real video 1024, all t25 | 21,275,343,558 / 20,960,770,766 | 314,572,800 |

These are one full-component observation per route, excluding public media
import/export, not balanced warm speed or physical-peak proof. Logical bytes
are separately labeled and are not presented as freed physical memory.
Source ownership is now transferred only for videos and explicitly staged
images in the next public candidate. Default resident images keep their
original evaluation/holders and the non-owning interface. Telemetry declares
the input-release recipe, logical bytes and measured release cost. Actual
App image/video/policy-switch and exported-media checks precede promotion.

### Public post-encoder ownership candidate and actual App verification

The isolated native/Swift/App builds and seven native contracts pass.
Candidate library SHA:
`b567c5fe367277c4777cac54235273c89da8d630df4509a0c0aacda4d02432e9`.
The normalized/padded source handles are transferred and dropped only after
independent matching FP16 conditioning is materialized. Videos and staged
images use this path; resident image source/RGB evaluation and retention
remain unchanged. No source pixel read follows the release inside sampling
or decoding; frame geometry and diagnostic input dumps precede the move.
Original model/numerical recipes, alpha/noise and conservative planning stay
unchanged. Release microseconds are included in restoration/request totals.

All three actual App chains terminate with success, seven requests each:
four media/size cases and resident/staged/resident transitions, executed
precision/counters/ownership telemetry, complete video geometry/FPS, history
and unload. Four exported-media required-equality pairs all pass against the
preceding v3 input-lifetime build and between the new hybrid policies. PNG
files and every decoded video frame are exact; MP4 container hashes remain
separate. Records: `seedvr2-app-owned-input-{gpu,resident,staged}-e2e-v1-20261014`
and `seedvr2-owned-input-app-exact-screen-v1-20261014.json`.

| App 1024/t25 observation | Seconds | Active MLX peak (bytes) | Sampled footprint (bytes) |
| --- | ---: | ---: | ---: |
| GPU resident | 84.806948 | 20,967,527,302 | 24,440,527,184 |
| GPU/CPU+NE resident | 83.289200 | 21,114,354,574 | 24,962,882,184 |
| GPU/CPU+NE staged | 83.728000 | 14,318,406,494 | 17,891,329,944 |

Active peaks drop another approximately 314.6 MB in the video observations.
Sampled process peaks vary and are lower bounds; no universal physical peak,
hard budget or equally large speed win is inferred. Default resident-image
active peaks exactly match the preceding version. This is tested functional
memory optimization, not a relabeling of the previous library's balanced
Comfy or GPU/hybrid speed records. New-binary warm speed qualification and
physical ANE activity/placement remain separate boundaries.

### Committed ownership revision: balanced 512-video before/after observation

The four-process before/after/after/before complete-request cohort terminates
with exit 0, six genuine warm requests per revision. Library/source/build
identities remain fixed, and full import/encode/sample/decode/export runs each
request. At 256->512 / all t25 / 24 FPS, borrowed-normalized-input library
median is **20.588601979 s**, committed ownership library
**20.601652062 s**, about 0.0634% slower; strict speed gate **false**.
The code is not described as a speed improvement and no threshold is changed.
Observed warm active maxima fall **10,785,491,118 -> 10,761,406,998 bytes**,
24,084,120 bytes rather than the 78,643,200 logical input bytes released.
This aggregate workload peak is not inferred by subtracting tensor bytes.
Record: `seedvr2-owned-input-video512-t25-revision-abba-v1-20261014`.

The former library's complete 1024 Comfy default speed gate remains bound to
SHA `596c160...`; the committed ownership library is SHA `b567c5f...` and its
actual App/component quality/memory proofs remain separately recorded. The
newer ownership library's own reference/hybrid speed evidence and physical
ANE execution remain further qualification work, not completed by this
before/after memory observation.

### Final handoff: existing measurements, no new inference experiments

At the user's request, stop the optimization/benchmark loop and deliver the
tested `b567c5f...` native library with its complete App image/video integration.
The existing 21 App requests and exported-pixel equality remain the functional
evidence; the final delivery is not a new paired speed qualification.
The targeted App build now includes its sibling model-library helper, and
packaging accepts `TURBOCIDER_PACKAGE_OUTPUT_DIR` so a live older App need not
be overwritten. The intended bundle is `dist/seedvr2-20261014/TurboCider.app`.
The canonical old App, native build, weights and user state are preserved.

Latest GPU-resident App observations (decimal GB) are:

| Output | Active MLX peak | Sampled process footprint |
| --- | ---: | ---: |
| Image 512 | 9.623 GB | 13.066 GB |
| Image 1024 | 16.491 GB | 19.900 GB |
| Video 512, all t25 | 10.710 GB | 14.638 GB |
| Video 1024, all t25 | 20.968 GB | 24.441 GB |

The original MLX parameter bank is 7.2855 GB: DiT 6.7842 GB and VAE
0.5013 GB. At the 1024 resident image/video active peaks this is roughly
44%/35%; the remainder is activations and workspaces, not an exactly measured
VAE-only allocator bucket. VAE decoding/convolution workspaces are the main
video peak hotspot. The 1024/t25 GPU request spends 50.148 s decoding,
20.209 s encoding and 14.258 s denoising: about 59%/24%/17% of the request.
The 1024 image instead spends 1.857 s denoising and 1.615 s in combined VAE,
about 53% and 46%. Import/export is small in these witnesses.

Hybrid staged/resident observations must be compared within that route:
14.318/21.114 GB active and 17.891/24.963 GB sampled process footprint.
Staging releases the GPU DiT parameter bank during VAE work, but retains the
Core ML bank outside MLX accounting. Sampled process/Metal figures overlap,
can miss short peaks and are not hard budgets. No measured physical ANE
occupancy, universal peak bound or new-binary Comfy speed gate is claimed.

Delivery completes without further inference: targeted Swift/App/helper build,
five CPU-only release tests, shell syntax and whitespace checks all pass.
The prepackage executable remains SHA
`6ef19c70fa54b2b7b02436dc5abfe22ef9cd52a436f6e714db4d0d3a9d354061`;
the native library remains the tested `b567c5f...` identity. The fresh bundle
and CLI pass ad-hoc signature verification and public-only release guards.
The packaged helper resolves the already registered `seedvr2-3b` model from
the existing default library; no settings/registration mutation is needed.
Packaging changes dependency rpaths/signatures, so the delivered App/library
have distinct hashes:
`43fe6ae5791feb8da36c26be9cb74a53a473eea3ac08e34ba33f993ae7739944` /
`d421f2a6691d8bf2962312ec87c41cdebb87088d8e694ac12db4370dddfbc92a`.
The running canonical old App/executable/library hashes are unchanged.
The new bundle is handed off for launch after the old App exits, avoiding
concurrent writers to their shared default state. No new GUI-click, inference,
Comfy benchmark or ANE-occupancy result is claimed by this packaging step.

### Active VAE optimization: streaming/configuration candidate (not final qualification)

The new goal retains full VAE acceleration, spatial tiling, buffer reuse,
public Core ML/ANE convolution/full-VAE feasibility, configurable tradeoffs
and real long-video/low-memory App support. Requirements and uncompleted work
are tracked in `docs/design/seedvr2-vae-optimization.md`; the 121-frame model
limit has not been lifted by the writer work.

Candidate v1 library SHA:
`ae18c7ae47bb0dcca327392f9018f4346960f40712f8b6b8828a5ef027ed55b3`.
Adds `seedvr2_vae` request-local decoder/convolution/tile-row/allocator-cache
controls in schema 1 and schema 2, matching Swift SDK/draft persistence and
App selectors. Streaming output preserves the original temporal slices and
causal histories, hands off one completed frame to a backpressured MP4 writer
and never constructs the complete decoded RGB tensor. Full source/condition/
DiT state still exists. Decoder wall/export telemetry avoids double-counting
the sink. Buffered mode remains default; stream mode refuses full pixel dumps.

Isolated native and Swift/App builds succeed. Nine native plan contracts,
repository C++ boundaries (2), independence (7), revision observer (3),
VAE execution-evidence observer (5) and media-screen guards (4) pass.
The writer tests pass exact frame/FPS geometry at 16/24 FPS, order/size guards,
incomplete/cancelled output cleanup, preserved existing destination and a
1000-frame producer. This is media export, not 1000-frame SeedVR2 inference.

Two real t25 latent codec screens compare every FP16 frame bitwise:

| Decoder-only observation | Buffered / streaming active peak | Buffered / streaming-with-host-comparison time |
| --- | ---: | ---: |
| 512 t25 | 3,914,304,330 / 3,893,332,818 bytes | 12.2176 / 12.1977 s |
| 1024 t25 | 14,153,255,758 / 14,069,369,686 bytes | 50.0647 / 49.9887 s |

These single observations exclude import/DiT/export and do not establish a
paired speed or whole-process improvement. Records:
`seedvr2-vae-stream-decoder{512,1024}-t25-screen-v1-20261014`.

The resident GPU/4-GiB-cache actual App chain passes seven requests, including
both image/video sizes, residency switches, history and unload. Required
comparison against the preceding correct App passes PNG file equality and
every decoded video frame. Records:
`seedvr2-app-stream-decoder-gpu-e2e-v1-20261014` and
`seedvr2-app-stream-decoder-gpu-exact-screen-v1-20261014.json`.
Its 1024/t25 observation is 84.839032 s, active peak 20,883,641,226 bytes and
sampled process footprint 24,676,162,016 bytes. The latter is not a demonstrated
improvement over the previous App; do not infer physical savings from RGB
logical bytes.

The same-library buffered/streaming full-request ABBA terminates successfully,
six real warm requests per mode, but strict speed gate **false**:
20.715874250 / 20.768963250 s (streaming about 0.256% slower).
Warm active maxima are 10,706,881,506 / 10,736,241,178 bytes: streaming's
aggregate peak is 29,359,672 bytes higher, despite the decoder-only observation.
It is not promoted as a whole-request speed/peak win. All eight cold/warm
export pairs independently preserve all t25/24-FPS frames bitwise. Records:
`seedvr2-stream-decoder-video512-t25-policy-abba-v1-20261014` and
`seedvr2-stream-decoder-video512-t25-policy-exports-v2-20261014.json`.
The first export-verifier prototype incorrectly treated ffprobe metadata as
a path; its failed v1 record is retained, and v2 reads the actual metrics path.

The staged GPU + streaming + 1-GiB-cache App chain also completes seven
requests. Its 1024/t25 observation is 95.990742 s, active peak
14,087,719,778 bytes, sampled footprint 14,880,835,120 bytes. This is a slower,
lower observed process-footprint configuration on the 64-GiB host, not proof
of a hard budget or of execution on a 16-GiB machine. **Required export
equality fails for the 1024 video** (decoded RGB RMSE 3.7598, maximum error
114); both PNGs and 512 video are exact. This is not called qualification or
silently relaxed. Records:
`seedvr2-app-stream-decoder-staged-cache1-e2e-v1-20261014` and its
`seedvr2-app-stream-decoder-staged-cache1-exact-screen-v1-20261014.json`.

Diagnosis is active before recommendation/promotion: isolate cache-dependent
raw FP16 VAE outputs against saved correct latents/pixels, and independently
hash exact RGB24 bytes before H.264 encoding. A separate v2 native candidate
adds opt-in `TURBOCIDER_SEEDVR2_VIDEO_PIXEL_HASH=1` telemetry for that purpose;
the flag is disabled normally and is not speed evidence. Codec differences
must be distinguished from actual model/quantization differences rather than
assumed. Existing tested/shipped binaries and all failed histories are intact.

The cache1 reference probe now passes: all buffered FP16 pixels match the
previous saved correct decoder witness, and all streamed frames match those
pixels bitwise. This independently isolates the VAE under the smaller cache.
Record: `seedvr2-vae-stream-decoder1024-t25-cache1-reference-screen-v2-20261014`.
Two v2 complete requests then produce identical pre-encoder RGB24 SHA
`b84615183ea8de4315676019440b04b117902f27bbde4572e14c69318835602c`
under resident/cache4 and staged/cache1. Their diagnostics are not timings
or proof that every old encoder output was identical.

The failed v1 staged movie has its first eleven frames as I frames; the
resident movie has only the first. A new streaming writer explicitly sets
source FPS, MaxKeyFrameInterval to 2*FPS and MaxKeyFrameIntervalDuration to
0.0 (no duration limit, as documented in the Apple SDK). The buffered legacy
writer is unchanged. Recipe:
`h264_source_fps_frame_gop2s_no_duration_limit_v1`.

A fast/slow 64-pixel producer check, including three 2.2-second delays,
produces identical decoded RGB SHA
`a2e46643697cc0e688bcbb46fe99076979101f6502d01bd455bda070f75cec17`.
The slow movie has one I frame and eight P frames. The actual 1024/t25 v3
resident/cache4 and staged/cache1 complete requests retain the same raw RGB
SHA above, and every decoded frame has aggregate SHA
`410bd515347d86d8a68fc42d6c0da3764c627d34545eea0ce9d788ecf00cecde`
in both policies. Independent ffprobe confirms 25 frames, 1024 square and
24/1 FPS; staged movie now has one I frame and 24 P frames.
Records: `seedvr2-stream-frame-gop-{resident4,staged1}-hash-v3-20261014`.

The v4 candidate additionally preserves the native typed cancellation error
inside writer backpressure/finalization and builds successfully. Library SHA:
`ff4b650eccf95db40275db7c7be33910044a47904f4aecb38984e35debff79d8`.
Its actual App requalification and typed-cancellation writer checks are next;
v1 performance evidence is not relabeled as v4 speed evidence. No shipped
App/native binary has been replaced during this qualification.

The v4 writer's typed cancellation, cleanup, geometry, 1000-frame producer
and delayed-producer checks pass. Native plan (9), execution-evidence (5),
revision observer (3), media guards (4), C++ boundary (2) and independence (7)
checks pass; this is not a claim that unrelated repository-wide tests pass.

Both v4 actual App chains now finish successfully, seven requests each:
resident/4-GiB cache and staged/1-GiB cache, with streamed video output,
buffered alpha-preserving images, residency transitions, history and unload.
Required **new-policy** PNG and every decoded-video-frame equality passes
at 512 and 1024. Record:
`seedvr2-app-stream-frame-gop-export-screen-v4-20261014.json`.
The deliberately separate previous-encoder comparison keeps both PNGs exact
but reports different video pixels (512/1024 exported RGB PSNR 33.347/39.189
dB). Do not label the new explicit encoder recipe as bitwise equivalent to
the legacy encoder or erase the earlier failed cache1 comparison. Raw VAE /
pre-encoder evidence above separates restoration from this encoding change.

| v4 App observation | Resident cache4 / staged cache1 request time | Active MLX peak | Sampled process footprint |
| --- | ---: | ---: | ---: |
| 512 t25 | 20.770367 / 21.731974 s | 10.689 / 9.035 GB | 14.625 / 9.899 GB |
| 1024 t25 | 84.841121 / 94.278125 s | 20.884 / 14.088 GB | 24.364 / 14.862 GB |

These are individual App observations on the 64-GiB host, not a paired speed
gate or a physical upper bound. The smaller cache is slower and reduces
observed retained process memory; it is available explicitly, not selected
as a speed default. Buffered/singleton2d/cache4 and the old stable App remain
available. The active objective is not complete: reusable causal/workspace
buffers, full spatial tiling, VAE ANE convolution/full-VAE feasibility and
genuinely longer-model clips still require implementation and qualification.

### Active VAE frame-history reuse candidate

The next candidate replaces repeatedly rebuilt two-frame histories with
individually owned frame references. Old tails are shared unchanged; an
untrusted new frame is physically materialized with `full`. Only a known
independent, materialized singleton Metal GroupNorm output may be retained
without copying. Do not infer ownership from contiguous flags. Normal MLX
convolution builds its required halo+input directly from those frames, without
an intermediate contiguous history. Bounded convolution reads the two history
buffers virtually. Temporal context, stride/phase, weights and precision are
unchanged. The established `compact` implementation stays default.

Tests exposed and corrected prototype assumptions before model qualification:
`copy`/`contiguous` may share an input backing, scalar/small Metal arguments
have different pointer/address-space forms, and split history needs singleton
buffer geometry rather than a contiguous halo tensor. Failed v1-v4 probe
binaries/output history are preserved. v5 CPU tests cover exact bytes including
signed zero/NaN payload, temporal updates, reused tail references and independent
untrusted-view storage. GPU tests compare split/contiguous histories exactly
for scalar/vectorized channels and both temporal strides.

Real 512/1024 t25 decoder checks pass every FP16 pixel against saved correct
reference outputs, and streamed frames match the newly buffered output too.
One-observation active peaks:

| Decoder witness | Frames buffered / streamed active peak | Previously recorded compact buffered / streamed peak |
| --- | ---: | ---: |
| 512 t25 | 3,679,423,306 / 3,658,451,794 bytes | 3,914,304,330 / 3,893,332,818 bytes |
| 1024 t25 | 13,234,703,178 / 13,129,845,586 bytes | 14,153,255,758 / 14,069,369,686 bytes |

Records: `seedvr2-frame-history-video{512,1024}-t25-exact-screen-v1-20261014`.
These are component observations, not paired speed or physical-footprint
proof. The normalized input/DiT parameters and full source/conditioning scope
remain separate; this is not full VAE spatial tiling.

Request-local `seedvr2_vae.history` now selects `compact` or `frames`.
Both schema plans expose the effective selection; Swift/older draft decoding
defaults missing history to compact, and the App exposes an explicit picker.
Ten native request contracts, execution-evidence checks (5), signed-zero-aware
VAE observer tests (8), C++ boundaries (2) and independence (7) pass. The initial
project Python lacks safetensors and reports the VAE tests as skips; they run
successfully in the existing validation interpreter without installing packages
or contacting a user Comfy server.

A same-probe/source/precision 1024/t9 VAE ABBA completes under
`seedvr2-frame-history-vae1024-t9-abba-v1-20261014`. Both routes use singleton2d,
Metal norm, FP16 activations and pointwise matmul; only history differs.
Four processes each run one cold and three real warm encode+decode iterations.
No encoded latent cache shortcut is used. Warm medians are **25.205170 /
24.766025 s**, about **1.01773175x**; strict component speed gate true. Warm
active maxima are **14,198,344,530 / 13,258,820,426 bytes**, reduction
939,524,104 bytes. Latent and decoded tensors are finite and **byte-identical**
in every required comparison, including signed-zero semantics. This is VAE
encode+decode, not whole-model or hard-process-budget qualification.

The isolated native/Swift/App candidate succeeds; native library SHA
`d3d9e5c182b438974aa7333f5f7b2c546f6c1d4d6399bb3434a665dcf8566873`.
Both actual App chains pass seven requests each: images and complete t25
videos at 512/1024, policy switches, persistence and unload, with frames
history, streamed video and the unchanged explicit frame-based encoder.
Required comparison to the preceding v4 correct build keeps PNG files and
every decoded video frame exact in both memory policies. Record:
`seedvr2-app-frame-history-export-screen-v1-20261014.json`.

| Current 1024/t25 App observation | Seconds | Active MLX peak | Sampled process footprint |
| --- | ---: | ---: | ---: |
| GPU resident, cache4 | 83.606837 | 19,944,111,514 bytes | 23,894,743,424 bytes |
| GPU staged, cache1 | 91.253532 | 13,215,304,538 bytes | 14,576,944,616 bytes |

These App rows are individual observations, not a bound on a smaller-memory
device. The same-library complete 1024/t25 request ABBA subsequently terminated
with exit 0 under `seedvr2-frame-history-video1024-t25-policy-abba-v1-20261014`,
streamed output/cache4 in both modes, changing only compact/frames history.
Four independent processes and six warm requests per policy give medians
**85.118659 -> 83.792985 s**, **1.015821x**, passing the strict speed gate.
Active maxima fall **20,876,885,338 -> 19,937,360,594 bytes**, a reduction of
**939,524,744 bytes**. The export screen verifies all 16 outputs/eight pairs:
every decoded frame is bitwise equal, full t25 geometry and 24 FPS preserved.
Record: `seedvr2-frame-history-video1024-t25-policy-exports-v1-20261014.json`.
This does not establish a smaller-machine physical peak. Default history remains
compact and the whole original objective remains open; no job was restarted.

### Original-parameter public Core ML VAE convolution feasibility probe

A separate offline bank targets the hot
`decoder.up_blocks.3.resnets.0.conv1` original O,C,T,H,W filter
`[128,256,3,3,3]`: prefix 64 outputs, prepared 128-square spatial tile plus
halo, original FP16 coefficients, temporal taps stacked into channels.
Bias/residual/normalization stay outside the bank to preserve native bias
rounding. It has a distinct **probe** schema and remains `qualified=false`;
the App/native hybrid VAE does not consume it.

Coefficient-order/prefix/shape/precision CPU tests (3) and independent
reference/quality-screen guards (3) pass. Actual Core ML CPU-only and CPU+NE
predictions plus GPU Conv3D match an independent original-order FP32
reference under unchanged cosine >=.999 / relative-L2 <=.05 gates. A separate
27-point FP64 reduction check verifies the finite FP32 reference (max absolute
error 2.768e-6), despite NumPy/BLAS floating-point warnings retained in output.

At this prepared synthetic tile, warm medians are **3.24954 ms GPU /
16.43825 ms CPU / 13.93738 ms CPU+NE**. The Core ML plan's namespaced
`ios18.conv` reports anticipated NeuralEngine preference/support. That is not
measured physical occupancy. The first plan-inspector filtered only `conv`
and missed its namespace; the v1 record remains intact and v2 corrects that
inspection. Public E5RT cache access required scoped host permission.
Record: `seedvr2-vae-hotconv-coreml-feasibility-v2-20261014.json`.

This is not a native zero-copy/cooperative scheduler or real-media full-VAE
quality/speed gate. Python Core ML copies and GPU readback are included;
packing/routing/full-frame overhead is excluded. This layout is slower than
GPU and is not promoted. Native-buffer scheduling, larger/different tiles,
whole-convolution/stage assignments and full-VAE feasibility still require
experiments; no VAE ANE acceleration or automatic route is claimed.

### Native fixed-tile IO changes the VAE ANE feasibility conclusion

Subsequent public-API native IO screens on M4 Max/64 GiB/macOS 26.6.2 add real
FP16 IOSurface-backed input/output reuse, regular/wrapped MLMultiArray controls,
CPU-only and CPU+NE controls, signed-input ABA verification, and native MLX
completion-timed GPU baselines. All requested output backings return expected
identity and every prediction passes the unchanged finite/.999-cosine/.05-L2
component gates against independently checked original-order FP32 references.
No private ANE API, original-parameter change or production VAE backend is added.

The full 128-output hot convolution at 64-square observes **0.818 ms GPU Conv3D /
0.688 ms GPU pack+Conv2D / 0.526 ms CPU+NE with reused surfaces**. The last timing
excludes packing into its surface: warm CPU refill alone is about 0.182 ms and
can consume the 0.162-ms GPU margin. A 256-square/64-output CPU+NE tile is about
26.235 ms versus 3.123 ms GPU pack+Conv2D. One real 256->1024 upsampling projection
shows a component opportunity (0.337 ms CPU+NE), while 512->4096 does not show a
standalone advantage. Reversing short-screen route order supports investigation,
not production/default speed qualification or a universal tile-size rule.

Records: `seedvr2-vae-native-io-tiles-v1-20261014/report.json`,
`seedvr2-vae-native-io-tiles-v2-20261014/report.json`, and
`seedvr2-vae-native-io-full64-v1-20261014/report.json`.
Detailed measured scopes, memory observations, global-GroupNorm/causal barriers,
fixed graph candidates and GPU/ANE scheduler gates are documented in
[VAE ANE feasibility](../design/seedvr2-vae-ane-feasibility.md).
Backing identity does not prove no internal copies; MLComputePlan anticipated
NE preference does not prove physical occupancy. Cooperative execution, real
full-frame/whole-VAE quality/speed, ANE memory reduction and App VAE integration
remain unqualified. Canonical App/native binaries and defaults are untouched.

### Actual GPU shared-IO / Core ML convolution boundary and negative promotion gate

The independent native prototype now runs GPU temporal/halo pack into public
IOSurface-backed FP16 arrays, serial and bounded two-slot schedules, stride-correct
GPU assembly and disjoint GPU output-channel suffixes. It supports scalar and
16x16 threadgroup transposes. All packed/assembled elements have independent
bitwise witnesses; every full-frame output is NaN-poisoned before inference
and passes the unchanged original-order FP32 numerical gate. Submitted commands
retain their external surface/MLX owners and drain on exceptional unwinding.

The initial strict page-size guard fails because CoreVideo's logical allocation
size includes 64 metadata bytes. The failure is preserved. Rounded no-copy Metal
bindings are now allowed only after public VM-region inspection proves the full
readable/writable mapping; logical kernel extents remain bounded. Every Core ML
output must honor its backing identity, shape and strides exactly.

The final corrected 512-square full-convolution screen has two independent
processes per route, forward/reverse order and nine warm full-spatial requests
each. Medians: **32.580 ms GPU Conv3D / 44.833 ms GPU pack+Conv2D / 61.325 ms
CPU+NE serial / 35.386 ms CPU+NE pipeline / 42.753 ms NE-prefix+GPU-Conv3D /
42.913 ms NE-prefix+GPU-Conv2D / 92.702 ms CPU-only serial**. Original full
128-channel weights and all 64 tiles are included; mixed routes merge the real
GPU suffix. Pack/producer wait/predict/consumer wait/layout and suffix join are
inside the observed boundary. This is not an end-to-end VAE/App request.

Corrected child process footprint maxima are **1,208,304,984 GPU Conv3D /
1,233,372,504 pipeline bytes**; MLX active peaks are both **480,002,072 bytes**.
The external two-slot pool is 21,626,880 mapped bytes and is not MLX-active
memory. An earlier v4 probe mistakenly charged GPU-only an unused 64-MiB
assembly buffer; v5 removes it and reruns the comparison. Neither a speed nor
memory win over GPU is established, so no production VAE/App route is enabled.

Each successful 512 non-GPU process bitwise-checks 642,318,336 pack values and
100,663,296 full-output assembly values (half-prefix: 50,331,648), then runs 768
real predictions with exact backing hits. All full outputs pass; the independent
27-point FP64 reference check has maximum error 3.11e-6. CPU guard/cohort tests
pass **20/20** with the existing NumPy/safetensors validation interpreter.

Final record: `seedvr2-vae-shared-tile-512-transpose-v2-20261014/report.json`.
The earlier successful scalar/transpose 128 screens, v4 512 memory-biased screen
and failed page-binding screen remain intact. Details and scopes are in
[VAE ANE feasibility](../design/seedvr2-vae-ane-feasibility.md). Full VAE/context,
real activation/video qualification, physical ANE residency, true longer clips
and low-memory hardware remain incomplete. No App rebuild/test is claimed by
this isolated probe commit; canonical App/runtime/defaults are unchanged.

### Bounded RGB source / causal VAE encoder integrated as an explicit option

The previously uncompiled RGB reader is now integrated into the native build
and the explicit video-only `seedvr2_vae.encoder=streaming` path. Native plan,
C/Swift SDK, persisted App draft and input-residency control expose it in both
schema versions. Buffered remains default, including old drafts. Full tensor
dumps and native still-image streaming requests fail explicitly; the App adapts
image source/sink settings to buffered. Existing image/alpha math is unchanged.

`VAE::encode_stream` uses the existing `encode_slice`, first-five/subsequent-four
partition, whole-frame normalization/attention and original causal history.
Singleton normalized FP32 frames are cast/evaluated to owned FP16 before host
reuse; only a bounded chunk is concatenated. Padding repeats the last real
frame. Pipeline source EOF verification precedes sampling/DiT staged reload.
This is not full spatial tiling, independent restoration windows or a new
long-model frame limit; complete conditioning/DiT state and <=121 remain.

Reader tests pass all eight display transforms, every decoded/normalized FP32
value in 72 frames, exact EOF/count/FPS/transform guards, VFR and file-stamp
change rejection, typed pre/mid-read cancellation and 1000-frame complete
reader consumption with constant 18,432-byte caller RGB capacity. The legacy
bulk loader still rejects 1000, and no longer-model support is claimed.
Source counts 1/2/4/5/6/9, including padding and multiple causal chunks, encode
to completely byte-identical FP16 latents. Records:
`seedvr2-rgb-reader-guards-v3-20261014/report.json` and
`seedvr2-vae-encode-source-128-v1-20261014/report.json`.

Isolated full native/App candidate: `build/seedvr2-stream-input-native-v1`.
Library SHA-256: `b2b13198a0e117dde2e7780f1de7af97cdfe13348642b6d747f65672066a0c40`.
Build identity: `tc-runtime-build-v1-df82b76b2a9e9a7a4fbadfe6d11de5686b43bf9baca6e37bf4610ba7cb7ead58`.
Both real App workflows finish seven requests: resident/cache4 and staged/cache1,
512/1024 images and complete 25-frame/24-FPS movies, draft/history/configuration,
resident/staged switches and unload. PNG RGBA and every video frame match the
previous frame-history candidate exactly in both policies. Record:
`seedvr2-app-stream-input-export-screen-v1-20261014.json`.

| New actual App 1024/t25 observation | Request seconds | MLX active maximum (bytes) | Sampled process footprint (bytes) |
| --- | ---: | ---: | ---: |
| Stream source/sink, resident/cache4 | 83.549936 | 19,944,117,114 | 23,875,590,456 |
| Stream source/sink, staged/cache1 | 90.754717 | 13,215,304,530 | 14,476,395,936 |

These are single observations and sampling lower bounds, not paired 1024 speed
or smaller-device physical budgets. Reused source RGB24 capacity is 786,432
bytes for these 512-input movies; that excludes codec/normalized chunks/model
state and is not a whole-process limit. Reader/resize work is timed inside
encoder wall and also separated as frame/media import; overlapping fields must
not be summed. There is no complete normalized RGB owner to report as released.

The same-library 512/t25 staged/cache1 full-request ABBA finishes four processes
and six warm requests per encoder mode. All else (sink/history/convolution,
source, precision, seed sequence/cache/residency) stays identical. Medians are
**21.254022 -> 21.251528 s**, **1.000117x**, exact speed gate true but effectively
flat. Active maxima are **9,035,267,926 -> 9,035,230,774 bytes**, a negligible
**37,152-byte** change, not a material request-peak win. All 16 outputs/eight
pairs have every decoded frame bitwise equal, complete dimensions and 24 FPS.
Records: `seedvr2-stream-input-video512-t25-staged-abba-v1-20261014/report.json`
and `seedvr2-stream-input-video512-t25-staged-exports-v1-20261014.json`.

A separate actual public-engine lifecycle screen cancels on source frame one
(status 2), rejects an extra source frame before sampling (status 1), then runs
a complete recovery on the same engine (status 0). Source and preexisting
destination survive; no private staging remains; unload reports active bytes 0.
Record: `seedvr2-stream-input-lifecycle-v1-20261014/report.json`.
Native contract tests (11) and full-request observer controls (6) pass. The
initial broad minimal-environment CPU guard run lacked safetensors for one
case; the same 17 guards subsequently pass with the existing validation
interpreter, without changing thresholds or installing into user environments.

The tested build is packaged separately at
`dist/seedvr2-stream-input-20261014/TurboCider.app`, with sibling `cli/` and
minimum macOS 26.2. Public release-binary checks and ad-hoc deep/strict signature
verification pass. Packaging does not replace the canonical running App/state;
close the older instance before opening this bundle with the same default
state directory. No model/media files or old failed records were removed.

No automatic source promotion or 16-GiB success is claimed. Inspection finds
that conservative residency admission can still reject smaller-memory hosts
at 512 video; this remains a separate estimate/hardware qualification task.
Full spatial VAE scheduling, workspace pools, larger-context DiT, useful VAE
ANE graph boundaries and real lower-memory hardware remain open.

### Global-statistics row/file decoder prototype: lower component peaks

`VAE::decode_spatial_stream` is an explicit offline entry, not selected by
requests/App/environment defaults. Eligible high-resolution norm/conv stages
use full-frame FP32 centered statistics and halo-correct row bands. Attention,
causal taps, padding, coefficients and temporal phases retain their original
scope. Large video histories use checksum-verified lossless private FP16 files
and bounded owned GPU read windows. This is high-resolution stage scheduling,
not independent crops or complete tiling of every VAE operator.

Default decoding still matches saved native output bytes. Ten global-statistics
normalization cases match byte for byte, and file/signed-zero/NaN/checksum/quota,
ownership/unknown-child/cleanup guards pass. Spatial cancellation and sink
failure drain files/windows and recover on the same VAE. Tile-shaped convs
change reduction rounding: prototype pixels pass quality but are not byte exact.

Single decoder-only/cache1 observations, including host quality scans:

| Component workload | Default / prototype seconds | Default / prototype active peak (bytes) | Default / prototype sampled process footprint (bytes) |
| --- | ---: | ---: | ---: |
| 1024/t1 codec image | 1.329 / 1.308 | 9,670,740,818 / 3,181,661,286 | 9,774,318,576 / 4,206,871,440 |
| 1024/t5 restored-video offline prefix | 10.847 / 35.855 | 13,121,981,262 / 5,639,867,218 | 14,464,500,768 / 6,690,424,008 |

The latter emits/checks all five selected diagnostic frames, not an App source
truncation or complete t25 qualification. Live files peak 6.5 GiB, with 15 GiB
writes and about 23.7 GiB reads; owned read windows peak 69,206,016 bytes. File
budget is 8 GiB with a free-space reserve, and only exact owned scratch is
retired. The single-frame path writes no histories. Timings/physical samples
are not balanced full-request/warm speed or hard memory/device guarantees.

The first attempted original-FP32 check correctly rejects the older saved
oracle pixels: existing restore `dtype=fp32` specifies DiT, while that oracle
explicitly uses an FP16 VAE. Existing records remain unchanged; future legacy
oracle reports now also expose `vae_compute_dtype`. A new VAE-only MPS oracle
promotes original coefficients to FP32 and checks all floating dispatch values
and parameters, with fallback forbidden and no service/source modification.
It covers the exact same latent prefix, output F32 `[1,3,5,1024,1024]`, 5,435
dispatches and 12,504 checked floating values. Default/prototype whole-pixel
relative L2 is **.000489190 / .000489242**, cosine **.999999883 / .999999883**,
passing unchanged .05/.999 gates. This is not a whole DiT/request claim.

Latest record: `seedvr2-spatial-restored-video1024-t5-batched-io-v3-20261014/report.json`.
Independent reference: `seedvr2-spatial-video1024-t5-fp32-vae-oracle-v1-20261014/report.json`.
Detailed scope, tradeoffs, files and next gates:
[global-statistics spatial scheduler](../design/seedvr2-vae-spatial-scheduler.md).
At this initial prototype milestone the active objective remains open: public
optional controls, full App/complete video qualification, broader stage/encoder scheduling, conservative admission,
bounded longer-context DiT and actual low-memory hardware remain incomplete.
The packaged `26f23e6` App is unchanged; no new default or release is claimed.

### Explicit row/file controls: complete25 FP32 quality and real App lifecycle

The subsequent candidate adds request-local `spatial=full|file_rows`, 8-aligned
`spatial_rows` in8...128 and a positive live-file budget <=8 GiB. Schema1/2 and
Swift drafts carry the same fields; omissions/older drafts keep `full`. The
public file-row canvas is bounded to1024 per axis, with only
singleton2d/conv3d accepted. The App offers an explicit selector/row height and
IO/numerical warnings; it does not automatically recommend spilling. This
still schedules only eligible decoder stages, not the encoder/entire VAE.

Complete1024/t25 decoder observations are56.074/189.350 s and active peaks
13,196,954,446/5,647,731,538 bytes (full/file-row), including host quality/export.
Sampled process peaks are14,661,682,424/6,824,477,992 bytes. The file route writes
75 GiB, reads about142 GiB, peaks at6.5 GiB live files and69,206,016-byte owned
read windows. These single component observations are not full-request speed
or device budgets.

The original full25 FP32 oracle attempt exits137 without usable output; its
cause is not established. A distinct complete-context attempt uses existing
CPU output-buffer storage while retaining audited FP32 MPS arithmetic. All
parameters/62,316 floating dispatch values are FP32; output is
F32 `[1,3,25,1024,1024]`. Every same-latent decoded pixel passes unchanged gates:
full/file-row relative L2 .000522324/.000522439, cosine about .999999869. Old
FP16-VAE reports and the failed oracle are not relabelled or replaced.

The separately built candidate completes two seven-request real App workflows
(full and file-row, staged/cache1, source/sink streaming, frame history), covering
512/1024 images/videos, complete t25/24-FPS exports, persistence, resident/staged
switches and unload. File-row1024 executes144 image tiles/eight global norms
and3504 video tiles/200 global norms; live file/window counters and cleanup
failures are zero after decode. Export comparison checks every frame under
unchanged .05-L2/.999-cosine gates plus exact PNG alpha; the512 pixels are exact.
The1024 H264 exported-video error includes lossy encoding and is not the raw
FP32 VAE error. GUI clicking and broad temporal/perceptual quality are not
claimed by these integration tests.

Single full-request1024/t25 observations are89.902/221.645 s, active peaks
13,148,195,662/12,235,730,826 bytes and sampled process peaks
14,476,445,112/13,240,293,640 bytes. They are not warmed ABBA; the remaining
request peak is not located by phase. Do not apply the decoder-only reduction
to the complete App, lower the admission assertion, or claim16-GiB support.

Actual public-engine tests cancel at first export (status2, no read-ahead),
reject a64-MiB scratch quota (status1, explicit budget error), and recover on
the same engine with a1024 image (status0). Source/existing private destination
hashes remain unchanged and scratch/media staging drain; unload MLX active is0.
CPU guards pass12 contract,7 observer,3 original-FP32-reference and6 exported-
media tests. Candidate file/norm tests and typed-cancel/sink/recovery tests pass.

Independent locally signed package:
`dist/seedvr2-spatial-rows-20261014/TurboCider.app`, minimum macOS 26.2. Release
packaging/signature guards pass; 12 contracts also pass against its actual
packaged library. The packaged CLI's real 1024 file-row image executes 144 tiles /
eight norms with clean retirement, and its PNG is byte-identical to the tested
App workflow output. This package does not replace the currently running user
App; the image smoke is not a packaged-video or GUI-click qualification.

Records and outstanding broader scheduling/admission/ANE/long-context gates:
[row/file scheduler](../design/seedvr2-vae-spatial-scheduler.md). The overall
goal remains active; no VAE ANE backend, >121-frame model support or automatic
file-row default is enabled by this work.

### Encoder-head row/file scheduling and qualified manual App controls (2026-10-15)

Adds independent `seedvr2_vae.encoder_spatial=full|file_rows`, full by default,
with the shared bounded row/live-file settings and <=1024 canvas scope.
Eligible singleton encoder head operators use complete-frame statistics and
lossless causal row files; temporal downsampling/later multi-frame stages/all
attention are unchanged. Source streaming retains the original5/4 partition
and padding; six small buffered/streamed cases are byte exact. Original
coefficients are unchanged, but row-convolution rounding is a distinct numerical
recipe, not bitwise equality to the full encoder.

Image encoder ABBA (four processes, six warm fresh encodes each) observes
.659063/.561925 s, about14.7% lower latency, and active maxima
5,425,578,594/2,931,575,910 bytes. Complete real1024/t25 single encoder
observations are20.665/69.627 s, active8,001,784,646/5,854,301,002 bytes; this
video route is slower, writing25 GiB/reading49.40625 GiB with2.25-GiB live files.
Independent same-source all-FP32 MPS encoder audits pass image/full25 gates:
native/row-file relative L2 .005067376/.005067904 (image) and
.003270006/.003269107 (video), unchanged .05/.999 thresholds. The references
audit actual dtypes/parameters; no FP16 report is relabelled or service modified.

Two seven-request real App workflows pass, including complete25-frame source
FPS, persistence/switches and actual encoder/decoder tile counters/retirement.
The new default exports are byte exact to the old full control; combined rows
pass every exported frame and exact PNG alpha. Single full/combined staged1024/
t25 requests are90.043/272.910 s, active13,215,304,526/12,235,734,666 bytes.
Adding encoder rows does not materially reduce the earlier decoder-only row
request peak. No hard RAM/16-GiB/phase attribution or warm full-request speed
claim follows; admission/121-frame/defaults remain unchanged.

Public engine encoder-head cancellation returns status2 without DiT/export,
64-MiB quota rejects explicitly (status1), and same-engine1024 recovery succeeds
(status0); source/existing private destination and scratch cleanup pass.
Independent signed package `dist/seedvr2-encoder-spatial-20261015/TurboCider.app`
(minimum macOS26.2) passes release/signature guards,13 packaged-library contracts
and a real1024 combined-row image smoke whose PNG matches the App workflow bytes.
The currently running user App remains untouched.

Captured-real Core ML projection/shape-shuffle tests and native fused-upscale
experiments are documented separately. Corrected CPU+NE projection is speed-flat
with a512-MiB prepared-layer allocation reduction, not an enabled VAE backend.
GPU fused-upscale candidates are exact but slower/no whole-VAE peak win; they
are isolated behind an experimental compile flag and rejected by stable
packaging. See [ANE feasibility](../design/seedvr2-vae-ane-feasibility.md) and
[row/file scheduler](../design/seedvr2-vae-spatial-scheduler.md) for receipts.
The overall goal stays active: broader scheduling/workspace reuse, actual
low-memory hardware/admission, useful whole-VAE ANE boundaries and bounded
longer-context DiT remain incomplete.

### Full-decoder ANE projection qualification and default regression (2026-10-15)

The fixed original-FP16 Core ML spatial projection now executes inside the
complete real VAE decoder, behind the existing experimental compile guard.
Only the256->1024 spatial projection changes; outer bias/shuffle and the
following causal convolution retain the original phases/rounding/history.
Complete1024/t25 CPU+NE execution records25 projection frames,1600 predictions
and1600 exact pixel-buffer/shape/stride backing hits. All78,643,200 decoded
values and each of25 frames pass the independent same-latent audited FP32
reference; native/NE relative-L2 .000522323646/.000522323592. Signed projection,
strided inputs, retained output, actual pipeline cancellation, sink failure,
same-runtime/same-VAE recovery and zero-allocation teardown tests pass.

Single decoder observations are55.045/55.363 s, with identical active maximum
13,129,845,582 bytes. NE sampled process footprint is slightly higher, not lower.
The earlier isolated512-MiB reduction does not reduce this decoder's maximum.
The prototype is not a useful qualified speed/peak optimization, a public/App
backend, a whole-VAE conversion or a measured physical ANE overlap. Its failed
initial output-backing lock attempt remains preserved. Ordinary builds and
stable packaging exclude its hook/selector/shader bytes.

Optional bounded read-window pooling reduces allocations6720->912 (5808 actual
reuses), preserves complete25 output hashes and original-FP32 quality, but
single191.402/190.566-s row-decoder observations do not establish speed/peak
improvement. Pooling stays disabled and absent from public/App configuration.

Fresh ordinary native and Swift/App builds pass the release audit,13 public
contracts and a complete seven-request actual App workflow:512/1024 images,
complete25-frame24-FPS videos, streaming input/output, staged/resident switches
and persistence. All exported media are byte exact to the prior qualified
default at decoded-pixel level, including every video frame and exact PNG alpha.
No model/weights/admission/defaults or the existing running App are replaced.

Records: `outputs/seedvr2-vae-ane-full-real1024-t25-{native,ne}-v1-20261015/`,
their separate `*-fp32-quality-v1-20261015.json` receipts,
`outputs/seedvr2-projection-public-app-default-e2e-v1-20261015/generation-report.json`
and `outputs/seedvr2-projection-public-app-export-equality-v1-20261015.json`.
Detailed boundaries and remaining work are in
[ANE feasibility](../design/seedvr2-vae-ane-feasibility.md) and
[row/file scheduler](../design/seedvr2-vae-spatial-scheduler.md). The complete
optimization goal remains active: the true request peak/critical path, broader
scheduling, useful ANE routes, real low-memory hardware and >121-frame model
support remain unresolved.

### Phase attribution and opt-in spatial-upscale history sharing (2026-10-15)

A complete25 original-boundary diagnostic trace has471 verified rows, preserves
all78,643,200 output words and reproduces the noninstrumented13,129,845,582-byte
maximum. The peak is in spatial upsampling plus its following causal conv;
deduplicated cached history already occupies9,060,745,216 bytes. Timings are
bundled/synchronized diagnostics, not independent arithmetic or warm-speed
qualification; inclusive mid-block children are not added twice.

The new public strict-Boolean `seedvr2_vae.share_upscale_history` is false by
default and requires frames/singleton2d/full decoder scheduling. It retains
the completed independent spatial-upscale owner after checking actual backing
extent, rather than copying another frame. Images retain no unused histories.
The stable numerical recipe/coefficients/casts/phases are unchanged. Actual
frame/logical-copy counters are reported;13.422 GB avoided cumulative logical
copy bytes at1024/t25 is not a physical or peak saving.

Materialization alone does not reduce the maximum. Materialization plus owner
sharing lowers the upsample region512 MiB, but the peak moves to final residual
norm1/conv1: the ordinary complete decoder maximum is12,996,152,418 bytes,
only133,693,164 bytes lower overall. All25 words and per-frame original-FP32
gates pass unchanged. Typed cancellation, sink failure, source preservation,
same-VAE recovery, option switching and zero-allocation teardown pass.

Two real seven-request App workflows pass, with exact512/1024 image/video
exports and PNG alpha, matching the prior default too. Same-build single staged
1024/t25 disabled/enabled requests observe90.375/90.053 s and active maxima
13,148,195,662/13,014,502,498 bytes. No paired-speed, physical hard bound or
16-GiB-hardware claim is made. The App exposes the option and its tested
user-applied memory advice includes streaming video input/output, frame history,
compatible sharing and1-GiB cache; explicit rows/workspaces are preserved.

Independent signed `dist/seedvr2-upscale-history-20261015/TurboCider.app`
(macOS26.2+) passes signature/release guards and14 packaged-library contracts.
Its complete512/t25 packaged-library smoke executes25 shared frames and matches
the App's decoded RGB, with complete25/24-FPS geometry. Explicit relocation
manifest validation checks the actual library's runtime ID; the first missing-
manifest preflight failure remains preserved. No existing running App is changed.

Detailed records and the next actual-peak/virtual-halo ANE gate:
[phase/ownership evidence](../design/seedvr2-vae-phase-and-upscale-history.md).
The full optimization goal remains active; broader scheduling/workspaces,
useful VAE ANE integration, real low-memory hardware/admission and bounded
longer-context DiT are still incomplete. Current defaults/121-frame cap remain.

### Direct-frame public Core ML causal-conv experiment (2026-10-15)

The next actual-peak boundary now runs inside the complete VAE: only subsequent
`up_blocks.3.resnets.0.conv1` uses original three-tap fixed Core ML convolution.
GPU packs separate past/current owners and exact spatial halos into two bounded
surface slots; no full temporal concatenation. Initial GPU FP32 folding/global
norm/remaining layers/phases stay native. This is an isolated compile-guarded
prototype, not a public VAE backend or full-VAE conversion. Stable packaging
rejects both its flag and actual shader/guard bytes.

Independent signed-ABA FP32,120,434,688 pack-word/11,857,664 consumer-word
witnesses, ragged borders, real strided/offset inputs, retained outputs,
pre/active prediction cancellation, same-runtime/VAE recovery and zero-MLX-
allocation teardown pass. Complete real1024/t25 executes24 subsequent frames,
6144 predictions/backing hits and checks all78,643,200 pixels. The same-latent
audited all-FP32 VAE gate passes globally/per-frame, relative-L2 .000525150398;
the first raw frame matches native byte for byte.

Full-decoder ABBA has four processes, each one cold/two freshly recomputed warm
clips, all pixels checked each iteration. GPU/CPU+NE warm medians are
55.459255625/55.189463709 s, about0.49% apart, and active maxima
12,996,152,418/12,592,974,678 bytes. The403,177,740-byte reduction is the clearer
result; no substantial speed/full-request/App/hard-RAM/physical-ANE claim follows.
The first full cohort's host snapshot overflow (FP32 latent sized as half words)
and crash are preserved; corrected snapshots retain original dtype/byte extent.

Full qualification/records and remaining public-bank/full-request work:
[virtual causal boundary](../design/seedvr2-vae-virtual-causal-conv.md).
Overall VAE/long-video/low-memory objectives remain active and unchanged.

## Reproduce component checks

```sh
.venv/bin/python tools/models/download_seedvr2.py --verify-only
bash tools/native/build_seedvr2_probe.sh
.venv/bin/python tests/native/test_seedvr2_download.py
.venv/bin/python tests/native/test_seedvr2_geometry.py

# Requires a Python environment containing torch, safetensors, Pillow and
# NumPy. The oracle additionally uses an explicit pinned Comfy checkout and
# its dependency environment; it never contacts a running user service.
python tools/validation/seedvr2_oracle.py prepare --image SOURCE.png \
  --source-size 512 --size 1024 --frames 1 --output outputs/seedvr2-new-fixture
build/seedvr2/seedvr2-probe restore models/Comfy-Org-SeedVR2 \
  outputs/seedvr2-new-fixture/input.safetensors outputs/seedvr2-new-native
```

Metal/MPS execution requires host GPU access; a sandbox without Metal access
fails explicitly. The original source image used in this turn is the existing
`outputs/z-image-turbo-512-cli.png`, not a newly generated Z-Image benchmark.

## Implementation sequence

1. Verify downloaded weights and record source provenance.
2. Implement native MLX windowed multimodal DiT and causal video VAE.
   Validate matching inputs, block outputs, VAE and one-step sampling.
3. Wire image/video upscale operations, media import/export, sessions,
   CLI/SDK/library and App capability handling; add fail-closed contracts.
4. Run full image/video samples and quality checks, then paired ComfyUI
   benchmarks. Optimize the native GPU route against measured bottlenecks.
5. Export an identity-bound ANE FFN partition, validate it and benchmark
   GPU+ANE without weakening the pure-GPU acceptance requirement.

Model files, dependency checkouts, generated media and raw benchmark output
remain ignored. Never delete unrelated user artifacts to make disk space.
