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
The GPU+ANE path is not implemented.

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
