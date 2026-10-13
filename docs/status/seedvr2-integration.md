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

The earlier fixture generator enlarged before source downsampling; its saved
input tensors remain immutable valid parity fixtures. The generator now
downsamples the original directly and saves the actual source PNG/provenance.
Use fresh v2 fixtures for final real-media acceptance and benchmarks.

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
