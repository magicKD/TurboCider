# Benchmark evidence

[Documentation](README.md) · [Performance highlights](PERFORMANCE.md)

This is a public, self-contained export of existing project measurements.
The summaries below preserve selected timing records and test conditions.
The original publication pass exported existing measurements. The September 16
M5 entry below summarizes new local measurements; its complete experiment records
remain in a Git-ignored local directory.

Weights, raw images and full execution logs are not bundled. Consequently this
is inspectable reported evidence, not a claim that every original experiment
can be reproduced from documentation alone. Record IDs identify dated retained
measurements; they are not live links to unpublished files.

## SeedVR2 on M4 Max — October 14, 2026

Published October 15 by exporting eight existing paired reports, with no new
inference runs. Hardware: Apple M4 Max, 64 GiB, macOS 26.6.2. Original
Comfy-Org/SeedVR2 3B FP16 DiT/VAE, revision
`8d30a39cd60f07cfff11306406f97021a50ff0eb`, no LoRA. One Euler/simple step,
CFG 1, no color correction; PNG or silent H.264 MP4 export.

| Workload | Native GPU, default pair s | Comfy default median s | Native GPU, SDPA pair s | Comfy SDPA median s |
| --- | ---: | ---: | ---: | ---: |
| Image 256→512 | 0.827034 | 3.985151 | 0.827852 | 3.977832 |
| Image 512→1024 | 3.506320 | 15.535424 | 3.503769 | 15.636062 |
| Video 256→512, 25 frames / 24 FPS | 20.585554 | 71.189599 | 20.598051 | 71.337080 |
| Video 512→1024, 25 frames / 24 FPS | 85.010773 | 295.438371 | 84.878555 | 298.901318 |

The two native columns belong to their respective default/SDPA paired records;
they are not interchangeable controls. Headlines select the faster Comfy
median per workload and **its own paired native median**: 4.81×, 4.43×, 3.46×,
3.48×. No later hybrid, tiling or block-stream sample is substituted.

Each comparison has four independent processes/servers in native/reference/
reference/native order, one cold plus three warm requests per process, seeds
42/43/44/45: six warm requests per route, 16 complete outputs per comparison.
New input filenames invalidate conditioning cache; native reports cache misses
and Comfy media/resize/encode/sample/decode/export nodes actually execute.
Model loader caching is allowed. Every output has checked dimensions; every
video preserves all 25 frames at 24 FPS. Timing includes import/resize, VAE,
sampling and export, excluding construction, server readiness and offline
compilation; cold samples are separately exported.

Native uses resident GPU `seedvr2-stable-fp16-fp32-v1`: FP16 projections,
attention and VAE, with FP32 residual/time/norm/output/Euler and a 4 GiB
allocator cache limit. ComfyUI 0.39.0 commit
`b0b743566f65daafc423b4fea8a2fbda94b3384a`, Python 3.13.12, PyTorch 2.12.1
MPS; both presets explicitly use `--fp16-vae`. Default uses sub-quadratic
attention / ordinary residency; SDPA adds `--use-pytorch-cross-attention`,
`--gpu-only` and `--fp16-unet`. Both logs report
`model weight dtype torch.float16, manual cast: torch.bfloat16`.
This is a configuration comparison, not equal-precision/bit-exact parity.

Images use unmodified Comfy sources. Video uses only the disclosed
[non-CUDA cache-transfer guard](../../tools/validation/seedvr2_comfy_mps_cache.patch)
in an isolated reference copy; it disables CUDA-event cache transfers on MPS.
Unmodified stock video crashes are not denominators. User services/source
were not changed. All speed gates pass, but all eight records still retain
`overall_qualification_passed=false`; independent FP32 quality gates are
described in the [SeedVR2 guide](SEEDVR2.md#validation-boundary).
No matched Comfy process-memory ranking, ANE occupancy, long-video or 16 GiB
qualification is implied.

The committed [sample export](../design/validation/seedvr2-comfy-benchmark-current-20261015.json)
contains every cold/warm sample, input/model SHA, per-record native library/build
SHA, and relative IDs/SHA-256 of paired reports, child reports and dtype logs.
Raw reports/media are retained locally, not distributed. The
[detailed benchmark](../benchmark/SEEDVR2_COMFYUI.md) lists cold controls,
absolute savings, record IDs and selection boundaries.

## M5 Pro — September 16, 2026

M5 Pro, 16-core GPU, 24 GiB, macOS 26.4.1, MLX 0.32.0. Medians include
PNG export and exclude each explicitly marked warmup; rows are separate experiments.

| Workload | Before → after | Warm samples per route |
| --- | --- | ---: |
| FLUX.2 Klein 4B, 512², 4 steps, resident | GPU 1.784 → hybrid 1.524 s (−14.6%) | 18 |
| Z-Image, 512², 9 steps, streaming, GPU suffix loading | Original hybrid 10.773 → 7.217 s (−33.0%); GPU control 10.747 s | 8 |
| Z-Image, 1024², 8 steps, streaming, GPU segment compilation | Original hybrid 34.666 → 33.476 s (−3.4%); GPU control 28.895 s | 4 |
| Z-Image, 512², 8 steps, streaming, GPU segment compilation | Original hybrid 7.338 → 7.119 s (−3.0%); GPU control 10.578 s | 4 |
| Z-Image, 512², 8 steps, streaming, correct row selection | Oversized 4128-row hybrid 12.387 → 1120-row hybrid 7.119 s (−42.5%) | 4 |

Z-Image uses Comfy BF16, no LoRA and a 10 GiB GPU sampling budget. The budget
excludes Core ML and is not a process memory cap; paging was observed. 1024²
hybrid remains slower than GPU in this run, with substantial control drift.
M5 Z-Image automatic execution remains GPU. Earlier resident and full-weight
streaming experiments without a hybrid benefit are also retained.
The new runtime optimizations are enabled only by the built-in M5 Pro 24 GiB
device policy; M4 and other unmeasured devices keep their existing paths.

[Implementation, quality checks, limitations and reproduction instructions](../design/m5-ane-adaptation.md).

## FLUX on M4 Pro — September 5, 2026

- Hardware: Apple M4 Pro, 48 GiB; macOS 26.6; MLX 0.32.0.
- Workload: official FLUX.2 Klein 4B, fox prompt, 512×512, 4 steps, seed 42,
  28 encoded text tokens.
- Method: original flux2-engine versus native TurboCider, sequential AB/BA
  within each route. Two processes per engine, seven requests per process;
  discard each first request, leaving 12 warm samples per engine/route.
- Timing includes generation and PNG export, but excludes the constructor.
  OS and Core ML disk caches were not flushed. No tensor dumps were enabled.
- Native GPU: **4.7576625625 s** median. Original GPU:
  **6.0727734581 s**. Their ratio is **1.2764×**.
- Optimized native hybrid: **2.8956900001 s**. Original hybrid:
  **2.8702483125 s**. Native hybrid was **0.886% slower**, not a win over the
  original hybrid engine.
- The highlighted **1.6430×** GPU-to-hybrid value is derived from the native
  GPU median in the earlier phase and native hybrid median after graph-cache
  optimization. It combines phases; no new same-build paired test is implied.
- GPU used BF16; hybrid used the same local 1088-row manifest with 20
  single-block per-channel INT8 MLP partitions and FP16 I/O.
- Legacy artifact identity used path/size, without a source SHA. No hardware
  ANE-residency trace was captured. Floating-point fusion differed from the
  original compiled GPU route; equal seeds do not establish pixel identity.
  Do not reuse the later M4 Max quality numbers as proof for this M4 Pro case.

## FLUX on M4 Max — September 7, 2026

- Hardware: Apple M4 Max, 64 GB unified memory.
- Workload: FLUX.2 Klein 4B, 512×512, 4 steps, seed 42.
- Five warm samples per route after one separately recorded first request.
  Native GPU median: **2.2663414580 s**; hybrid: **1.6279351670 s**.
  Ratio: **1.3922×**, or **28.2%** less request time.
- Hybrid assigned FFN channels [0, 6144) to Core ML and [6144, 9216) to GPU.
  Each request made 80 Core ML calls; recorded session output-copy bytes were 0.
- GPU/hybrid PNG cosine: **0.999840318**; correlation: **0.999262057**;
  MAE: **1.284796/255**.
- Direct MLX GPU median was **2.2642201250 s**. TurboCider GPU was approximately
  0.094% slower, within the observed small differences.
- A matched FLUX ComfyUI workflow was unavailable. No FLUX-versus-ComfyUI claim
  is supported by this record. Five samples do not establish long-tail latency.

## Z-Image on M4 Max — September 7, 2026

- Hardware: Apple M4 Max, 64 GB unified memory.
- Workload: Z-Image Turbo, 1024×1024, 9 steps, fox prompt.
- Native seed 42: two warm samples per route after the first request.
  GPU: **36.3684708120 s** median; hybrid: **30.0014044585 s**.
- Stock ComfyUI installation version recorded as **0.32.0**, custom nodes
  disabled, official embedded Z-Image workflow. Seeds 43–45:
  **40.122 / 40.090 / 40.110 s**, median **40.110 s**.
- ComfyUI timing used execution-start/execution-success timestamps; native
  timing used resident request wall including PNG export. The seed sequences
  and boundaries are not identical. Treat the **1.3369×** hybrid comparison as
  a historical workflow observation, not an equal-seed framework ranking.
- Native GPU versus ComfyUI: **1.1029×**. Native hybrid versus native GPU:
  **1.2122×**.
- Hybrid used a4096 FFN partitions, with 288 Core ML calls per request and
  0 output-copy bytes. Its warm samples were **29.995564 / 30.007245 s**.
- Hybrid/GPU PNG cosine: **0.999903302**; correlation: **0.999230800**.
  INT8 partitions change numerical precision.
- Separate shared-noise ComfyUI parity: PNG cosine **0.9997699**,
  correlation **0.9986646746**, MAE **1.2825/255**. Those quality tests are not
  the seed-varied timing run and do not establish bitwise identity.

## Selection policy

The README highlights favorable recorded workload medians, not the best single
request. Historical, cross-framework and same-runtime comparisons are labeled
separately. No multiplier is a promise for all models, devices or prompts.

Bigger block microbenchmark ratios were excluded from whole-generation claims.
Results from removed external-runtime paths and private ANE APIs were excluded
from shipping-runtime marketing. Video paths without qualified end-to-end
speed/quality evidence were not promoted. The retained samples are small and
do not support strong confidence intervals or a general market ranking.
