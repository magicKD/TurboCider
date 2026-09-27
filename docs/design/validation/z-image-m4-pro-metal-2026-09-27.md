# M4 Pro Metal kernel screening — 2026-09-27

Four local Z-Image Turbo generations tested the additional M4 Max Metal
kernels on Apple M4 Pro 48 GiB. All completed without errors and produced
byte-identical 512×512 PNGs. This is an exploratory result, with one warm
sample per variant, not a statistical performance qualification.

## Conditions

- Same `dev-verify` binary (`95b1760` source checkout) and local BF16 model.
- Pure GPU, resident loading, 512×512, 9 steps, seed 42; no ANE or LoRA.
- Same short English prompt, dynamic text, two sequential processes.
- Default process: first request, then one warm request; Metal process: same.
- Exactly four generated images including warmups; no separate parity run.
- No context-cache override, precision change, model download or rebuild.
- The pre-existing streaming qualification stayed stopped.

Baseline uses the current M4 Pro defaults, including already-enabled fused
QKV preparation and vector normalization. The candidate adds these switches:

```text
TURBOCIDER_Z_MPP_SWIGLU=1
TURBOCIDER_Z_MPP_PROJECTIONS=1
TURBOCIDER_Z_MPP_QKV_PREPARE=1
TURBOCIDER_Z_GATE_NORM_VIRTUAL_THREADS=128
```

The large-shape virtual normalization path was not tested at this resolution.

## Results

| Request | Default wall | Additional Metal wall | Default denoise | Additional Metal denoise |
| --- | ---: | ---: | ---: | ---: |
| First in process | 21.130 s | 22.807 s | 18.999 s | 19.313 s |
| Warm, same engine | 19.592 s | 19.001 s | 19.008 s | 18.408 s |

Warm request wall decreased by 3.02% (0.591 seconds), and denoising decreased
by 3.16%. The first Metal request was slower by 1.677 seconds; the first
requests also had different text-encoding times (0.414 / 1.263 seconds), so
that difference cannot be attributed solely to Metal compilation. Timings
cover generation requests, not process startup or engine construction.

The sequential order and one warm sample per variant do not rule out system
drift. This result supports that the extra kernels run correctly on this
local workload and suggests a modest warm benefit; it does not justify a
universal M4 Pro speed claim. Automatic device defaults remain unchanged.
No additional experiments were started after these four images.

The same PNG SHA-256 was observed for all four images:
`0376727a3ef96137b2010debf101b8dc67aaaba37cb0599683b9faef61717164`.
The candidate image was also inspected visually and showed the requested
fox in a snowy forest; this is a one-image visual inspection.

## Reproduction and evidence

Use the existing `tools/native/benchmark_z_image_metal.py` with a local model
path, the same library for both arms, and a fresh output directory:

```text
--size 512 --steps 9 --seed 42 --runs 1 --schedule baseline,fused
--control-default --mpp --mpp-projections --mpp-qkv-prepare
--gate-norm-virtual-threads 128
```

Full command, prompt, raw reports, PNGs and independent hash/shape checks are
under `outputs/integration-20260927/m4pro-metal-4-image-screen`.
The compact tracked [result](z-image-m4-pro-metal-2026-09-27.json) records
binary identity, environment switches, timings and original artifact hashes.
