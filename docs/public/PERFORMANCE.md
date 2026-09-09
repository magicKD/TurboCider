# Performance and fidelity

[Documentation](README.md) · [Benchmark evidence](BENCHMARKS.md)

## More speed from the same Mac

TurboCider overlaps GPU work with selected Core ML FFN partitions instead of
leaving the GPU to execute every supported branch alone. Shared output buffers,
compiled GPU graphs and resident sessions reduce repeated work and copying.

Selected recorded highlights include **1.39× GPU-to-hybrid speedup on M4 Max**
and **1.34× versus the recorded stock ComfyUI Z-Image workload**. An older M4 Pro
FLUX snapshot yields **1.64×**, derived across measurement phases as explained
below. These are dated workload results, not current-release or all-model claims.

## Selected warm request results

| Recorded workload | Baseline | TurboCider candidate | Speedup | Time saved |
|---|---:|---:|---:|---:|
| FLUX 4B · M4 Pro 48 GiB · 512² · 4 steps · September 5 historical snapshot | Native GPU 4.758 s | GPU + ANE 2.896 s | **1.64×** | **39.1%** |
| FLUX 4B · M4 Max 64 GB · 512² · 4 steps · September 7 | Native GPU 2.266 s | GPU + ANE 1.628 s | **1.39×** | **28.2%** |
| Z-Image Turbo · M4 Max 64 GB · 1024² · 9 steps · September 7 | Stock ComfyUI GPU 40.110 s | GPU + ANE 30.001 s | **1.34×** | **25.2%** |
| FLUX 4B · M4 Pro 48 GiB · 512² · 4 steps · September 5 | Original flux2-engine GPU 6.073 s | Native GPU 4.758 s | **1.28×** | **21.7%** |

All dates are in 2026. Speedup is baseline median divided by candidate median;
time saved is `1 - candidate / baseline`. A 1.39× ratio means about 28% less
time, not 39% less time. Rows are selected highlights, not an exhaustive model
ranking. We use recorded medians rather than the fastest individual samples.

**Read the comparison boundaries:**

- The **1.64×** value divides GPU and optimized-hybrid medians from two phases
  of the September 5 work. It is not a same-build paired GPU/hybrid rerun.
  The older hybrid artifact recorded path/size provenance, not a checkpoint
  SHA. It is historical evidence, not a currently certified artifact.
- The **1.34× ComfyUI comparison** uses different timing interfaces and seeds:
  native seed 42 versus ComfyUI seeds 43–45. Native has two warm samples per
  route; ComfyUI has three. It is a recorded workflow comparison, not a
  seed-identical paired framework ranking.
- GPU + ANE rows use quantized INT8 FFN partitions with floating-point
  boundaries. They are not equal-precision or bit-exact GPU-only comparisons.
- Warm request timing excludes initial process/model setup and offline
  compilation. Reusing conditioning benefits repeated prompts; changed prompts
  and first requests can take longer.
- These archived observations were not rerun for this documentation update.
  See [samples and conditions](BENCHMARKS.md) before quoting a number.

## How much comes from ANE?

For the recorded Z-Image workload, native GPU alone took **36.369 s**, versus
**40.110 s** for stock ComfyUI: **1.10×**. Adding the selected ANE partition
reduced the native request to **30.001 s**, another **1.21×** over native GPU.
The combined comparison is 1.34×. Do not multiply rounded factors as if they
were independent universal improvements.

On M4 Max FLUX, the same record's direct MLX GPU engine took 2.264 s versus
TurboCider GPU's 2.266 s: essentially equal, not a GPU-only framework win.
The standout in that workload is the hybrid route.

## Fidelity accompanies speed

| Recorded comparison | PNG cosine | PNG correlation |
|---|---:|---:|
| M4 Max FLUX GPU versus hybrid | 0.999840 | 0.999262 |
| M4 Max Z-Image GPU versus hybrid | 0.999903 | 0.999231 |

These values describe tested images, not every prompt. They do not establish
bitwise equality or replace a perceptual study. Separate shared-noise Z-Image
checks against ComfyUI reported PNG cosine 0.999770 and correlation 0.998665;
those were not the seed-varied timed samples.

## What is deliberately not a headline

Block-only FFN microbenchmarks can show larger multipliers, but omit text
encoding, attention, VAE and export; they are not whole-image or video results.
Private-API ANE experiments and removed external-runtime paths are not evidence
for the shipping public runtime. LTX hybrid quality remains gated, and the
recorded full Wan hybrid run was slower than GPU. We do not advertise universal
video acceleration, arbitrary-resolution gains or a universal SOTA ranking.

GPU remains the product default. Exact device, checkpoint, geometry and LoRA
compatibility govern optional hybrid routing. Core ML compute-unit preferences
and call counts do not prove actual per-task ANE occupancy.

## Reading and reproducing a result

The [public benchmark record](BENCHMARKS.md) includes the retained timing samples,
sample counts and known limitations. Check chip, memory, OS/toolchain,
checkpoint identity, prompt tokens, seed, shape, steps, precision and residency.

For any future comparison, separate load/compile time from warm wall time, avoid
other GPU jobs, alternate route order, preserve individual samples and compare
quality as well as speed. Use complete request wall for end-to-end claims.
No performance experiment is needed to read or validate these documents.
