# Performance and fidelity / 性能与保真度

TurboCider targets compute-intensive inference on unified-memory Apple silicon.
CPU orchestration overlaps supported GPU work with Core ML FFN partitions.
The goal is to use available hardware with low transfer overhead; not every
model or split benefits. A larger ANE partition can be slower.

Apple documents the hardware's [unified-memory architecture](https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/)
and Core ML's [flexible input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html).
Our timings below come from the repository's own experiments, not Apple's
marketing benchmarks.

## Published measurements

| Workload | GPU warm median | Hybrid warm median | Throughput | Wall-time reduction |
|---|---:|---:|---:|---:|
| FLUX 4B, M4 Max64 GB,512×512,4 steps |2.2663 s|1.6279 s|1.392×|28.2%|
| Z-Image, M4 Max64 GB,1024×1024,9 steps |36.3685 s|30.0014 s|1.212×|17.5%|
| Z-Image, M4 Pro48 GB,512×512,512 tokens,9 steps,enumerated |23.964 s|20.425 s|1.173×|14.8%|

Throughput gain is `GPU / hybrid - 1`; time reduction is `1 - hybrid / GPU`.
A 1.39× speedup is 39% more throughput, not 39% less time. These medians are
warm request walls from named workloads. Compilation, initial loading and
first-shape preparation are reported separately in the source records.

Sources:

- [FLUX / Z-Image on M4 Max,2026-09-07](status/optimization-validation-2026-09-07.md)
- [Flexible Z-Image on M4 Pro,2026-09-08](status/z-image-flexible-ane-2026-09-08.md)
- [Machine-readable flexible-input summary](design/validation/z-image-flexible-ane-2026-09-08.json)

The M4 Pro fresh-process recheck produced1.21×/1.24× at short/long lengths,
with one warm repetition each. The table uses the earlier repeated median
instead of selecting the fastest later observation. A busy desktop App run
is a functional check, not a controlled performance comparison.

## Quality

Hybrid FFN weights use per-channel INT8 with floating-point boundaries. GPU
and hybrid results are not bit-identical. On the recorded M4 Max examples:

| Model | Pixel cosine similarity | Pixel correlation | Other evidence |
|---|---:|---:|---|
| FLUX 4B |0.999840|0.999262|GPU versus hybrid PNG|
| Z-Image |0.999903|0.999231|MAE0.879/255|

These metrics support high fidelity on the tested examples; they are not a
perceptual study or a guarantee over every prompt. On the M4 Pro flexible
experiment, corresponding fixed and enumerated hybrid outputs were pixel
identical. That establishes shape-conversion parity, not GPU-to-INT8 parity.

## Reproduce responsibly

1. Record chip, memory, macOS build, MLX/Core ML versions and checkpoint hash.
2. Match prompt, encoded length, seed, size, steps, LoRA and residency.
3. Separate compilation, first load, prompt-cache misses and warmed requests.
4. Run GPU and hybrid sequentially, alternating their order; avoid unrelated
   GPU jobs. Keep individual samples and report median plus variation.
5. Compare output images/latents as well as timing. Report approximations.
6. Use full request wall for end-to-end claims; label block-only timings.

Tools include `tools/native/benchmark_comparison.py`,
`tools/native/benchmark_z_image_shapes.py`, and
`tools/coreml/benchmark_flexible.py`. Use each tool's `--help` for the exact
local model and artifact arguments. Large weights and raw run directories
are intentionally excluded from Git.

## Boundaries

GPU is the product default. Validated artifacts are required for explicit
hybrid execution. Public Core ML compute-unit selection and invocation counts
do not establish actual per-task ANE occupancy. Other families have different
performance and validation coverage; see the retained model status records.
We do not claim universal losslessness, a universal20–30% improvement, or SOTA
across all inference engines and models.
