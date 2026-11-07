# Bounded read-ahead for source SHA-256 verification

Large-file `memory_sha256_fd` verification now overlaps one reader with
the sequential SHA-256 consumer. Two 1 MiB heap buffers bound the working
storage. The reader uses the original borrowed descriptor and `pread`,
preserves the caller's file position, and is joined on success, failure,
cancellation, and stop. The existing source-generation and content-identity
checks remain in place. Files smaller than 2 MiB use the serial path.

`TURBOCIDER_SOURCE_HASH_READ_AHEAD=0` selects the serial reference;
unset or `1` enables the new large-file path. Invalid values are rejected.
This changes source verification scheduling, not model arithmetic.

## Correctness and lifecycle checks

The normal source-lease test and its ThreadSanitizer run both passed,
with empty sanitizer stderr. Their production-source and test-source
hashes match the files being committed. Coverage includes:

- 36 portable SHA prefix comparisons across serial, read-ahead and default
  modes, including 1 MiB and 2 MiB boundaries.
- 11 range/control rejections and 12 pre-cancel/stop checks.
- Eight in-flight cancellation/stop checks with a bounded return and 16
  immediate descriptor reuse/unlink checks after return.
- The read-ahead path on a 256 KiB worker stack.

Memory-manifest tests and all 24 source-registration tests passed. The
new native build completed with runtime ID
`tc-runtime-build-v1-abc3c50fc34c6da7af459f6e8fdfdb9c8b0d08accd7915f3f9a68b61d0019016`.
All 683 recorded runtime source inputs were verified.

The Qwen suite ran 149 tests: 141 passed, six optional integrations were
skipped, and two tests were refused by the physical-memory reserve guard
before numerical execution. After measured memory recovery, both unchanged
targeted retries passed. The original failure log is retained. The real
eight-step DiT trajectories passed with shared A8 enabled and disabled,
including exact final latents and compact prefix storage.

## Measured verification stage

Apple M5 Pro, 24 GiB, macOS 26.5.1, MLX 0.32.0. The actual new native
library verified the original Turbo DiT and Qwen3-VL encoder INT8 ConvRot
files with `F_NOCACHE` on the held descriptors. Eight interleaved arms
provided two samples per mode and file. Canonical digests, descriptor
positions, source generations, and the loaded library were checked.

| Source | Serial median | Read-ahead median |
| --- | ---: | ---: |
| Turbo DiT | 3.794 s | 2.207 s |
| Qwen3-VL encoder | 4.682 s | 2.865 s |
| Sum of stage medians | 8.476 s | 5.072 s |

This is approximately 3.40 s less verification time in this small stage
experiment. It does not establish an end-to-end generation speedup.

## Complete image validation and timing limit

A fresh-registration candidate CLI request completed a 512×512 Turbo8
image with the original dual INT8 ConvRot weights, GPU W8A8, streaming,
zero optional weight-cache budget, typed stores and compact K/V enabled,
shared A8 disabled, and TF32 disabled. The complete PNG is byte-identical
to the existing baseline:
`4757fa2f794e82469095a90d68130ecd363baa4ed531aff6b8454e95370947e5`.
The native library hash matches the measured verification library.
The process-tree sampler recorded 153 samples, maximum gap 110 ms,
3,763,127,544 B peak physical footprint and zero swap I/O.

The preceding full-image timing comparison stopped in its first serial
arm: encoder and all eight denoise steps completed, but elevated system
memory pressure prevented VAE admission. No PNG was produced in that arm.
That failed comparison remains preserved; the separate successful
candidate establishes output correctness, not paired whole-image speed.
No memory guard was weakened. Actual GUI interaction and the latest
read-ahead build's desktop packaging are outside this validation.

Local evidence is under
`outputs/qwen-convrot-gpu-fusion-20261107/source-hash-read-ahead-tests/`
and `outputs/qwen-convrot-gpu-fusion-20261107/source-sha-overlap/`.
The latter contains `production/sha/report.json`, the failed
`production/cli/` comparison, both Qwen regression logs, and the successful
`candidate-image/report.json`. Raw logs, model weights and build products
are not committed.
