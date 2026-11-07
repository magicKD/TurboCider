# Streamed prefix K/V: remove retained target buffers

The streamed Transformer cached prefix views of full prefill K/V arrays.
Those views retained every target-image token for the remaining denoise
steps. Only conditioning prefix rows are needed after prefill.

The streaming retirement path now finishes value-preserving, independent
compact copies before retiring the current weight slot. It uses the same
same-dtype `astype(copy=true)` / contiguous copy mechanism as the existing
prefix snapshot exporter; a plain MLX `copy` can alias its source. No
normalization, quantization, projection, or sampling arithmetic is changed.
The copy is performed once per layer during prefill.

The default is enabled. `TURBOCIDER_QWEN21_STREAM_COMPACT_KV=0` selects the
old backing ownership for controlled comparisons. Shared A8 remains off
by default because its full DiT timing pilot did not improve.

## Real-model memory and correctness

Apple M5 Pro / 24 GiB / macOS 26.5.1 / MLX 0.32.0. Native build:
`tc-runtime-build-v1-5b5b6abde11261ecfa33658e7b9c2186e5523e7ef7d3d8e3869dd9aa79e5f642`.

The same library ran original Turbo INT8 ConvRot DiT weights with owned
512px conditioning/noise fixtures. Both ownership modes completed all
eight steps with exact pre-change final latents, 1536 W8A8 projections,
253 drained refills, and forced optional weight-cache eviction after step
three. The fixture contains 32 prefix rows, 1024 target rows, and 64 K/V
arrays across 32 layers.

| Live storage | Old views | Compact copies |
| --- | ---: | ---: |
| Logical prefix bytes | 16 MiB | 16 MiB |
| Actual unique prefix backing | 528 MiB | 16 MiB |
| MLX active after weight-cache eviction | 1,263,151,758 B | 726,280,846 B |

The live allocation reduction is exactly **512 MiB** in this workload.
Prefix backing stayed constant through all eight steps. The diagnostic
inspection counts unique evaluated buffer owners and performs no GPU work;
tests additionally require compact leaf layout and page-bounded backing.
This is a DiT residency reduction, not a claim that the entire App's peak
falls by 512 MiB: the original VAE can still determine the request peak.

Qwen discovery ran 149 tests: six optional integrations were skipped.
One encoder-prefill test initially reported a system-reserve admission
failure; its unchanged targeted retry passed. The affected streaming,
snapshot, lifecycle, Float-A/W8A8, typed-store and shared-A8 tests passed.
The full real trajectory also passed with shared A8 on and off.

## Complete image check and remaining limit

The new public CLI generated a short-prompt 512×512 Turbo8 image using
original encoder/DiT INT8 ConvRot weights, GPU, streaming, zero optional
weight-cache budget, typed stores on, shared A8 off and TF32 off. Its
complete PNG matches the previous cache-budget baseline byte for byte:
`4757fa2f794e82469095a90d68130ecd363baa4ed531aff6b8454e95370947e5`.
Encoder and DiT receipts contain 252 and 1536 actual projections; text,
initial noise and final latents are finite and saved as diagnostic tensors.
The selected library/source identity was verified, including dyld loading.

The batch's second, long-prompt request was rejected before any new model
phase by `system_memory_pressure`, with about 4425 MiB free and a 29 MiB
worker. It produced no long image. The batch sampler recorded about
3.80 GiB peak physical footprint and zero swap I/O; its verifier correctly
marks the whole batch unsuccessful because that second request failed.
No memory guard was disabled and no complete-batch or speedup claim is made.

Raw logs and both real-model ownership receipts are local under
`outputs/qwen-convrot-gpu-fusion-20261107/`. The complete-image failed batch
and successful short-image witness are under `kv-compact-full-image/run/`;
`postmortem.json` supplements rather than replaces its failed `report.json`.
