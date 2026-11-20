# Qwen Image 2.1 ConvRot streaming experiments

These records document the November 7–8, 2026 experiments on Qwen Image 2.1
ConvRot. Each report specifies the tested source, hardware, workload and
validation boundary. Treat the App defaults and candidate switches as dated
observations.

## Request lifecycle and memory

| Record | Scope |
| --- | --- |
| [M5 streaming and Runtime ANE](2026-11-07-m5-streaming.md) | Token-row embedding reads, exact generation, and the ANE timing boundary |
| [Weight-cache budgets](2026-11-07-m5-weight-cache.md) | Fixed buffers, optional DiT retention, live eviction and measured memory/speed tradeoffs |
| [Memory pressure study](qwen21-convrot-memory-stability-2026-11-08.md) | Historical App memory warning policy and the remaining validation limits |
| [Immediate generation](qwen21-convrot-immediate-generation-2026-11-08.md) | Removal of the later pre-wait and handling of optional cache allocations |
| [Source verification read-ahead](qwen21-source-hash-read-ahead-2026-11-08.md) | Bounded SHA-256 reader, correctness and same-source warm timing |
| [Prefix K/V compaction](qwen21-streamed-prefix-compaction-2026-11-08.md) | Independent retained prefix values and real-model correctness |
| [Time-condition preparation](qwen21-timestep-condition-cache-2026-11-08.md) | BF16 arithmetic boundaries, retirement of time weights and CLI checks |
| [Early suffix prefetch](qwen21-early-prefetch-time-cache-2026-11-08.md) | Optional read overlap, exact joint trajectory and stopped CLI gate |
| [VAE decode residency](qwen21-vae-decode-residency-2026-11-08.md) | Decode-only candidate, complete image correctness and allocator observations |

## Projection and activation candidates

| Record | Scope |
| --- | --- |
| [Typed projection output](qwen21-convrot-typed-output-2026-11-07.md) | Kernel output stores, numerical checks and projection timing |
| [Shared activation quantization](qwen21-convrot-shared-a8-2026-11-08.md) | Exact QKV-group candidate and the full streamed DiT timing limit |
| [Input preparation](qwen21-convrot-input-preparation-2026-11-08.md) | Original A8 packet, rotation ordering and numerical veto |
| [A8 input caching](qwen21-convrot-a8-cache-2026-11-08.md) | Rejected larger tile and packet caching component measurements |
| [Paired gate/up fusion](qwen21-convrot-gu-fusion-2026-11-08.md) | Epilogue controls, numerical qualification and complete-request timing gate |

Use the recorded commands from the repository root unless the report specifies
another working directory. The two original tracked JSON receipts accompany
their November 7 reports. Other reports reference their historical local
outputs and logs. The general [experiment guide](../README.md) covers adding a
new result.
