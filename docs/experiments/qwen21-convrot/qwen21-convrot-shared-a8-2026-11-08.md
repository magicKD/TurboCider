# Shared ConvRot activation quantization: exact, not enabled by default

This candidate remains **off by default**. On the M5 Pro, an isolated DiT
QKV group improved about 5.4%, but the complete streamed DiT pilot did not
improve. A component win is not a reason to enable the route in the App.

## Implementation and boundaries

`TURBOCIDER_QWEN21_CONVROT_SHARED_A8=1` makes the selected groups share the
existing H256 rotation and one physical per-token A8 dispatch. Each GEMM
still consumes its own original I8 weight and F32 row coefficient arrays;
no weight concatenation, requantization, or ANE route is introduced.
Integer accumulation, both F32 scales, typed output rounding, online LoRA,
and bias ordering are preserved.

The primitive produces one flat allocation partitioned into independent
contiguous views. This avoids multi-output primitive sibling ownership.
Even if one view is requested, the opaque group executes all its GEMMs;
later views reuse the completed allocation. It is not demand pruning.

Encoder QKV and gate/up can use the candidate. DiT uses it only for decode
QKV: prefill remains independent so cached V prefixes cannot retain an
entire shared Q/K/V output. Legacy per-projection counters remain intact;
new physical quantization counters separately show the reduced dispatches.
M4 and older floating activation execution is unchanged.

## Correctness

Native package build and matching runtime identity passed:
`tc-runtime-build-v1-87c24e281e4698978f4c08ebf5b7119ff1c5b091b2ca0314bec1fa4946c59c92`.

- 204 shared-A8 full-byte comparisons passed on real Metal: three input
  and output dtypes, heterogeneous/odd widths, small/large M, strided
  sources, changed dynamic W/scales/inputs, partial evaluation without
  replay, nonfinite rows, and source-owner retirement. Invalid geometry,
  dtype, and flat-shape overflow are rejected before allocation.
- 71 typed-store comparisons, the independent integer CPU oracle, and
  raw/packed/W8A8 LoRA tests also passed. Shared Weights tests preserve
  online LoRA and the following typed bias, with one physical A8 dispatch.
- Qwen discovery covered 149 tests; six optional integrations were skipped.
  Initial failures reported physical-admission pressure/reserve, not
  numerical differences. Targeted retries passed all affected cases after
  recovery, including BF16/ConvRot streaming and encoder prefill.
- The original real 512px Turbo8 trajectory passed both modes, including
  cache eviction after step three, 253 drained refills and exact final
  latents. Projections remain 1536; physical A8 dispatches are 1536 versus
  1088. The expensive integration test accepts an explicit shared mode or
  tests both when the mode is unset, and retains real admission checks.
- Complete 36-layer encoder outputs for short/long prompts matched all
  BF16 bytes across both modes. Projection count remains 252; physical A8
  dispatches decrease from 252 to 144. Every request drained 36 two-slot
  refills with no optional resident prefix.

## Timing pilots

Compiled production Weights projections, including H256, used two seeded
BF16 inputs, five warmups, ABBA/BAAB windows and 20 samples per block.
Output validation ran outside formal windows. All 40 full-byte output
pairs and physical projection/quantization receipts passed.

| Group | Independent ms | Shared ms | Independent/shared |
| --- | ---: | ---: | ---: |
| encoder QKV M64 | 0.4461 | 0.4475 | 0.997 |
| encoder QKV M256 | 0.5928 | 0.5782 | 1.025 |
| encoder gate/up M64 | 0.8210 | 0.8202 | 1.001 |
| encoder gate/up M256 | 1.5475 | 1.5219 | 1.017 |
| DiT QKV M1024 | 3.0469 | 2.8897 | 1.054 |

Gate/up rankings reversed between windows. DiT QKV improved in both
orders. Encoder M64 window drift makes its combined median inconclusive.

The complete streamed encoder pilot used ABBA processes, each with one
warmup and two short/long pairs (38/186 actual input rows). Overall ratios
were 0.998 for short prompts and 1.012 for long prompts; the ordering
windows do not establish a stable encoder win.

The complete 512px Turbo8 DiT pilot used ABBA processes, each with one
warmup and two measured trajectories, zero cached prefix, identical owned
conditioning/noise, and full final-latent equality. Control median was
8.020 s versus 8.256 s shared (0.971×). Decode alone was 0.970×. All four
process memory receipts completed with zero swap I/O. Compression and IO
variation remain relevant; these data do not qualify a product speedup.

These phase timers exclude the other model stages and product Session
admission; they are not complete image-generation timings. A subsequent
typed-store-only DiT ABBA attempt completed three arms, but its last
control process hit physical memory admission; no closed ABBA conclusion
is claimed from that incomplete attempt.

Raw inputs, drivers, samples, source/library identities, memory traces,
compile failures and rejected attempts remain local under the ignored
`outputs/qwen-convrot-gpu-fusion-20261107/` directory, in `shared-a8-micro/`,
`shared-a8-encoder/`, `shared-a8-trajectory/`, and `typed-store-trajectory/`.
