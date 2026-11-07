# ConvRot A8 input caching: component gain, whole-request gain unproven

The production M5 GPU route keeps the existing BM32/BM128 integer
TensorOps tiles and original dense H256 transform. Two narrower candidates
were investigated on Apple M5 Pro, macOS 26.5.1 and MLX 0.32.0, with TF32
disabled. Neither candidate is enabled in the packaged App or main.

## Larger GEMM tile rejected

A raw quantize-to-typed-GEMM probe compared the actual native primitive,
the original MSL template forced to BM128, and the same template forced to
BM256. BN128/BK512/SK32/WM4/WN4, original checkpoint I8/F32 parameters,
integer dot arithmetic and scale/store semantics were unchanged. The
candidate increases the M fragment and accumulator storage per simdgroup.

All 639 three-arm typed-output checks and 213 packet/I32 checks passed.
The subsequent 5,760 synchronized-wall timing samples covered 72 ABBA/BAAB
windows; 288 retained timed-output owners and 288 separate pre/post
comparisons matched the actual native output bytes. No sampler or source
scan ran inside these component timing windows.

For BF16 M1024, BM128 to BM256 median latency changed from 1.1408 to
2.8578 ms for q, 6.2275 to 16.7961 ms for gate_up, and 3.0078 to
8.5645 ms for down. Both order windows regressed. The native/BM128
large-M controls were within approximately 0.983–1.018 per window.
Retain the existing tile selection. Increased register/fragment storage
is a hypothesis for the regression; no occupancy or spill counters were
collected to identify the hardware cause.

## Cached A8 input retains the original packet

The A8 kernel normally reads each rotated activation twice: once for the
whole-row maximum and once for quantization. A candidate retains each
thread's original inputs in fixed-index local arrays. It preserves the
original F32 divide, rounding-to-nearest-even, clamp to [-127,127], padding
and whole-row nonfinite policy. The dense H256 and GEMM were unchanged.

All 360 quantizer cases / 720 dispatches matched the original complete A8
packets and independent CPU oracle. A full quantize-to-GEMM component test
also retained exact output bytes. Formal component timing used 100
warmups and 20 repetitions per segment, with 5,760 samples over 72
windows. BF16 M1024 down latency fell 3.32% and 2.21% in the two orders;
smaller q/gate_up effects were closer to control drift. M64 regressed.

The isolated integration therefore selects only single BF16 down
projections with M>=1024, K12288 and N4096. It is default off under
`TURBOCIDER_QWEN21_CONVROT_CACHED_A8`. One full diagnostic request proved
256 cached dispatches / 262,912 rows, complete original text/noise/final
latent payload equality and identical final PNG. Its outer checker
originally failed on a public JSON field name after successful generation;
the retained CPU-only postverification completed the remaining gates.

Eight separate full CLI requests then completed in fixed ABBA+BAAB order,
without tensor dumps. Every PNG matched the baseline exactly. All arms
used the same actual native library, 252 encoder / 1536 DiT W8A8
projections, cached source-verification receipts and prefix0 weights.
The independent sampler observed zero swap-in/out traffic in each arm.

Pooled request medians were **9.1200 s off / 9.6654 s on**. Four-arm
window control/candidate ratios were **1.0127 / 1.0555**. These statistics
group the samples differently; neither favorable window ratio establishes
a stable gain. First/last-arm drift, nonzero and differing system
compression, and pressure-recovery waits constrain causal interpretation.
Keep the feature off and the prototype outside main.

The source-registration hit skips SHA verification, not the declared
uncached data-FD policy: both SourceLease capture and the pager's duplicate
set and check `F_NOCACHE`. Logical bytes divided by host read-span timings
are not measured physical SSD throughput. Read and wait spans overlap GPU
work and do not identify the cause of drift.

Local raw evidence remains under
`outputs/qwen-convrot-gpu-fusion-20261107/`: `w8a8-bm256-probe/attempt-002/`,
`a8-read-cache-probe/`, and `a8-down-full-image/`. The authoritative whole
campaign is `performance-attempt-1794080040978201000/report.json`, with
the independent `ANALYSIS.md` and `interpretation.json` beside it. Failed
compile/checker attempts are retained. These results do not change model
weights, quantization precision, default arithmetic or the packaged App.
