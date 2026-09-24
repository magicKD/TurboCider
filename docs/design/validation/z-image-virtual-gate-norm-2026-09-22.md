# Exact virtual gate + FFN norm fusion: not promoted

Implemented opt-in two-reduction kernel in `metal/gate_norm_virtual.hpp`.
Preserves30 virtual SIMD partial sums per reduction, original lane sum order,
precise inverse RMS, intermediate BF16 boundaries, residual output and FFN
input output. Vector accesses,128/256/512 physical threads. A shared-memory
barrier separates reading the first reduction from overwriting its partials.
No weight quantization, no production-default change.

Standalone probe compares with two current-default norms: vector960 for
small shapes, virtual256 for BF16 rows4096/4128. Independent random gate and
scale weights/modulations avoid the old test's tied-parameter blind spot.
GPU test passed54 combinations: rows1/33/1024/1056/4096/4128, three thread
counts, BF16/FP16/FP32. Both outputs have max_abs0 against the control.

20 alternating-pair microbenchmark after4 excluded pairs,256 threads:

| BF16 rows | Two-kernel control ms | Fused ms | Max abs |
| ---: | ---: | ---: | ---: |
| 1056 | 2.98981 | 1.02954 | 0 |
| 4128 | 8.04008 | 8.95956 | 0 |

Only small shapes were integrated behind
`TURBOCIDER_Z_GATE_NORM_VIRTUAL_THREADS=256` (also128/512), rows<=1056.
Large shapes keep current dispatch. Values outside supported counts and
conflicts with disabled modulation/scalar gate fusion are rejected. Explicit
experimental setting can supersede other norm geometry settings; it is not
an automatic path. Benchmark parent flag is `--gate-norm-virtual-threads`.

## Whole-model result overrides the microbenchmark

Native-only build SHA256:
`349630000153d6b294712f325c1e90b48fb8f11f5cf9048352d1d0c03f84b818`.
Same-library ABBA,512x512,8 steps, greenhouse prompt,seed123,BF16,pureGPU.
Two warm requests and a separate parity request per fresh process; cold and
parity excluded. All actual loaded runtime fingerprints match.

- Control diffusion:14.673019,14.898943458,14.780051167,14.851989417 s.
- Candidate diffusion:14.890973625,14.932039583,14.917979458,14.876069542 s.
- Medians14.816020292 versus14.904476542 s; speedup0.9940651x (no gain).
- Measured request-wall medians19.962414062 versus20.202492833 s.

All four parity runs have identical11 tensor dump files and PNG. This rules
out a numerical regression for the tested request, not a performance regression.
The microbenchmark win did not translate to the model; no speculative cause
is asserted. Candidate retained as explicit experiment, not promoted.

Raw reports: `/private/tmp/z-image-virtual-gate256-512-abba-20260922`.
Session4732 completed exit0. Native-only build completed, no app packaging.
Seven CPU benchmark tests passed including candidate-env isolation and worker
flag rejection. GPU new-candidate test passed; the full existing GPU suite
was not rerun this turn. `git diff --check` passed.

Remaining goals:512<10s and historical same-machine regression are unresolved.
