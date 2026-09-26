# Wider SwiGLU GEMM screen: no promotion

Independent probe only; production native library unchanged at SHA256
`9f3dd08f7243b4f1128cbd8f057a5eae6f30458bb5f6fda089405b1bbd02cedb`.
No GPU benchmark was running when the probe started. No quantization added.

Prior sweeps already covered BM16/24/32/48/64 and BN64/128/256, so this
screen extended BM to96/128 and BN to512. The new probe mode
`mpp_swiglu_wide` compares directly to the actual production32x256 SwiGLU
kernel, not to the slower unfused MLX epilogue. Each row below is a separate
alternating candidate/control measurement,12 measured pairs after3 excluded
pairs, rows1056, BF16. Values are synchronized operator wall milliseconds,
not full-model or isolated hardware GPU timestamps.

| SIMD groups | Tile | Candidate ms | Matched production ms | Max abs |
| ---: | --- | ---: | ---: | ---: |
| 4 | 96x128 | 9.55475 | 9.10331 | 0 |
| 4 | 96x256 | 20.7087 | 10.1605 | 0 |
| 4 | 128x128 | 14.5973 | 11.8034 | 0 |
| 4 | 128x256 | 67.8113 | 9.26769 | 0 |
| 8 | 96x128 | 10.2701 | 10.2028 | 0 |
| 8 | 96x256 | 9.73258 | 8.66183 | 0 |
| 8 | 96x512 | 27.1804 | 10.9129 | 0 |
| 8 | 128x128 | 14.0643 | 9.57073 | 0 |
| 8 | 128x256 | 16.4224 | 9.47462 | 0 |
| 8 | 128x512 | 65.7694 | 8.58515 | 0 |

4-group96x512 and128x512 produced nonfinite errors and are rejected.
The initial existing probe printed NaN and exited0 even with those failures;
that exit status was not validation. Fixed it to emit valid JSON with
`valid:false`, omit invalid timing, and return1 if any output is nonfinite.
A fresh2-pair smoke reproduced both rejected geometries and exited1 as
expected. Their initial timings are not usable performance evidence.

The generic32x256 source is not identical to production (guards and shape
wrappers differ), and must not be called a same-kernel self-control. Two
independent40-pair reruns using `mpp_swiglu_control 4` measured:

- Generic8.22660 ms versus production10.21780 ms.
- Generic9.13358 ms versus production8.99933 ms.

Both had zero max error, but opposite timing conclusions: no stable winner
and no integration is justified.8-group generic32x256 measured14.4701 versus
9.97479 ms and is also not selected. No whole-model test or speedup claim
was made for any of these candidates.

Probe build completed; `git diff --check` passed. Original production tile
remains32x256 with4 SIMD groups. This search did not advance512 toward10s;
it excludes an additional tuning region and fixes invalid-result reporting.
