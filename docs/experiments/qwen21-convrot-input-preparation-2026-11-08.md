# ConvRot input preparation: preserve the A8 packet

The default `Weights::project` ConvRot W8A8 route still uses
`convrot_rotate_dense` in `native/backends/mlx.cpp`: grouped H256 matmul,
rounded to the activation dtype, followed by the production dynamic-token
A8 kernel. The existing radix-4 Metal rotation is a different reduction
order. Shared-versus-register radix tests do not establish equivalence to
the default dense transform.

One proposed optimization was to combine register radix-4 H256 with A8 in
one threadgroup per complete row, removing the rotated intermediate and
its rereads. Before implementing that fusion, a component experiment
compared the current dense expression and existing register radix-4, each
consumed by the original `convrot_quantize.metal` under its production Safe
math policy. Complete INT8 codes and F32 token-scale bytes were compared.

## Actual numerical veto

Apple M5 Pro / macOS 26.5.1 / MLX 0.32.0, TF32 disabled. Runtime ID:
`tc-runtime-build-v1-abc3c50fc34c6da7af459f6e8fdfdb9c8b0d08accd7915f3f9a68b61d0019016`.
Source and loaded-library hashes matched the current production build.
Inputs were synthetic; the experiment did not read model payloads.

The matrix covered BF16/F16/F32; K=4096/12288; M=1/33/64/65/1024; and
normal, cancellation, extreme, zero, nonfinite and inverse-H tie patterns.
All 180 cases completed, with 360 actual production A8 Metal dispatches.
An independent CPU A8 oracle matched each actual packet byte for byte.

| Input dtype | Cases | Cases with different H bytes | Cases with different A8 packet |
| --- | ---: | ---: | ---: |
| BF16 | 60 | 28 | 10 |
| F16 | 60 | 38 | 2 |
| F32 | 60 | 40 | 19 |

Two examples explain why the existing radix implementation cannot replace
the default transform under the current exact-output contract:

- F16, K4096/M1024, ordinary input, row 430/column 132: dense H was
  `-0.2386474609375`, radix H was `-0.238525390625`. Both scales were
  `0.028066560626029968`, but the A8 codes were **-9 and -8**. The second
  F16 mismatch was also an ordinary-input case at K12288/M1024.
- BF16, K4096/M1, extreme input, row 0/column 2: dense H was finite
  (`8.473828473128839e37`, code 127), while radix H overflowed to NaN.
  The latter produced NaN scale and zero row codes. The unnormalized
  radix intermediates change overflow behavior before the final `1/16`
  factor. All ten BF16 extreme cases failed; BF16 normal cases passed in
  this prescribed matrix.

F32 normal inputs also had scale/code differences. No tolerance was
relaxed and no default rotation or quantization recipe was changed.
This veto concerns the existing radix replacement; a future fusion with
a different, independently qualified arithmetic design needs new evidence.
There is no speed claim from this correctness-only experiment.

The component process exited zero after recording the negative result.
Sampled peak physical footprint was 329,745,200 B, maximum sample gap
110.1 ms, and swap-in/out were both zero. Full counterexample rows were
preserved for all 106 cases with different H bytes. The first experimental
driver failed to compile because of an ARC pointer conversion, followed by
a clang frontend crash; that attempt and its original sources remain
preserved and performed no GPU work.

Local evidence is under
`outputs/qwen-convrot-gpu-fusion-20261107/h256-a8-numeric-veto/`.
The completed numerical attempt is `attempt-1794075166404951000/`, with
`report.json`, `analysis.json`, per-case records, full-row safetensors,
source snapshots and process-memory evidence. The original unsuccessful
compile is `attempt-1794075119290303000/`. Production files, model files
and formal tests were unchanged by this experiment.
