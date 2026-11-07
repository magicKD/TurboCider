# ConvRot W8A8 typed projection stores

Validated on Apple M5 Pro, 24 GiB, macOS 26.5.1, MLX 0.32.0.

## Change

Ordinary ConvRot W8A8 projections previously wrote an F32 GEMM result and
then cast it to the activation dtype. The GEMM now performs that final
BF16/F16 cast directly in its store epilogue. Original signed checkpoint
codes, dynamic activation quantization, I32 accumulation, and both F32
scale multiplications retain their existing order. Bias and LoRA still
follow the original output rounding boundary. Partial/split consumers
and the existing `cider::convrot_linear` API retain F32 output.

This removes an F32 intermediate and its subsequent read for ordinary
typed projections (8 × M × N bytes of intermediate traffic). It does not
change hardware eligibility: M5 and later use the existing supported
TensorOps route; older devices retain the existing floating activation
route. No ANE route or streaming memory guard is changed.

`TURBOCIDER_QWEN21_CONVROT_TYPED_OUTPUT=0` selects the original F32 store
plus cast for a same-binary comparison. The default and explicit `1`
select typed stores. Other values are rejected when projecting.

## Validation

- The complete native package build passed. Runtime identity:
  `tc-runtime-build-v1-8a72434d7cce0f7322e901856caa2b687587286feb73714b4fa39dadb0977464`.
- Real Metal tests passed 71 typed-output cases, comparing all logical
  output bytes against the original F32 projection followed by its cast.
  These cover BF16/F16/F32 inputs and outputs, small/large GEMM selection,
  production channel dimensions, tails, strided operands, compiled mixed
  dtype graphs, changed inputs, signed zero, nonfinite inputs, and the
  bias rounding boundary. The existing CPU integer oracle, shared weight,
  and raw/packed/W8A8 LoRA checks also passed.
- Qwen regression discovery ran 149 tests: 143 passed after retrying
  three physical-admission failures; six optional integration tests were
  skipped. The three initial failures explicitly reported system reserve
  or elevated system pressure. All passed unchanged after available
  memory recovered, including BF16 streaming, ConvRot Float-A/W8A8
  streamed/resident equality, and encoder prefill.
- The real 512×512 Turbo8 DiT trajectory matched the pre-change final
  latent exactly: 1,536 W8A8 projections, 253 drained refills, and forced
  cache eviction after step three.

The following command reproduces the core Metal test after building the
matching library:

```sh
TURBOCIDER_TEST_NATIVE_DIR="$PWD/build/qwen-gpu-fusion" \
TURBOCIDER_NATIVE_LIBRARY_DIR="$PWD/build/qwen-gpu-fusion" \
TURBOCIDER_TEST_GPU=1 MLX_ENABLE_TF32=0 \
  .venv/bin/python -B tests/native/test_qwen21_convrot_w8a8.py
```

## Limits

An additional complete 512×512 image comparison could not finish on this
host: the **original F32-store control**, with streaming and zero optional
weight cache, hit `system_memory_pressure` after the first denoise step.
Reusing verified source registration on retry produced the same decline.
No guard was bypassed. This is not an end-to-end PNG equality receipt or a
measurement of the candidate's speed. End-to-end acceleration remains to
be measured under stable host memory conditions.

Raw build/test logs, failed image attempts, diagnostic inputs, and model
receipts remain local under the ignored directory
`outputs/qwen-convrot-gpu-fusion-20261107/`.
