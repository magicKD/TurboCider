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

## Projection timing pilot

On the same M5 Pro, original extracted checkpoint matrices and two seeded
BF16 rotated-domain inputs were compared in compiled, synchronized Cider
calls. A is F32 GEMM output followed by BF16 cast; B is direct BF16 store.
H256, bias, checkpoint loading, encoder/DiT scheduling, and image output are
outside these timers. Each shape has ABBA and BAAB windows, five warmups
per arm/window and 20 samples per block. Complete output bytes are checked
before and after each window; formal windows contain no CPU output reads.

| Shape (M,N,K) | A median ms | B median ms | A/B |
| --- | ---: | ---: | ---: |
| encoder Q (64,4096,4096) | 0.5670 | 0.5573 | 1.017 |
| encoder Q (256,4096,4096) | 0.4855 | 0.4590 | 1.058 |
| DiT Q (1024,4096,4096) | 1.1764 | 1.1130 | 1.057 |
| DiT gate/up (1024,24576,4096) | 5.5401 | 4.9501 | 1.119 |
| DiT down (1024,4096,12288) | 3.0512 | 2.9951 | 1.019 |

The M64 ordering reversed between windows (1.042 versus 0.890); no stable
gain is established there. The other four shapes improved in both orders.
All 800 formal samples and 980 actual projection executions passed their
coverage checks, and all output byte comparisons passed. An initial
observer-affected attempt is preserved separately and excluded. These
component results do not establish an end-to-end speedup.

Raw evidence: `outputs/qwen-convrot-gpu-fusion-20261107/typed-store-micro/`.

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
