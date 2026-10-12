# Qwen Runtime FFN phase lifetime — 2026-10-02

This stage continues `1fef2ce` on the M4 Pro / 48 GiB laptop. It does not
change the App's default GPU route or claim a new ANE speedup.

## Implementation

The previous runtime FFN route required resident execution. Its 512² plan,
including the existing 4 GiB safety margin, required 48.5 GiB without a reference
and 51 GiB with one standard reference, exceeding this laptop's physical memory.
The staged plan already budgets the optional runtime separately; no estimate,
physical-memory check, 2 GiB optional cap or live-memory admission is reduced.

`TURBOCIDER_QWEN21_RUNTIME_STAGED_DIAGNOSTIC=1` admits only explicit Qwen runtime
FFN, component-staged, 512² requests with 0–3 standard 1024px references and no
prompt enhancement. Existing approximation/precision/cache restrictions remain.
The adapter case is limited to the pinned Viggle v0.2.1 r128, six steps, strength
1 and inference-time binding; native loading verifies its SHA-256 and 227
projections. Base requests use their existing schedule. Other runtime, frozen,
QKV and encoder routes are not widened. This is a CLI diagnostic, not an App
performance preset.

A staged request first retires any previous resident runtime owner before
encoding. After denoising it removes callbacks, joins pending work, copies the
metrics and fallback reason, then destroys the Core ML owner before VAE decode.
Preparation without sampling also retires that owner before returning. Exception
and cancellation cleanup retire it before clearing DiT weights. The existing
prepare operation still retains prepared GPU weights; releasing the optional
Core ML owner does not promise a completely unloaded engine.

`hybrid.runtime_weight.session_released` records actual wrapper destruction.
`counter_scope` states that runtime counters belong to that graph owner and
restart when it is reconstructed. Slot-byte and timing counters remain
historical. Private verified graph snapshots are distinct from the source graph
and Core ML/OS caches; owner release does not promise disk-free Core ML or
system-cache eviction.

Reconstruction also resets the adaptive scheduler and repeats model load and
self-test. Any performance comparison must include those costs. A staged
warmup does not preserve this runtime owner for the next request. Resident
benchmark tools that assume monotonically increasing owner counters cannot be
used unchanged for multi-request staged runs.

## Validation

Validation evidence is recorded under `outputs/runtime-staged-20261002/`.

The final complete native/Swift build passed with 427 source identities checked.
Final-library CPU checks passed: 13 CLI methods (including base 25/40-step and
r128 six-step admission with zero through three references) and 42 Qwen contract
methods. These plan checks do not replace real base/multi-reference inference.

A new small Metal test covers parameterized base FFN, captured `Weights` base
FFN, and real inference-time LoRA loaded from tiny local safetensors. It compares
two row counts against independent eager gate/up/output math, then repeats all
three graph constructions three times. All 18 numerical comparisons were exact
in this run; all nine destructions returned active memory to the initial zero
bytes without clearing the compiler cache. The initial sinusoidal test fixture
was replaced by deterministic integer-hash inputs because one shape cancelled
to a numerically trivial oracle; numerical and memory tolerances were not relaxed.

Run this regression without downloading a checkpoint:

```sh
TURBOCIDER_TEST_NATIVE_DIR=build/verify-runtime-staged-20261002-native \
  .venv/bin/python -B tests/native/test_qwen21_runtime_activation.py
```

### One cold GPU/Runtime pair

Both requests used Qwen2.1, the verified local Viggle v0.2.1 r128 at strength 1,
six steps, seed 42, 512×512 output, one standard 1024 reference, and prefix
snapshots disabled. The prompt changes a red ceramic teapot to cobalt blue.
Each route ran in a separate fresh process, sequentially. No models were downloaded.
This pair preceded the final activation ownership fix described below.

| Metric | GPU | Runtime FFN |
| --- | ---: | ---: |
| Complete request | 37.013 s | 40.795 s |
| Denoising | 26.550 s | 27.812 s |
| Runtime load/self-test setup | 0 | 2.249 s |
| First denoising step | 13.119 s | 13.887 s |
| Remaining steps | 2.675–2.708 s | 2.727–2.828 s |

Runtime took 10.22% longer; GPU/Runtime was 0.907×. There is no speedup claim
or new App preset. Runtime executed 159 Core ML predictions, 128 hybrid blocks
and 64 complete GPU probe blocks, without fallback or failure. Weight staging
was 2.985 s with only 0.000098 s of staging wait, so it overlapped other work;
these overlapping timers must not be added to request time. The six-step run
did not enter the untimed steady scheduling route. Rebuilding the staged owner
also discards that owner's learned scheduler state.

The compute plan preferred Neural Engine for all 432 nonconstant operations,
including 144 MatMuls. This is placement preference, not measured physical ANE
occupancy; the latter remains unknown. The diagnostic remains FP16, not W8A8.

The process sampler completed for both routes. GPU peak physical footprint was
21,141,156,032 bytes; Runtime was 22,168,744,848 bytes. Runtime's process also
includes prepare/cancel/recovery checks, so these are not matched generation-only
peak comparisons. Driver/system services are outside the process-tree accounting.

### Captured MLX weights and final-binary acceptance

The first Runtime request correctly destroyed its Core ML wrapper and private
graph snapshot before decoding, but still retained **604,946,986 MLX bytes** after
the request. The equivalent GPU request retained 967,178 bytes. The difference
was 603,979,808 bytes, or 576 MiB plus 32 bytes. This was separate from the
327,942,144-byte Runtime weight slot.

The compiled Runtime GPU FFN and LoRA FFN used a shared multi-output `mx::split`.
A small reproduction retained 32 KiB after graph destruction; replacing it with
independent single-output slices returned active bytes to zero. The production
activation helper now uses those slices, matching the ordinary Transformer's
boundary. No dtype, weight fusion, LoRA scale or schedule was changed.

One final-binary Runtime edit then passed with **967,178 active MLX bytes** after
completion, eliminating the extra 576 MiB. Its 160 Core ML calls completed with
no runtime failure or fallback, and no private snapshot remained after generation,
prepare, cancellation or recovery preparation. Cancellation was injected after
Runtime load, before the first denoising step; this campaign does not qualify
cancellation inside an in-flight FFN. All owned source-graph/export fixtures were
removed after the test. OS/Core ML caches are not claimed to be empty.

The final request took 38.527 s, with 27.674 s denoising and 0.150 s Runtime setup.
Setup was already much warmer at the OS level than in the original pair. This
single follow-up therefore demonstrates memory release and functionality, not
a 40.795→38.527 s speedup attributable to the slice change. No second GPU pair
was run, keeping the laptop workload to three full generations in this stage.

The final PNG versus the previous Runtime output had RGB mean absolute difference
0.0924/255, PSNR 58.34 dB, and maximum channel difference 4/255. Versus the GPU
output, the mean difference was 0.3680/255 and PSNR 50.31 dB (maximum outlier
53/255). Visual inspection preserved the teapot, edit, composition and background.
This is one output pair, not broad image-quality certification.

The final binary's source snapshot differs from the cold-pair binary only in
`pipeline.cpp` and the new `runtime_activation.hpp`; toolchain inputs match.
Final evidence is in `final-runtime/report.json`, `final-runtime/pixel-comparison.json`,
`runtime-activation-tests.log` and `final-validation.json`. An initial CPU invocation
accidentally selected the old default CLI; its failed log is preserved as
`cli-admission-wrong-binary.log`. The final 13-method pass explicitly selected the
final staged binary with `TURBOCIDER_TEST_CLI`.

## Delivery and remaining scope

The App Swift sources remain identical to the previously verified `d572abe`
stage; the native/Swift package is rebuilt here. Local delivery is a signed
`dist/release-<commit>/TurboCider.app` and `dist/TurboCider-macOS-arm64-<commit>.zip`,
with source, executable-section and archive-byte checks recorded in the matching
`dist/TurboCider-<commit>-delivery.json`.

The Mac remained locked at this acceptance attempt. The running/installed App
still belongs to `67563e8`; this package has **not** replaced it and has no new
real UI acceptance. Multi-image drag/drop and the final running-App update remain
pending an unlocked desktop. Existing App drafts and history were not changed.

Further Runtime performance work must retain memory bounds, distinguish reusable
scheduler metadata from graph/weight ownership, and compare against the normal
GPU route including its prefix-cache behavior. This stage does not qualify encoder
ANE, integer/Hadamard acceleration, base 25/40-step Runtime performance or a
new App speed mode.
