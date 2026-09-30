# Resumed local acceptance — 2026-10-01

The user authorized testing again. This round uses the M4 Pro / 48 GiB laptop,
the existing local Qwen Image 2.1 checkpoint and Viggle v0.2.1 r128 Turbo LoRA.
No model or dependency was downloaded. Real inference and Core ML probes ran
serially; there was no production-size ANE experiment or sustained stress loop.

## Changes established by testing

- Fixed the local RPC parser rejecting a valid exact page integer such as
  `0e-999999999`. Only a failed Foundation decode may enter the restricted
  `jobs` envelope fallback. Duplicate keys, malformed numeric syntax, unknown
  fields, non-jobs actions, fractions, booleans and range violations remain
  rejected. No arbitrary request JSON is rewritten.
- Added inventory integration coverage to `make test-api`, covering helper
  identity/arguments, isolated settings, bounded output, deadlines and owned
  child cleanup. The inventory does not open model-weight paths.
- Replaced the isolated candidate's dense Hadamard reference multiplication
  with Sylvester butterflies. The host tests verify ordering, projection
  preservation, input ownership, zero rows and unrepresentable scales. This is
  a CPU reference-preparation improvement, not a production ANE backend.
- Executed the previously source-only FFN failure-retirement regression.
  Unusable graph resources are released before GPU fallback, while healthy
  cancellation/reuse behavior and the original GPU callback exception survive.

## Checks and evidence

| Area | Result | Evidence directory |
|---|---|---|
| Runtime host contracts | 7 tests passed | `tests/native/test_ane_runtime_host.py` |
| INT8/Hadamard preparation | 4 tests passed | `outputs/resumed-validation-20261001/int8-host.log` |
| Reference preparation | Built and passed | `build/verify-20261001-app/reference-run.log` |
| Canvas sizing | Built and passed, including state apply/persistence/locks | `build/verify-20261001-app/canvas-run.log` |
| Playground | Built and passed, including actual NSItemProvider callbacks, cancellation and stale-context cleanup | `build/verify-20261001-app/playground-run-unsandboxed.log` |
| API client / RPC / installations | 11 + 2 + 8 tests passed | `outputs/api-validation-20261001/report.json` |
| Native FFN / LoRA micrographs | Passed, test binary 3.50 s | `outputs/native-verify-20261001/focused-ffn-report.json` |
| Qwen GPU | Generation and single-image edit succeeded | `outputs/resumed-validation-20261001/qwen-gpu/report.json` |

The new Swift regression sources link the frozen c4caaa2 runtime; the App
product sources are unchanged since that build. Actual GPU checks use the
newly linked `build/verify-20261001-native` runtime, including failure retirement
and the RPC fix. Its source hashes and incremental-build audit are recorded in
`outputs/native-verify-20261001/build-evidence.json`. The earlier delivered
c4caaa2 package was not overwritten.

Sandbox-only failures were distinguished from product failures: Unix socket
binding, Core ML temporary-directory creation, Metal discovery and provider
callbacks required the normal host environment. The same focused tests passed
there. The original extreme-zero RPC rejection was a product failure and was
fixed before acceptance. No dependency installation was used to resolve these.

## Real Qwen smoke results

Two sequential requests, 512×512 output, six-step student schedule, seed 42,
LoRA strength 1, `inference_time`, `component_staged`, GPU only, DiT cache off:

| Request | Request wall | Denoising | MLX peak allocation |
|---|---:|---:|---:|
| Blue teapot generation | 25.13 s | 17.74 s | 16.60 GiB |
| Blue-to-red teapot edit | 42.93 s | 30.83 s | 19.99 GiB |

Both receipts report six actual steps and 227 LoRA-applied projections. The
PNGs decode at 512×512 and were visually inspected: generation is coherent;
editing changes the pot to red while also changing surface/background texture.
That establishes functional execution, not exact material/background preservation
or a general editing-quality acceptance. Process wall for both was 69.22 s.
MLX allocation is not total process/system memory. These are single smoke
requests, not a warmed matched speedup benchmark.

The edit's first denoising interval was 15.01 s, followed by 3.12–3.18 s.
The generation intervals were 3.33 s then 2.85–2.92 s. This smoke run does not
separate prefix-cache construction, lazy materialization and compiler cost;
it does not prove compilation alone causes the edit's first-step delay.

The edit used a 512×512 source file but the production default
`qwen21_reference_size=1024` still produced 4,096 reference tokens. App-side
file resizing and the native encoder's canonical reference resolution are
different controls. This round does not claim that file resizing alone reduces
Qwen's reference-token cost. Low-resolution conditioning remains a separate
quality/performance decision.

## INT8 candidate decision

The reproducible probe is `tools/validation/runtime_ane_int8_probe.py`:

```sh
.venv/bin/python -B tests/native/test_runtime_ane_int8_candidate.py
.venv/bin/python -B tools/validation/runtime_ane_int8_probe.py --output /tmp/tc-int8-small-new
# Optional bounded second geometry, no checkpoint:
.venv/bin/python -B tools/validation/runtime_ane_int8_probe.py --medium --output /tmp/tc-int8-medium-new
```

Converted MIL retains both dynamic operands, both INT8 Q/DQ pairs, and FP32
scale restoration. Zero/signed inputs, A/B/A weights and scales, group-64
rotation and a finite output whose FP16 intermediate would overflow were
checked. The explicit temporary package/compiled directories were removed.

| Geometry (rows / hidden / width) | FP16 total median | Q/DQ total median | MatMul preferred device |
|---|---:|---:|---|
| 32 / 64 / 96, final butterfly preparation | 0.140 ms | 0.144 ms | CPU for both |
| 128 / 512 / 768, earlier dense-H preparation | 2.409 ms | 2.374 ms | CPU for both |

Timing includes group-one normalization/preparation, prediction and in-graph
restoration, with four warmed ABBA samples per variant. It does not time
Hadamard rotation. This small diagnostic does not isolate other CPU activity.
The apparent 1.015× medium ratio is insufficient acceleration evidence.
The larger probe initially stopped on an arbitrary 0.3% FP16 relative-error
bound; the complete diagnostic preserves the measured error instead of calling
that an image-quality pass. Maximum relative L2 versus the original FP64
projection was about 0.335% for the medium FP16 control and 1.107% for Q/DQ.
The final small candidate was about 0.129% / 0.836%, respectively.

Core ML listed NE support for some projection/QDQ operations, but preferred
CPU for these graphs; FP32 restoration operations were CPU-only in the plan.
A plan is not a hardware trace or proof of integer arithmetic. Consequently,
the candidate stays separate from the App runtime. No W8A8 speedup, encoder
acceleration, or image-quality parity is claimed.

Raw MIL, metrics and numerical cases are under
`outputs/resumed-validation-20261001/int8-fwht-final/` and
`outputs/resumed-validation-20261001/int8-medium-complete/`.

## UI and remaining boundaries

An isolated copy of the c4caaa2 App used its own state/library/settings under
`outputs/resumed-validation-20261001/`. Live UI checks covered settings-panel
collapse/expand, Playground layout and template switching, missing-model
explanation, file-panel import, reference-size selection, original restore,
and template input isolation. The original running user App was not replaced.
The teapot was only an import/control fixture, not a person-identity test.

Mouse drag/drop dispatch, actual outfit/identity generation, full cancellation
and restart lifecycle, and broad visual-quality comparisons remain outside
this round. State/provider tests cover ordering and import cancellation but
do not substitute for those full UI workflows.

The FFN micrograph run left no new private runtime lease or explicit test graph.
This is not a zero-disk guarantee: source compiled graphs and Core ML system
caches are outside the private snapshot lifetime. QKV fault injection and
RSS-level teardown measurements remain unverified. Production-size dynamic
INT8 scheduling and a demonstrated end-to-end ANE speedup remain open research.
