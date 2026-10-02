# Qwen Runtime graph preparation overlap — 2026-10-02

The request-scoped preparation path works and releases its resources, but this
stage does **not** qualify a faster App preset. GPU remains the default. The
new Runtime switch is an explicit CLI diagnostic and remains off by default.

## Implementation

`RuntimeGraph::prepare` validates and privately snapshots the graph, checks the
complete eventual memory estimate, loads Core ML and validates its FP16 feature
interface. Its movable `Prepared` owner does not allocate IOSurfaces, start a
prediction worker, call MLX or predict. Binding rechecks the complete estimate
against current memory and the new budget before creating the slot bank. The
old synchronous constructor uses the same preparation/binding implementation.
The loaded model is released before its private artifact lease.

For the existing explicit staged Qwen FFN diagnostic only,
`TURBOCIDER_QWEN21_RUNTIME_PREPARE_EARLY=1` starts this lightweight preparation
before reference/text encoding. The factory captures owned metadata, not MLX
tensors or event callbacks. The request thread joins before binding and before
capability self-test. Cancellation and encoder exceptions join and release the
completed result too. Core ML's synchronous model load cannot be interrupted;
cancellation can wait for that call. Invalid artifacts remain errors, while
memory/capability failures retain the established reported GPU fallback.

Existing staged admission, 512²/standard-reference restrictions, pinned Viggle
r128 checks, approximation opt-in and memory margins are unchanged. The switch
does not enable Runtime on an ordinary GPU or resident request.

`load_seconds` now exposes separate artifact validation/snapshot, model load
(including interface validation), and binding spans. Their sum is checked in
tests. `runtime_weight_prepare_before_join_seconds` measures work completed
before joining; it is not measured GPU occupancy or an end-to-end saving.

## Validation

Final source identity: 432 native inputs, all matching the final build;
`tc-runtime-build-v1-3dc70ac97727068cec3e3d2bb1dae9041f543886ced7e01a28c25765cd9799f0`.

55 test methods passed: 13 CLI admission/override, 38 Qwen contract, one host
async-ownership, one tiny Prepared lifecycle, and two existing small Core ML/Metal
regressions. The latter cover MatMul/SwiGLU/GELU, rebinding, output lifetime,
adapter corrections, memory fallback and cancellation. No extra model was
downloaded. The host test caught and fixed repeated-result access before the
final build. An earlier build was rejected by its source-identity guard after
source edits; only the subsequent frozen build was used for model acceptance.

The tiny 32×64×96 CPUOnly test checks unbound destruction, move assignment,
lowered bind budget, an actual loaded-model ABI mismatch, source removal after
preparation, same-lease binding, and bit-identical synchronous/prepared outputs.
All owned fixtures were removed. The initial sandbox attempt was unable to
allocate its IOSurface; the authorized normal-host rerun passed. Both logs are
retained, with the passing report at
`outputs/runtime-prepared-lifecycle-20261002-unsandboxed/report.json`.

## One serial real-model pair

Both fresh processes used Qwen2.1, local Viggle v0.2.1 r128, strength 1, six
steps, seed 42, 512² output, one standard 1024 reference, DiT cache off and
prefix snapshots disabled. The same teapot edit and graph were used. The
synchronous request ran first; OS/Core ML cache warmth was not controlled.

| Span | Synchronous | Early preparation |
| --- | ---: | ---: |
| Whole request | 39.762 s | 39.150 s |
| Text/vision encoding | 5.913 s | 5.649 s |
| Denoising | 28.378 s | 28.197 s |
| Exposed Runtime setup including self-test | 0.170 s | 0.127 s |
| Artifact verification/snapshot | 0.00162 s | 0.00200 s |
| Model loading and interface check | 0.05554 s | 0.02702 s |
| Slot binding | 0.00268 s | 0.00329 s |
| Self-test | 0.10949 s | 0.12315 s |
| Background preparation | — | 0.02909 s |
| Join wait | — | 0.000001917 s |

The graph preparation completed before the join. Whole time was 0.613 s lower,
but most of that difference is outside setup and model-loading warmth also
differs. This single pair does not establish a stable 1.56% acceleration or
Runtime superiority over GPU. No new GPU pair or base 25/40-step inference was
run in this stage. The route remains dynamic-weight FP16, not W8A8.

Both requests completed without Runtime fallback (160/158 Core ML predictions;
adaptive scheduling may make slightly different decisions). The outputs preserve
the blue teapot, composition and background. Decoded RGBA MAE is 0.09196/255,
PSNR 58.17 dB and maximum component difference 6/255. This is one image pair,
not a general quality guarantee.

After either generation MLX active memory was 967,178 bytes, preserving the
previous 576 MiB retention fix. Private graph leases were gone before VAE decode
and after generation, encoding-stage cancellation, recovery preparation, invalid
graph rejection and engine destruction. The cancellation returned code 2 in
0.034 s in this run; this does not bound cold-load cancellation latency.

All source/export/private fixtures (313,912 logical bytes at cleanup) were
removed. This checks TurboCider-owned files, not Core ML's system-managed caches
or external driver memory. Peak process memory for the early process also
includes its additional cancellation/recovery checks, so it is not a matched
generation-only memory comparison.

Raw requests, phase events, outputs, independent process sampling, pixel metrics,
test logs and cleanup receipts are in `outputs/runtime-early-prepare-20261002/`.

## Outpaint follow-up

One separate six-step GPU test used a fully opaque 512² white canvas containing
the centered 256² photograph, plus a fully opaque black/white mask reference.
Both input alphas were verified as 255. Inference completed in 28.960 s but the
output still had 74.708% near-transparent pixels around the center. It failed
the outpaint quality check. No production workflow change was made from this
pilot; the App's prompt-only expansion remains explicitly experimental.

Evidence: `outputs/qwen-outpaint-mask-20261002/verification-report.md`. A future
spatial/latent constraint must be evaluated separately; forcing alpha alone
would conceal the missing surrounding scene rather than fill it.

## Delivery scope

This stage changes the native diagnostic and its observability, not App UI or
default generation settings. The existing draft, Playground state and history
were backed up without modification. Desktop acceptance requires an unlocked
Mac; package and installation receipts state the actual delivery status.
