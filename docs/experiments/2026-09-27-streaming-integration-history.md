# Streaming integration history on M4 Pro

Historical implementation and experiment notes. For the current status and incomplete work, see [the integration status](../status/streaming-app-integration-2026-09-27.md).

The branch includes `feat/stream-dev` and supports the local Z-Image binding whose Qwen3 text encoder is a shared, sharded directory. Source leases cover the actual shard files and their index; retargeting a binding is rejected. No model download was used.

## Build and catalog identities

Worker transport retains the complete native build identity. Catalog compatibility uses a separate key that normalizes only test hooks and audit counters. All native source, compiler, SDK, MLX dependencies and other build flags remain part of that key. This allows measured calibration builds to correspond to a release build without distributing test APIs. It does not grant catalog publication or replace campaign/review evidence.

Packaging checks both build metadata and actual exported symbols before replacing an existing App. Test or audit builds cannot be packaged by the release script.

## App integration test

`PublicImageStreamingSmoke.swift` exercises native discovery, durable user selection, the real App job store and independent CLI workers. It runs success, cancellation during denoising and another success, verifies worker exit, checks persisted jobs after restart, and compares the two images byte for byte. Model and output locations are explicit arguments; user preferences are not changed.

An instrumented build may use `TURBOCIDER_TEST_BUNDLED_CATALOG` for this test. It requires test hooks at build time and retains the template confidence marker. The generated header fails compilation without test hooks. The normal catalog generator continues to require original verified campaign and review inputs. A test fixture is not a production catalog.

The real local App integration run passed on Apple M4 Pro 48 GiB: 512 × 512, 9 steps, seed 42, 16 GiB test target, P7/G1/K2/D0/Q1. Success took 25.78 and 25.83 seconds including a fresh worker, source verification and text encoding; these are smoke observations, not a performance comparison. Cancellation reached denoise step 1. Both successful PNG files were identical, all three workers exited and terminal jobs survived store reopening. Raw evidence is in `outputs/integration-20260927/stream-app-positive` in the integration checkout.

The production catalog is still empty. Available test-fixture targets do not establish public release readiness. The dev-verify merge is now committed as `bcf8415`; native tests, App/library/API tests, and the merged real App worker smoke passed. The new native build produced the same PNG as the premerge reference. Public release qualification remains in progress.

## Existing drafts

A legacy Z-Image sampling budget is not a calibrated full-request memory target. Old drafts retain their experimental residency setting and their original sampling budget; they do not silently select a new public tier. The App labels these controls separately. A previously selected public tier that is no longer available offers an explicit resident-loading recovery action, preserving prompt, LoRA and acceleration settings.

## Final merge regression checks

The merged native implementation was compared with `dev-verify` (`95b1760`) on the same M4 Pro using 512 × 512, 9 steps and seed 42. Each comparison used four serial ABBA sessions with four images per session; the first image of each session was excluded from warm medians. No compilation or other test inference ran concurrently.

| Execution | dev-verify warm median | merged feat/stream warm median | Observed ratio |
| --- | ---: | ---: | ---: |
| GPU | 17.671 s | 17.768 s | 1.0055 |
| GPU + registered ANE partition | 15.440 s | 15.043 s | 0.9742 |

All 16 PNGs within each comparison were byte-identical. These small samples are regression observations, not confidence-qualified speedup claims. The GPU raw bundle is `outputs/integration-20260927/gpu-abba-final` (its summary explicitly identifies this comparison); the ANE bundle is `outputs/integration-20260927/stream-merged-ane-abba`. The earlier main/dev-verify GPU bundle was overwritten by a temporary runner's reused output path and was repeated in `outputs/integration-20260927/root-gpu-abba-final-v2`: main 18.646 s, dev-verify 18.067 s, ratio 0.9689, all 16 PNGs byte-identical. The old overwritten timing claim must not be reused.

## Release App migration check

The signed release App restored an old 8 GiB experimental-streaming draft with the public selector Off. A real GUI Generate action produced a 512 × 512 image in 22.91 seconds; the result reported seven resident blocks, two refill slots and 207 fills. Its PNG matched the manual/public reference above. Restoring an unavailable 16 GiB public selection showed the recovery button; clicking it enabled generation again, selected resident/Off and preserved prompt, seed and steps on disk. Evidence: `outputs/integration-20260927/stream-ui-legacy-migration/validation.json`.

A four-request native same-layout preflight also produced matching layouts and images, with zero steady framework allocation/thread creation. Its whole-request median was 5.3% slower (denoise 1.6% slower), so the verifier correctly returned FAIL for the configured performance gate. A subsequent warmed confirmation completed 40 measured requests: wall medians were 21.148 s versus 21.247 s (ratio 1.0047), and denoise ratio was 0.9998. Images, layouts and steady allocation/thread checks passed. The final campaign verdict remains INCONCLUSIVE because its environment record is partial. Neither run qualifies a release record. Raw evidence is in `stream-p1-warmed-confirmation` in the integration output directory.

Campaign request identity includes dynamic-text and GPU-compilation choices, and reads V2 approximation/LTX policies from the execution object used by the native parser. Requests with different computation policies cannot be accepted as a same-workload comparison.

## Fixed text encoder padding

Public Z-Image planning incorrectly used padded encoder rows as the denoiser caption length. With `dynamic_text=false`, a short prompt uses 512 encoder rows, but `encode_text` removes padding before the denoiser aligns the valid caption to 32 rows. This caused `streaming_actual_plan_mismatch: layout digest differs` after generation. Dynamic-text metadata also rounded actual encoder rows to the denoiser alignment.

The adapter now records actual encoder rows and derives the denoiser layout from aligned valid rows. Its revision is bumped so old records cannot match the corrected contract. Host regression tests cover short dynamic, short fixed and over-512-token inputs. A real fixed-text request reproduced the old error; after the fix, the public test-catalog path succeeded and its PNG was byte-identical to the same-plan manual reference. A second real request with 600 valid tokens also passed receipt validation and matched its manual reference byte for byte; the result reported all 600 tokens. Evidence is in `stream-fixed-text-regression` in the integration output directory. This validation uses local weights and does not publish a production record.

## Variable text capacity (validation in progress)

An exact token-shape record cannot serve ordinary free-form prompts. The Z-Image adapter now supports an explicit `z-image-dynamic-text-capacity-v1` policy in canonical record v4. The record carries a verified BF16 worker route, dynamic text, a token interval bounded by 1024 and the exact upper-bound workload/layout. Exact records retain their existing canonical encoding and matching behavior.

Each request retains its actual encoder rows. The adapter independently compiles the upper-bound layout with the same source lease and plan configuration, verifies it against the record, and compiles the actual caption layout. The immutable snapshot, resolution digest, execution authority, C API result and actual receipt all bind the actual request. A capacity record does not permit changes to resolution, model sources, tokenizer, device, sampling policy or other workload features. Fixed-text requests still require exact records.

Host resolver and native catalog/API tests cover text boundaries, stale selectors after prompt-length changes, missing/wrong capacity proofs and over-limit rejection. Python/native canonical compatibility and the existing Z-Image public adapter regression pass. The test catalog builder accepts `--text-minimum-rows`; the supplied prompt must be the actual upper bound. Capacity publication requires the additional range evidence described below. The production catalog remains empty; this is not a released memory guarantee.

The preceding exact 26-token build (`3770042`) completed P2 memory qualification: 40 successful requests, byte-identical pairs and candidate peak P95 9,071,025,106 bytes. Its P0 comparison also completed all 40 requests with identical pairs, but remained INCONCLUSIVE: wall median ratio 1.0055 with a 95% interval spanning 0.9533–1.0577. Both experiments ended with thermal state fair. Those preserved results do not qualify the new capacity implementation, whose native identity is different.

## Capacity release evidence

`run_streaming_text_capacity.py` measures the native minimum prompt length, caption/encoder boundaries and the 1024-token upper bound with local weights. Each case preserves public/manual requests, source verification, native execution receipts, PNGs and process-lifetime memory sampling. Its public producer records its PID and start time; the independent verifier rejects memory logs from another process or reused across boundaries. No existing measurements are promoted to the new native identity.

For calibrated capacity records, the acceptance bundle must include `text-capacity/range.json`, and its quality review must explicitly reference that manifest. `verify_streaming_text_capacity.py` re-reads all referenced artifacts and independently verifies raw memory logs, actual token workloads, source/runtime identity, matching image dimensions/content, completed execution and memory headroom. Missing boundaries, stale evidence and status-only reports are rejected. P0/P1/P2 and the other acceptance gates remain required.

The capacity calibration estimator is `tree-phys-footprint-p95-with-text-boundaries-v1`: calibrated bytes are the greater of the P2 candidate P95 and all observed boundary peaks. Its maximum sampling gap includes both bundles, and its evidence digest binds the P2 summary and original range manifest. Thus a short-input memory spike cannot disappear behind the maximum-length measurement. The runner, receipt producer and verifier sources participate in native build identity. These checks produce reviewable evidence; they do not create review approval or authorize a production catalog by themselves.

### Capacity estimator validation

The `acd024b` build completed P0/P1/P2 and all ten text boundaries (8–1024 tokens). Its 40-request memory campaign and 20 boundary generations passed independent verification; the maximum boundary footprint was 9,126,976,496 bytes. Real App selection/persistence and success/cancel/success, source mutation rejection, same-plan English/Chinese images, and lifecycle recovery also passed with a test catalog.

Preparing the release record exposed a builder inconsistency: calibration required the boundary estimator, but record shape validation rejected that estimator. Shape validation now accepts it for capacity records. Exact records retain their existing estimator, and release construction still requires the original verified boundary evidence. A regression exercises calibration, shape validation and bundled catalog rendering together, including rejection of an exact record using the capacity estimator and a capacity release omitting boundary evidence.

These `acd024b` observations remain evidence for that build. The validator correction participates in native build identity; its rebuilt artifacts need their own matching qualification. The production catalog remains empty until the complete release requirements are satisfied.

## Resident App performance measurements

The App reuses one model engine between normal requests. Keeping both baseline and candidate resident engines alive during an alternating benchmark doubled model residency on the local 48 GiB host: the two worker footprints totaled about 42.7 GiB before other applications and OS memory. That attempt was stopped after four measured requests; all timings and the abort explanation remain in `/private/tmp/tc-stream-qualification-cc93154-p0/operator-abort.json`. It is not a passing performance result. The earlier per-request engine campaign remains independently INCONCLUSIVE.

P0 now supports the explicit `exclusive_warmed_worker_per_sample` protocol. Each sample launches one worker, runs the declared warmups on its persistent engine, measures that same engine, and waits for process exit before launching the next sample. Warmup and shutdown are outside the measured request, and every measured sample receives its own recorded warmup. ABBA/BAAB ordering, paired counts and performance thresholds remain unchanged. The independent verifier checks session intervals, clean process exits, warmup ordering, worker identity and engine generation; overlapping workers or substituted cold-engine results cannot qualify. P1/P2/P3 retain their existing lifecycle contracts.

The `cc93154` text-capacity run completed all 20 local generations at 10 lengths (8 through 1024 tokens), with exact public/manual PNG parity and actual token counts. It did not qualify for release: the 513-token memory log recorded 595,443,712 bytes of system-wide swap-out. Process-specific attribution is unavailable. The complete original evidence and rejection are preserved in `stream-text-range-cc93154/verification-failure.json`; successful generation does not override the memory gate.
