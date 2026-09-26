# Streaming App integration on M4 Pro

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

Host resolver and native catalog/API tests cover text boundaries, stale selectors after prompt-length changes, missing/wrong capacity proofs and over-limit rejection. Python/native canonical compatibility and the existing Z-Image public adapter regression pass. The test catalog builder accepts `--text-minimum-rows`; the supplied prompt must be the actual upper bound. Capacity publication is deliberately blocked until independently verified range evidence is implemented and supplied. The production catalog remains empty; this is not a released memory guarantee.

The preceding exact 26-token build (`3770042`) completed P2 memory qualification: 40 successful requests, byte-identical pairs and candidate peak P95 9,071,025,106 bytes. Its P0 comparison also completed all 40 requests with identical pairs, but remained INCONCLUSIVE: wall median ratio 1.0055 with a 95% interval spanning 0.9533–1.0577. Both experiments ended with thermal state fair. Those preserved results do not qualify the new capacity implementation, whose native identity is different.
