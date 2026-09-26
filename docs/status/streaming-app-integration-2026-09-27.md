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
