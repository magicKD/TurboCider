# Streaming App integration on M4 Pro

The branch includes `feat/stream-dev` and supports the local Z-Image binding whose Qwen3 text encoder is a shared, sharded directory. Source leases cover the actual shard files and their index; retargeting a binding is rejected. No model download was used.

## Build and catalog identities

Worker transport retains the complete native build identity. Catalog compatibility uses a separate key that normalizes only test hooks and audit counters. All native source, compiler, SDK, MLX dependencies and other build flags remain part of that key. This allows measured calibration builds to correspond to a release build without distributing test APIs. It does not grant catalog publication or replace campaign/review evidence.

Packaging checks both build metadata and actual exported symbols before replacing an existing App. Test or audit builds cannot be packaged by the release script.

## App integration test

`PublicImageStreamingSmoke.swift` exercises native discovery, durable user selection, the real App job store and independent CLI workers. It runs success, cancellation during denoising and another success, verifies worker exit, checks persisted jobs after restart, and compares the two images byte for byte. Model and output locations are explicit arguments; user preferences are not changed.

An instrumented build may use `TURBOCIDER_TEST_BUNDLED_CATALOG` for this test. It requires test hooks at build time and retains the template confidence marker. The generated header fails compilation without test hooks. The normal catalog generator continues to require original verified campaign and review inputs. A test fixture is not a production catalog.

The real local App integration run passed on Apple M4 Pro 48 GiB: 512 × 512, 9 steps, seed 42, 16 GiB test target, P7/G1/K2/D0/Q1. Success took 25.78 and 25.83 seconds including a fresh worker, source verification and text encoding; these are smoke observations, not a performance comparison. Cancellation reached denoise step 1. Both successful PNG files were identical, all three workers exited and terminal jobs survived store reopening. Raw evidence is in `outputs/integration-20260927/stream-app-positive` in the integration checkout.

The production catalog is still empty. Available test-fixture targets do not establish public release readiness. Final integration and release qualification remain in progress.
