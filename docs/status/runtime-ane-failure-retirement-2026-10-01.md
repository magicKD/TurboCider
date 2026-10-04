# Runtime ANE failure-resource retirement

The user reauthorized testing. Both FFN and QKV retirement paths have now been
executed with small synthetic Core ML/Metal fixtures on the M4 Pro laptop.
The delivered `0f8dfa7` App includes the retirement changes; the final QKV
regression adds tests and documentation without changing that App's binaries.

## Finding and change

`HybridFfn::degrade` and `HybridQkv::fail` previously disabled execution but
retained their RuntimeGraph after capability self-test, staging or prediction
failure. The model then ran GPU fallback while the unusable Core ML model,
worker, IOSurface weight slots, private artifact lease and scratch could remain
owned by the wrapper. QKV output-admission failures released the graph but could
retain previously allocated host output capacity.

Both terminal-failure paths now destroy the graph before releasing borrowed
tensor references or host scratch. RuntimeGraph teardown joins its worker and
releases Core ML references before removing the private snapshot. This order
also covers a still-pending stage during memory admission failure. The wrapper
clears pending state, its scheduler and chunk count after graph destruction.
Failure reasons/counters and historical allocation metrics remain available.
`retains_resources()` reports current wrapper ownership separately from those
historical metrics.

The FFN constructor returns after failed self-test and teardown, rather than
accessing the retired graph to construct a scheduler. A failed prediction still
recomputes the entire affected tail on GPU. If that GPU callback throws, the
exception cleanup only joins a graph that still exists and has pending work;
the original exception is retained. Successfully copied outputs keep their
existing independent ownership.

Cancellation and user callback exceptions do not permanently disable a healthy
graph. Its existing drain/reuse behavior remains. The Qwen/Z model layers still
reconsider an unavailable runtime on a subsequent request; this change does
not introduce a new retry or scheduling policy.

## Validation boundary

The FFN regression passed with 32 × 64 × 96 base/LoRA micrographs. It checks
release after partial-tail failure, async adapter failure, invalid affine
metadata, low-memory admission and injected resident memory pressure. The
original exception from GPU tail recomputation survives after graph retirement;
healthy cancellation still permits reuse. The test binary took 3.50 s and left
no new private lease or explicit test graph. Evidence:
`outputs/native-verify-20261001/focused-ffn-report.json`.

The QKV regression compiled with warnings as errors and passed against the
same native library shipped in `0f8dfa7`. It uses synthetic diagonal BF16 weights,
five input rows and two-row prediction chunks; no checkpoint is loaded:

- A finite BF16 weight that overflows FP16 staging retires all wrapper resources
  and its private lease before full-input GPU fallback.
- The first prediction chunk succeeds and the second overflows FP16 output.
  All four tail rows are recomputed on GPU, the result matches the complete
  synthetic GPU reference, and resources/lease are gone while the wrapper is
  still alive. A later request on that failed instance does not retry Core ML.
- Cancellation after launch drains the worker, keeps the healthy lease's path
  and inode, and permits the next request to reuse the graph. Destruction removes
  the lease.

QKV compile/export/test took 1.05 / 1.16 / 2.40 s (4.61 s total). No private lease
remained before the runner deleted its owned fixture directory; that directory
was removed, and source/library hashes were unchanged. Evidence:
`outputs/runtime-ane-qkv-focused-20261001/report.json`.

The focused command is:

```sh
.venv/bin/python tools/validation/runtime_ane_qkv_regression.py \
  --native-dir build/verify-20261001-native \
  --output outputs/runtime-ane-qkv-focused-new
```

These are ownership, fallback and reuse checks. A graph with a valid interface
but deliberately failing capability self-test, physical ANE placement and
RSS-level teardown measurement were not separately exercised. No RSS reduction
or inference speedup is claimed from these tests.

## Scope still open

Runtime graph interfaces and slots remain FP16. Affine Q4/Q8 source weights are
converted into FP16 slots; this does not implement W8A8 arithmetic. Runtime FFN
splits token rows and stages weights ahead of GPU attention; the frozen-graph
FFN path uses intermediate-channel partitioning. Qwen text/image encoders are
not moved to runtime ANE by this change.

Compiled Core ML source graphs and verified private snapshots still use disk.
Their source artifacts and Core ML system caches are outside private-lease
cleanup, so zero disk activity/remnants is not established. Dynamic W8A8 /
Hadamard projection experiments were run separately, without establishing a
production speedup. The [laptop decision](2026-10-01-runtime-ane-decision.md)
records the measured candidate and the encoder/GPU A8W8 research boundaries.
