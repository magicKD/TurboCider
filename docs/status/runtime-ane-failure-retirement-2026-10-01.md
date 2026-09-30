# Runtime ANE failure-resource retirement

The `c4caaa2` App/ZIP is the delivered package. This subsequent source change
does not replace it. Model inference, Core ML execution, UI checks and test
execution remain stopped at the user's request.

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

Only C++ syntax/type checking with warnings treated as errors was performed for
both wrappers and the FFN integration source. No test binary was linked or run,
and no App, service, GPU inference or Core ML model was started.

The existing FFN regression source now checks release following partial-tail
failure, async adapter failure, invalid affine metadata, low-memory admission
and resident memory pressure. It also covers an exception from GPU tail
recomputation after the ANE graph has been retired. These assertions remain
unexecuted. QKV failure injection and a graph with a valid interface but failing
capability self-test remain acceptance work, as do checking temporary snapshot
removal and actual process memory after teardown. No RSS reduction or runtime
speedup is claimed from source inspection.

## Scope still open

Runtime graph interfaces and slots remain FP16. Affine Q4/Q8 source weights are
converted into FP16 slots; this does not implement W8A8 arithmetic. Runtime FFN
splits token rows and stages weights ahead of GPU attention; the frozen-graph
FFN path uses intermediate-channel partitioning. Qwen text/image encoders are
not moved to runtime ANE by this change.

Compiled Core ML source graphs and verified private snapshots still use disk.
Their source artifacts and Core ML system caches are outside private-lease
cleanup, so zero disk activity/remnants is not established. Dynamic W8A8 /
Hadamard graph construction, encoder scheduling and sparse matched M4 Pro
performance/quality measurements remain part of the original goal.
