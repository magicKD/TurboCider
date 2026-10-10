# Splash private ANE protocol and synchronization

`native/backends/private/ane_program.{hpp,mm}` adapts the private Objective-C
selectors, IOSurface representation and shared-event enqueue protocol from
Splash PR #260, `runtime/ane/Program.{hpp,mm}`, commit
`0ac3d5b170523774d8c85cd86493908a64c80eee` (incoai/splash).

Splash is distributed under the Apache License, Version 2.0; the complete
license is included in [Apache-2.0.txt](Apache-2.0.txt). The referenced source
files contain no separate copyright notice; upstream attribution is preserved
here without inventing one.

TurboCider modifications (2026-10-03): independent opaque API; checked dynamic
selectors; named bindings; SHA256 identity binding OS build/device/ABI/source;
cache ownership/process locking/content validation; retained asynchronous
resource ownership and bounded timeout quarantine; distribution build gate.
The native FP16 micrograph emitter and Executor adapter are TurboCider code.
No Splash model loader, inference engine or binaries are linked.

TurboCider modifications (2026-10-05): move-only prepared request bindings,
separate driver submission, CPU-pre-met calibration dependencies and one-shot
calibration batch cleanup. Normal inference keeps its GPU-produced dependency
signals. The calibration fixture's two shares and one-versus-four evaluations
follow the methodology in Splash `runtime/ops/AneFfn.cpp` at the same pinned
revision; its sampling/ownership implementation is TurboCider code, not a
port of Splash's model loader or Metal command-graph engine.

TurboCider additions (2026-10-07): the independent execution primitives in
`native/backends/metal_graph/backend.{hpp,mm}` are informed by Splash's ordered
dispatch/event graph, bounded command submission and asynchronous owner
retention. Design references pin PR #260 at
`80827e1c3593f7d10c266394e4d4381321b73ad0`; implementation review also read the
local Splash checkout at `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`.
These primitives are independently written TurboCider code with explicit
same-device imports, copied parameters, allocation admission hooks,
physical-completion tickets and failure quarantine. The dense BF16 FFN
kernels adapt TurboCider's existing MPP algorithms, rather than Splash's
Q4/GGUF kernels. No Splash model files, executable or linked runtime are
introduced. This initial FFN island does not yet implement a complete native
DiT or a new ANE command graph.
