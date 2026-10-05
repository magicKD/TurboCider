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
