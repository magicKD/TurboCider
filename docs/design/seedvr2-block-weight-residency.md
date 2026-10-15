# SeedVR2: serial verified block-weight residency experiment

## Why this is part of the VAE/low-memory objective

The [request high-water trace](seedvr2-vae-norm-pack-and-resource-advice.md)
places the dual-VAE route's request maximum in the last DiT-block/output-head
interval, before decoding. More VAE-local savings alone cannot lower that
maximum. This experiment addresses the remaining model residency so that a
future complete request can combine it with exact-context lower-memory VAE
schedules. It is not a substitute for those schedules, actual low-memory
hardware, useful public configuration or complete-context longer videos.

No public request field, App picker/default, memory admission estimate or121-
source-frame limit is changed. The prototype is compile-isolated behind
`TURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_EXPERIMENT`; release packaging rejects
both its claimed flags and actual executable marker bytes. Reader fault
injection is compiled only into the native test binary and is also release-
guarded. No private ANE API or quantized/derived checkpoint is introduced.

## Exact computation and ownership boundary

Use the original3B checkpoint, SHA
`98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660`.
Its6,784,202,176 payload bytes are not all retained as evaluated arrays.

1. A native verified `SourceLease` hashes the original held descriptor. Subsequent
   lazy readers duplicate that same descriptor, rather than reopening a pathname
   after hashing. Path/open-file generation checks surround reads and each drained
   block. First-use full-content verification is real IO, not a caller proof.
2. Public MLX safetensors parsing creates lazy arrays. Drop every unselected array
   before evaluation; retain all global tensors plus one current block. Metadata
   parsing alone must read zero payload, or the implementation fails closed.
3. Preserve the existing FP16 projection/FP32 residual/time/normalization/output
   recipe. Original RoPE records are F16 promoted to F32; positive/negative
   conditioning records are BF16 converted to F16. Those are the original loader's
   conversions, not new rounding/quantization choices.
4. Keep `blocks.0.attn.rope.rope.freqs` in globals: later window layouts still
   reference it after block0 retirement. Other block parameters retain their
   original keys, shapes, strides and coefficients. Wrong-group/duplicate keys
   fail before publishing anything.
5. Execute the existing complete-context `DiT::block` unchanged. Materialize its
   actual video/text outputs and drain GPU submissions before erasing current
   parameter owners. Preserve all windows, shared text aggregation, shifted
   indices, residuals and the final output projection; no temporal crop is used.
6. Retain only globals between calls. Cancellation/failure removes any adopted
   current group after draining; the same model can recover without stale block
   owners. Serial reads use4-MiB cancellation checkpoints and a bounded allocator
   cache. Optional prefetch/pools are not implemented or assumed faster.

The source uses `F_NOCACHE` on its owned descriptor, but this does not establish
disk-cold reads, remove preexisting OS page-cache residency or impose a physical
RAM budget. Source verification time/bytes, group IO and native forward time
have separate counters. Model-ready time excludes verification; report that
explicitly rather than presenting hashing or reading6.78 GB as free work.

## Critical IO-future failure and corrected guard

Pinned MLX0.32's `Load::eval_cpu` waits the IO future with `wait()`, not `get()`.
Throwing from a custom Reader alone therefore does not propagate its failure to
the adoption thread. The first actual mid-read cancellation test demonstrates
this: only5120 bytes were read, yet adoption returned normally with438,862,036
parameter bytes. This is a failed lifecycle gate, not a usable memory mode.

The corrected Reader latches the original `exception_ptr`. The owner explicitly
checks it immediately after evaluating each raw source array, before a cast or
GPU kernel can consume an incompletely initialized allocation. It checks again
after conversion and before atomically merging the complete group. Main-thread
cancellation checks also close late-read cancellation paths. The same latch
handles non-cancellation read failures; a cancellation flag alone is insufficient.
No MLX installation is patched, and errors/thresholds are not suppressed.

Primary implementation references (motivation/API semantics, not this model's
performance evidence):

- MLX v0.32.0 `mlx/io/safetensors.cpp`:
  https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/io/safetensors.cpp
- MLX v0.32.0 `mlx/backend/common/load.cpp`:
  https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/common/load.cpp

The corrected native lifecycle gate actually exercises:

- Pre-cancelled group/forward with no new payload reads.
- Cancellation after a genuine partial read, typed propagation and no partial
  parameter publication, followed by same-source recovery.
- A real dispatched Reader failure without cancellation, explicit owner rethrow
  and same-source recovery (test-only injection, not a simulated passed return).
- Cancellation after block0 has actually completed, parameter retirement and
  complete32-block same-model recovery; a second complete output is byte equal.
- Duplicate/wrong-group rejection, unchanged source noise, unverified lease
  rejection, generated private source mutation/path replacement rejection and
  fresh native verification recovery. Original model files are not modified.
- 66 completed blocks across the intentional cancel/recovery/rejection cases and
  zero MLX active allocations after all tensors leave scope.

## Coefficient, read-range and complete-forward gates

The native parity probe compares all637 tensors and6,784,203,520 compute-storage
bytes against the original whole loader. The difference from source payload
bytes is exactly the existing F16->F32 RoPE promotion. Every coefficient matches.
An independent Python header/range audit checks2227 actual positional reads:
each lies in the selected tensor/group, all33 global/block groups are covered,
and only the explicitly requested block31 recovery is read twice.

The full-forward driver retains the complete original noise/condition geometry,
does not decode/resize/re-encode media, and executes all32 blocks on every fresh
iteration. It compares every velocity and Euler-latent byte, including signed
zeros, rejects nonfinite/shape/dtype changes, and can compare the result with a
previous frozen complete native restoration. Every selected executable, input
and checkpoint is SHA-bound before/after the screen. The corrected v2 driver
requires the actual native lifecycle gate as well as coefficient/range gates.

Initial small128-square/t9 and real1024-square/t25 uncancelled observations pass
all velocity/latent words; the real observation also equals the earlier frozen
restoration. These v3-binary records predate the corrected error latch and are
not relabelled as final lifecycle qualification.

### Final corrected-source result (2026-10-15)

The v7 replay is terminal complete. Its v2 screen first requires the actual
native lifecycle gate and all637 coefficient/read-range checks, then freshly
runs the whole/streamed complete32-block forward twice each. All velocity and
Euler-latent bytes equal both the same-binary whole route and the preceding
frozen native restoration. Every route unloads to zero MLX active allocations.

Apple M4 Max /64 GiB, original1024-square25-frame fixture, FP32 residual state
and1-GiB allocator-cache retention:

| Complete DiT component observation | Whole weights | Serial block weights |
| --- | ---: | ---: |
| Cold forward seconds |14.788842125|15.478938416|
| Subsequent warm forward seconds |14.772146708|15.475340292|
| Forward MLX active maximum bytes |11,734,408,548|5,230,433,676|
| Ready parameter bytes |6,784,203,520|121,595,092|
| Highest retained parameter bytes |6,784,203,520|438,862,120|
| Sampled process footprint maximum, load/cold/warm portions |12,919,690,808|6,411,289,680|
| Source payload bytes per fresh forward |not instrumented by this Reader|6,662,607,126|

The observed active reduction is6,503,974,872 bytes (about55.43%). The one warm
observation is about4.76% slower, including serial group reads. This is **not** a
balanced ABBA speed qualification or a hard memory bound. The process sampler
runs at20 ms and may miss short-lived maxima; its load/cold/warm scope differs
from the forward active counter. Do not sum these overlapping metrics.

Each route's separate first-use source verification reads6,784,268,336 file bytes
and costs about2.20 s, with zero native verification-cache hits. That work is
excluded from the reported model-ready/forward times and must not be hidden in
an end-to-end claim. The whole route's zero custom-Reader counter does **not**
mean its initial whole-bank load performed no IO.

This does not establish a6.50-GB reduction of the complete request: VAE histories
and workspaces remain, and may become the new highest phase. No VAE decoder,
media export, App request or real16-GiB device is qualified by this DiT screen.

The current ordinary, experiment-disabled library was independently rebuilt in
`build/seedvr2-block-stream-ordinary-v2/` (library-only, no App replacement).
All125 SeedVR2 CPU/contracts plus9 release guards and8 build-identity guards
pass. Actual ordinary library identity/source freshness and release-byte/link
checks are verified separately. These are not a new App inference/GUI test.
Its runtime ID is
`tc-runtime-build-v1-1df4a6c7d29600d9846ce4e1cd5fa9ab7460ed07eddc0e4f7f7ab1f71961937b`.

## Records and reproduction

Generated artifacts remain ignored under `build/` and `outputs/`, not committed
models/media:

- `build/seedvr2-block-stream-probe-v1/`: failed C++ inclusion of an Objective-C
  sampler header; preserved, replaced by a plain read-only Mach sampler.
- `build/seedvr2-block-stream-probe-v2/`: failed ADL `equal` collision and mixed
  counter `auto` declaration; preserved and corrected in v3.
- `build/seedvr2-block-stream-probe-v{3,4,5}/`: initial uncancelled functional
  probes and failed cancellation gate; not release-qualified.
- `build/seedvr2-block-stream-probe-v6/`: built, but review caught incomplete
  Reader catch coverage before execution; not a passed gate.
- `build/seedvr2-block-stream-probe-v7/`: corrected error latch/adoption guards
  and explicit test-only fault-injection binary.
- `outputs/seedvr2-block-weight-parity-v1-20261015/`.
- `outputs/seedvr2-block-stream-{128-t9,real1024-t25}-v1-20261015/`:
  initial uncancelled observations, not final corrected-source qualification.
- `outputs/seedvr2-block-stream-lifecycle-v{1,2}-20261015/`: failed cancellation
  attempts, preserved separately from the corrected v3 receipt.
- `outputs/seedvr2-block-stream-lifecycle-v3-20261015/`: terminal corrected native
  source/cancellation/error/recovery/teardown gate; private fixture mutations only.
- `outputs/seedvr2-block-stream-real1024-t25-v2-20261015/`: final corrected v7
  screen, actual lifecycle/read-range gates and complete frozen-output equality.
- `outputs/seedvr2-block-stream-final-gates-v1-20261015.json`: read-only final
  receipt/index audit; no new GPU experiment or App deployment.

Build into a fresh `TURBOCIDER_BLOCK_PROBE_OUT` with
`bash tools/native/build_seedvr2_block_stream_probe.sh`. Run
`seedvr2_block_stream_screen.py --probe ... --model ... --input ... --output ...`.
An optional `--condition-fixture` supplies the original complete encoded latent,
and `--frozen-reference` requires exact original velocity/Euler bytes. Do not
select inherited diagnostic precision overrides. Repetitions freshly execute
all32 blocks; caches/weights may remain warm, outputs never replace computation.

Future integration must bind the same verified source in real session/pipeline
ownership, expose an explicit slower/lower-residency policy without automatic
fallback, exercise whole-request/App import/export/cancel/recovery, and measure
the combined VAE/model maximum. After DiT residency drops, unchanged decoder
history/workspace can become the limiting phase again. Current local space is
below the measured1024 decoder-history scratch requirement; no multi-GiB spill
is launched and no prior records are deleted to make space. Real <=16-GiB
hardware, broader media and complete-context model clips beyond121 remain open.
