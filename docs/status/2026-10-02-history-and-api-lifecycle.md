# Reference history and App API lifecycle — 2026-10-02

This continues `3c03472` on the M4 Pro / 48 GiB laptop. The stage concerns App
history and local-service state. No additional model experiment, checkpoint or
dependency download was performed; the preceding numerical runtime acceptance
remains separate.

## Reference preparation survives history reuse

Previously, a 2048×1024 reference prepared to fit 512×512 became a 512×256
input. After generation and history reuse, only that submitted input path was
restored. The original file remained on disk, but its relationship to the
derivative and the preparation setting was lost. Consequently “restore original”
and original-size canvas matching treated the derivative as the original.

New App jobs carry an optional `inputAssets` snapshot, including the original
managed copy and preparation choice. Creation and Playground pass the frozen
submission draft's active assets. The job store captures this value before its
first suspension and matches the actual request's input count, order, paths and
image kinds, along with unique asset IDs and valid dimensions/absolute paths.
History reuse validates again before restoring the whole group. A mismatched
snapshot is ignored as a group; metadata from one image is never borrowed for
another by position alone. The native request is unchanged.

Successful imports and prepared derivatives already use distinct immutable
paths. Removing/reordering a draft reference or starting a new draft removes
bindings only; failed import/preparation cleanup deletes only unpublished new
files. Therefore history can retain those files without duplicating every image.
Older histories cannot reconstruct metadata that was never saved; they remain
readable using the submitted files as their original baseline.

## Local API lifecycle and queued cancellation

An asynchronous status response from a stopped service can arrive after a new
service starts. Without lifecycle identity checks, its success or failure can
overwrite the current service's visible state. The old stop polling task can
also clear the new start operation's busy flag. Lifecycle completion must be
owned by the process/generation that initiated it.

Separately, cancelling a queued service job left its ID in the pending queue.
While another job ran, cancelled entries still consumed the queue's 32 slots;
shutdown could also rewrite their terminal `cancelled` state as `interrupted`.
Cancellation now removes the queued ID after persisting the cancelled state.
Shutdown only marks still-queued entries as interrupted. Active-job cancellation
retains its existing cancellation path.

The App controller now assigns each lifecycle a generation and checks both that
generation and the owned `Process` after awaited work. Start completion/error/
cleanup and refresh success/failure cannot mutate a later lifecycle. Stop
invalidates outstanding work and leaves final cleanup to the identity-bound
termination handler; its redundant asynchronous stop polling task is removed.
The startup execution reservation remains held until its child exits.

## Acceptance and delivery

The final complete native/Swift package build passed. Three Swift suites passed:

- Reference preparation covers resize → JSON history → reuse → exact original
  bytes/dimensions, original-size canvas matching, undo, draft removal/reordering,
  mismatched metadata, and histories without the new field. Existing orientation,
  alpha, batch rollback and annotation-baseline cases also passed.
- History management exercises the actual submission/persistence entry with
  missing-model failure and cancellation before preflight, confirming that both
  terminal histories retain the snapshot and input/original files. Its existing
  fake-worker publication, recovery, event and output-integrity tests passed.
- App API tests use the real owned service and pause responses after receipt.
  An old success, old transport failure and old startup continuation cannot
  overwrite a replacement service or clear its startup execution reservation.
  Normal start/stop/restart, duplicate-socket isolation and launch failure passed.

The CPU service suite passed four methods, including its new controlled held
worker: fill all 32 queue slots, cancel them, successfully submit a replacement,
then stop/restart. Cancelled jobs stay cancelled, the remaining queued job becomes
interrupted, only the active fixture executes, and all created fixture engines
are freed. Four production RPC validation methods and one offline CLI discovery
method also passed. These tests load no model weights.

The reference-history test initially used an empty model catalog with the public
import action, which correctly refused image input. Its fixture now uses the
same real importer as the existing suite to prepare two retained inputs. All
assertions remain; only that test was recompiled/retried. The first failed log
is preserved as `reference-preparation-empty-catalog-fixture.log`; production
sources were unchanged after the final package build.

`outputs/app-history-api-20261002/final-validation.json` records all final logs,
427 matching native inputs, 52 matching Swift source files, frozen test-source
hashes, and unchanged user draft/Playground/history fingerprints. Compared with
`3c03472`, the native source manifest differs only in `service.mm`; toolchain and
model numerical sources match. No new image-quality or performance result is
attributed to this stage.

The commit is packaged as `dist/release-<commit>/TurboCider.app` and
`dist/TurboCider-macOS-arm64-<commit>.zip`; the matching delivery JSON records the
local ad-hoc signature, executable-section and archive-byte checks. The actual
desktop remains locked and the running App remains `67563e8`. This package has
not replaced it. Mouse multi-file drag, the new history restoration UI and the
App API toggle still require final acceptance after an unlocked desktop is
available. Source/state tests do not substitute for that interaction check.

## Runtime follow-up decision

No further model benchmark is added in this stage. Read-only review identified
separately reusable scheduling metadata as a possible later candidate: the
row scheduler stores integer/EMA/set state, not tensors or Core ML objects.
However, warmed chunks and visit counts cannot be copied as proof that a newly
constructed owner is warm. Old cost samples may also reflect different load.
Any candidate must keep cold-owner admission/self-test, invalidate on workload
changes and compare complete repeated requests against normal GPU prefix reuse.
The previous 64 GPU probe blocks performed required denoising work in place of
hybrid blocks; their roughly 5.53 seconds cannot be counted as removable overhead.

Checkpoint-header inspection confirmed that Qwen2.1's language FFN is
4096→12288→4096 without bias across 36 layers, logically compatible with the
Runtime SwiGLU geometry. The visual encoder instead has 27 layers of
1152→4304→1152 GELU with biases, outside the current graph contract. A language
integration must preserve the original three-axis mRoPE and DeepStack path;
the generic one-dimensional Qwen3 hybrid encoder cannot simply replace it.
Standard 1024-square references produce 4096 visual patches and 1024 merged
visual tokens; the receipt's retained text-token count is not the encoder's
full input length. The measured 5.585-second conditioning stage combines
weight loading, visual and language work, with no FFN-only timing. These facts
guide later profiling; they do not establish encoder ANE acceleration.

`outputs/app-history-api-20261002/encoder-geometry.json` records the 216 checked
FFN tensor headers and relevant source hashes. Only 87,224 bytes (length prefix
and JSON header) of the existing `text_encoders/qwen3vl_8b_bf16.safetensors` were
read; model weights were not loaded or hashed in full.
