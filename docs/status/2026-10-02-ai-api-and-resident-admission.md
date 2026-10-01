# Local AI discovery, workflow acceptance and Runtime memory admission

This continues `d572abe` on the M4 Pro / 48 GiB laptop. No model or dependency
was downloaded. The only real-model work in this stage is two sequential
512×512, six-step GPU edits using the existing Viggle v0.2.1 r128 adapter at
strength 1. Builds, functional model requests and regressions are serialized.

## CLI and request discovery

The CLI previously rejected conventional `--help`, and its no-argument usage
line did not provide a practical AI entry sequence. `help`, `--help` and `-h`
now describe discovery, planning, generation and local API workflows. The new
`capabilities` command emits the same object as the live RPC action's result,
without starting a service, reading an installation registry or loading weights.
Both paths call the existing shared protocol definition; a second schema was
not introduced. Invalid command/argument errors point to `--help`.

Qwen `reference_encoding.constraints` previously described only the strict
Viggle shortcut despite also advertising base sizes 256 and 512. Discovery now
separates `base_constraints` from `viggle_r128_gpu_edit_constraints` and reports
ordinary LoRA's reference size of 1024. `plan` remains authoritative for the
complete combination. This changes metadata, not route admission.

The initial native build passed the CLI discovery test and all four RPC
validation tests. Real CLI/RPC checks verify no registry/state creation by
read-only discovery, equality of offline/live capabilities, base 256/512
admission with one/three references, r128 opt-in, ordinary-LoRA rejection and
malformed-envelope isolation. Final build/recheck receipts are recorded under
`outputs/api-discovery-20261001/`.

## Two real API edit stages

The public Python client discovered the registered local LoRA, planned each
request, submitted once and polled progress. The first output became the second
request's reference input. Both used reference encoding 512 and output 512²:

| Stage | Instruction | Native request wall | Observed output |
| --- | --- | ---: | --- |
| 1 | Change the existing red teapot to cobalt blue | 25.176 s | Dark blue teapot; overall handle, spout, lid, framing and table retained |
| 2 | Add a thin white band around that blue teapot | 24.496 s | White band added; overall shape and framing retained |

Both results report six denoising steps, GPU encoder/execution, 1,024 reference
tokens and 227 LoRA-applied projections. PNG signature, IHDR, RGBA dimensions,
chunk CRCs, decompressed scanline size/filter types and SHA-256 were checked.
Both images were visually inspected. Background/glaze grain and contrast
increased, particularly in the second edit. This qualifies the API workflow's
execution and broad instruction following, not pixel preservation or repeated
editing quality. Standard 1024 reference encoding remains the default.

The harness first failed before starting any service because Pillow was absent;
it now validates PNGs with the standard library. After stage 1 succeeded, an
incorrect harness expectation for the App worker's `image_artifact` field also
failed: native service results expose `output`, not that worker-only envelope.
The service was stopped cleanly. The corrected harness verified the persisted
successful job and existing output, then resumed stage 2 without resubmitting
stage 1. The original failure logs/reports are retained. Consequently these
are two functional cold requests with a service restart between them, not a
resident throughput comparison or a cache-speed measurement.

Cancelling each already-succeeded job left it unchanged. Both jobs survived a
service restart with no model session opened; final shutdown removed the owned
socket. Active denoising cancellation was not exercised in this campaign.
Evidence and both images: `outputs/api-discovery-20261001/real-api-report.json`,
`api-edit-stage-1.png`, `api-edit-stage-2.png`. The user's App draft/history was
not used as the service store or modified.

## Accurate service session metadata

Source review found `service_session_reused` was gated on the LTX-specific
`resident_candidate` flag even though Qwen and other native models also retain
their engine for the same model/path. Their later requests could therefore
report false while using the existing engine. The field now uses the actual
service-owned engine identity independently of the LTX routing label.
`service_execution_path` keeps its original meanings. Shared discovery and
public documentation clarify that engine reuse does not guarantee weight
residency or a prompt-cache hit.

The same lifecycle review found that a failed model replacement freed the engine
but retained its old identity. A later request for that old model/path could
skip creation and try to generate with a null engine. Replacement now clears
the identity before loading, and a missing engine always requires creation.

These service bookkeeping corrections follow the two real requests above; the final
source manifest differs from their build only in `service.mm` and
`capabilities.hpp`. Numerical model sources, request admission, toolchain and
build flags remain identical. A dedicated test service uses the real plan/RPC/
queue/lifecycle code with CPU fixture engines whose creation IDs and free events
are observable; it is not a production binary or model-quality test.

The final native/Swift package build passed. All eight test methods passed on
that build: one CLI discovery test, four real RPC validation tests and three
CPU-engine lifecycle tests. The latter verify observed engine creation IDs,
reuse, model/path switching, recovery after a controlled load failure,
disposable-worker ownership and final engine/socket cleanup. No model weights
were loaded by these regressions. `final-validation.json` records log hashes,
all 426 native source inputs, matching numerical sources/toolchain since the
two real edits, and unchanged App Swift sources relative to `d572abe`.

## Runtime FFN admission on this laptop

A read-only audit used the existing `d572abe` release's actual `plan` command and
confirmed physical memory of 51,539,607,552 bytes (48 GiB). For the existing
runtime route, 512² r128 six-step requests require resident execution:

| Request | Plan estimate | Runtime minimum including 4 GiB margin | Available |
| --- | ---: | ---: | ---: |
| No reference image | 44.5 GiB | 48.5 GiB | 48 GiB |
| One standard reference image | 47 GiB | 51 GiB | 48 GiB |

This is a real execution gate, not an inference from an informational estimate:
`qwen21/pipeline.cpp` calls `ResidencyPolicy::validate_budget` before encoding
or DiT loading, and `native/runtime/residency.cpp` requires physical memory to
cover the estimate plus 4 GiB. `plan.executable=true` does not certify this
runtime check. `component_staged` is currently rejected by the runtime route's
residency guard. No model load was attempted merely to reproduce this known
rejection, and no memory protection was relaxed. The retained plan responses,
physical-memory reading, release hash and execution-gate source hashes are in
`outputs/api-discovery-20261001/runtime-admission-report.json`.

The historical Qwen runtime manifest path in the repository's older evidence
is absent on this laptop. The registered local ANE inventory contains Z-Image
and FLUX partitions, not a matching Qwen runtime graph. A Viggle runtime route
also needs the complete v2 activation-input contract; the already-cleaned tiny
research graphs are not substitutes.

Read-only lifecycle analysis shows a possible future direction rather than an
implemented solution: text/vision weights are locally scoped, and the runtime
FFN graph is constructed for DiT afterwards. Normal denoising clears callbacks
and drains weight references before staged DiT unload. However, current staged
cleanup does not reset `runtime_ffn_`, so its model/slots/scratch could survive
VAE decode, idle time and the next request's encoding. A staged runtime candidate
must establish explicit phase lifetime and failure/cancel cleanup, a matching
v2 graph, memory admission and a complete-request benefit before relaxing the
existing residency guard. Qwen encoder ANE remains unimplemented.

## Delivery boundary

The Mac is still locked. The latest actually installed/opened App remains
`67563e8`; new packages and state regressions do not establish fresh UI acceptance.
Actual App replacement/relaunch, mouse multi-file drag, canvas controls and the
App-owned API toggle still need an unlocked desktop. The previous lifecycle,
reference-sizing and Playground evidence remains indexed in the preceding
reports. No claim of universally bug-free UI or whole-model ANE acceleration
is made.

The committed stage is packaged separately under `dist/release-<commit>/` and
`dist/TurboCider-macOS-arm64-<commit>.zip`; its delivery JSON records source,
executable-section, signature and archive verification. These artifacts do not
replace the running App while the desktop remains locked.
