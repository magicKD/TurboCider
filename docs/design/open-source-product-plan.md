# Open-source product readiness

This is the implementation checklist for the September 2026 product pass.
Existing development and benchmark records are retained. Checked items need
code or runtime evidence; a working backend alone does not prove a working UI.

## Deliverables

- [x] Bilingual README, reference-inspired vector logo, packaged App icon,
  newcomer guide, contribution guidance and reproducible performance claims.
- [x] Searchable model library with separate detail/preparation views; show
  executable input/output capabilities, local paths and resident sessions.
- [x] One configurable model root shared by App and CLI; external registrations
  and symlinks; reusable tokenizer/encoder components with validated identity.
- [x] ModelScope-first and Hugging Face downloads, progress, cancellation,
  integrity checks, partial-file handling, and install validation. Test with
  tiny fixtures; do not download full models during this work.
- [x] Matching CLI model management and friendly ANE export/compile/cache flow.
- [ ] App image/video workflows, input restrictions, clear step progress,
  honest device metrics and useful memory state.
- [x] Local API accessible from App and CLI, documented lifecycle and smoke tests.
- [x] Prompt caching across seed changes, observable cache hits, scoped cleanup
  controls and configurable retention for regenerable tensor caches.
- [ ] Build/package tests, CLI integration checks, App UI inspection, and a
  requirement-by-requirement completion audit.

Current evidence: [2026-09-08 requirement audit](../status/open-source-product-2026-09-08.md).
The two open items retain the unverified actual local App video run. Package,
image workflows, input restrictions, progress, telemetry and cache controls
have passed their checks; a compatible existing video model is still needed.
The progress sections below are chronological records, not the current checklist.

## Decisions

Use the native model registry as the authority for executable capabilities.
An upstream model's advertised capabilities must not expose an unimplemented
TurboCider operation. Keep download preparation outside the inference process.
Treat a model library as installations and shared components, rather than a
second copy of every upstream repository. Never delete external model files
when removing a registration.

CPU orchestrates native execution while selected GPU and ANE subgraphs run in
parallel. Report measured speedup for named workloads; do not claim universal
losslessness, three-device arithmetic overlap, or SOTA without supporting
comparative evidence. ANE routing and Core ML calls are observable; public
per-task ANE utilization is not currently available.

## Progress and remaining integration (2026-09-08)

Implemented and verified:

- Bilingual README, new vector branding matching the supplied Squeeze concept,
  reproducible App icon, `GETTING_STARTED.md`, `PERFORMANCE.md`, and contribution
  guidance. The top-level license remains pending the owner's async selection.
- `ModelLibraryView.swift` replaces the all-model cards with search/filter,
  detail view, executable input/output badges and one session banner. Packaged
  App UI was inspected; searching LTX shows only text-to-video and leaves the
  active Z-Image draft unchanged. Sidebar logo loads from the package.
- `services/model-library/` implements a standalone Foundation helper with an
  atomic index, external registrations, process lock, content-addressed blobs,
  shared links and staged publication. Native CLI dispatches `library list`,
  `register`, `remove`, `plan`, and `download` to this helper.
- ModelScope-first and Hugging Face clients parse official metadata, resolve HF
  revisions, traverse/page file listings, verify file sizes/hashes, strip
  credentials on cross-origin redirects, and cancel without publishing partial
  installations. No Python dependency for this helper.
- Tiny loopback integration tests passed for both providers, reuse, bad hashes,
  bad sizes, unsafe paths, cyclic pagination and cancellation. Real Z-Image
  metadata-only preflight succeeded on ModelScope (33 entries) and Hugging Face
  (32 entries, revision `f332072aa78be7aecdf3ee76d5c247082da564a6`). No weights
  were downloaded. These full-repository plans include docs; product recipes
  should select runtime files and shared components instead.
- `make test` passed after removing an accidental default dependency on a
  mutable sibling LTX checkout. The committed runtime snapshot has a SHA-256
  source manifest; explicit `TURBOCIDER_LTX_REFERENCE` still enables byte-for-byte
  reference synchronization checks. No LTX math changed.
- App behavior tests passed with actual clipboard/trash access; model search
  and capability tests passed; CLI helper dispatch and local index operations
  passed. `git diff --check` passed. App/CLI package includes the new icon and
  helper, with ad-hoc signature verification.

Next required work (not completed by the backend tests above):

1. Integrate the helper into the App: shared root configuration, existing path
   migration, installation selection, remote source picker, download preview,
   byte progress/cancel/retry, and component reuse controls. The App's current
   folder registration still lives in `StudioDraft.modelPaths`.
2. Provide verified per-model download/preparation recipes for all current
   families, including video runtime layouts and dependencies. Generic repo
   download does not establish runnable models. Validate component compatibility
   from model metadata, not solely caller-provided labels. Add config-file import
   and convenient CLI model-path resolution from registrations.
3. Harden installation maintenance: persisted download history, safe retention
   for abandoned staging, root changes and repair, disk accounting, and useful
   integrity/availability states. Current retries reuse complete blobs but do
   not resume partial byte ranges. Keep external data intact.
4. Integrate friendly ANE export/compile operations with model installations;
   retain identity-bound cache reuse and exact input-capacity checks.
5. Finish runtime UI: prominent step progress, actionable device/session/memory
   details, prompt-cache hit reporting and tensor-cache cleanup/retention.
6. App control of the local API and matching CLI lifecycle; verify endpoints
   against actual service behavior. Unix socket service already exists, but
   App service controls / optional HTTP access are not implemented by this pass.
7. Rebuild/package after integration, verify App image/video workflows and all
   explicit requirements before marking the goal complete. Preserve existing
   local weights, history and development records.

## App/API/cache integration update (2026-09-08)

The earlier "next required work" list is historical. Item 1 is now integrated:
App and CLI share the persisted model root and index; existing App paths are
migrated without moving files. App download preview/source selection, shared
text selection, progress and cancellation use the same native helper. On the
actual App, Z-Image preview with the local FLUX text components selected yielded
9 runtime files / 24.79 GB and skipped encoder/tokenizer weights. No full model
was downloaded. Reuse checks validate Qwen3 structure, tokenizer and shard
availability rather than trusting a caller label. Native CLI generate/batch
also resolve `@model-id` registrations.

App's Local API page now starts/stops the packaged CLI service. Start releases
the embedded session and excludes conflicting App inference. Service identity
is checked against the child PID. Parent-loss detection shuts down App-owned
services; independent CLI services remain independent. Service status exposes
active job, open-session model and external worker state. App-controller tests
passed start/status/stop/restart, duplicate-socket isolation and failed launch
recovery. Parent-kill tests passed socket/state-lock release and restart. Native
service tests used existing FLUX weights to generate real images, verify cache
reuse across seed42→43, cancellation, GPU exclusion and crash recovery. Actual
App start→external RPC submit→512×512 image→stop succeeded. Raw evidence:
`outputs/open-source-api-app-20260908/` and
`outputs/open-source-api-seeds-20260908/`.

Runtime UI now highlights completed denoise steps. RunInsights renders optional
prompt-cache, timing, MLX snapshot and Core ML cumulative metrics; it never
substitutes zero for missing measurements. Historical App evidence with calls
288→576→864 and row1536→1536→1056 passes its regression reader. The model page
separates a session snapshot from unrelated cache-resource reports. ANE real
occupancy remains explicitly unknown.

Tensor cache management is shared by App Settings and `turbocider cache`.
It recognizes identity-bound LTX connected tensors, supports manual deletion
and configurable retention (off by default), and acquires the runtime GPU
lease before removal. App runs an hourly sweep only when enabled and idle.
Fixture tests passed age selection, manual cleanup, active-runtime refusal,
unknown-media preservation and symlink exclusion. CLI policy persistence and
invalid-day rejection passed. This currently excludes arbitrary research output
tensor dumps, LoRA merged weights and model-library blob GC; these need their
own explicit ownership/retention treatment rather than deleting `outputs/`.

`make test`, tiny-provider integration tests, model search/capability tests,
model-library transport/persistence tests, telemetry fixtures and tensor-cache
fixtures passed in this pass. Some existing optional NumPy/model-fixture tests
remain skipped in the baseline environment. UI inspection of the latest
telemetry/settings and final rebuilt package remains required at this point.

Remaining product work: finish all-family install validation and video model
preparation recipes, registration repair/availability/storage accounting,
friendlier installation-bound ANE preparation, scoped diagnostic-output cleanup,
actual video App workflow inspection and final requirement audit. Owner license
selection remains pending. The full open-source goal is not complete.

### UI verification follow-up

The packaged App Settings page was inspected. Selecting 30-day retention was
visible to `dist/cli/turbocider cache settings`; restoring "不自动清理" returned
retentionDays0. No actual tensor entries were deleted. The API page was visually
inspected, started and stopped successfully. The model page rendered real
Z-Image metrics (32 partitions, 1056 rows, zero output-copy bytes), and releasing
the session removed its runtime snapshot. Three actual App runs completed at
27.235/16.566/16.435 seconds, with changed random seeds and cache hits false/true/
true; Core ML model load stayed6.7664 seconds across the session. These are
functional checks under desktop load, not a controlled speed comparison.

Capturing an in-flight frame exposed an existing progress bug: Z-Image's
`z_image_denoise_block` callbacks replaced the denoise step display. The App
now suppresses these layer counters like FLUX's `transformer_block`, while
keeping elapsed-time/cancellation updates. The expected UI is sampling n/9.
A subsequent fresh-process check exposed Foundation `attributesOfItem` waiting
inside getxattr during ANE discovery. Only size/device/inode were needed, so
that path now uses POSIX stat; native checkpoint hashing remains unchanged.
The stat/progress changes still require the final runtime recheck below.

The latest POSIX stat build, shape-routing test (including512/513 boundary),
Studio behavior suite and ad-hoc package verification passed. Runtime UI recheck
is currently pending macOS file access: App PID64902 is alive, waiting inside
`open` while AccelerationDiscovery reads a compiled partition identity file.
Before the stat change, the wait was in Foundation getxattr. This is not evidence
of a geometry mismatch or model compiler work. The system UserNotificationCenter
process exists; CUA explicitly refuses to inspect that app. The user was asked
to check any Documents/model-file access prompt. Do not bypass that restriction
or repeatedly restart the current request. The prompt remains the safe fox text,
GPU+ANE,512×512,9steps,LoRA off. Restore empty prompt and release the verification
session after the pending request/UI progress check completes.

The stat change avoids unnecessary extended-attribute queries but does not
resolve or bypass OS access requirements. The progress callback fix is compiled;
its final in-flight screenshot is still pending the file-access check. Before
this OS wait, Settings, runtime metric cards, App/API generation and seed-cache
reuse were all verified on the actual desktop App. Retention remains disabled.

### Resumed App verification

After the user asked to continue, the pending App generation had completed at
26.1 seconds. No system dialog was operated or bypassed. Another actual App
GPU+ANE generation completed at16.6 seconds with a different random seed and
text-conditioning cache hit. Live accessibility inspection at five seconds
showed “采样 2 / 9 步已完成”, progress2/9,1.82 seconds/step and an estimated13
seconds remaining. Both jobs reused the enumerated partition cache; model load
stayed6.38155 seconds, cumulative calls288→576 and output-copy bytes0.
The empty prompt was restored through the App and the session released.
Evidence: `outputs/open-source-ui-20260908/progress-recheck.json`.
The earlier file-access wait no longer blocks App verification; its underlying
OS cause was not established. The product goal remains active.

### Installation and download follow-up

App and CLI now expose a read-only installation inspection for all six catalog
families. It checks component directories, every indexed shard, bounded JSON
headers and safetensors byte ranges. FLUX configuration checks distinguish4B/9B.
Z-Image accepts the native loader's single-file linked-component case. Results
separate files-present, incomplete and needs-preparation; they do not certify
tensor math or runtime provenance. Download completion runs this inspection and
does not describe an intentionally partial selection as ready.

Fixture tests passed missing shards, unsafe paths, truncation, invalid offsets,
shared/broken links, architecture mismatch, video preparation and report
transport. Existing local Z-Image and FLUX installations pass CLI inspection;
the actual App displayed FLUX's successful check,4 weight files/15.96GB. App
Z-Image inspection timed out while opening a weight file; an own-child sample
confirmed FileHandle→open, not tensor parsing. No OS dialog was bypassed. The
App remained responsive and reported the timeout; permissions were requested
from the user. This is distinct from the successful prior image generation.

Remote metadata exposed a nonexistent LTX tokenizer prefix in the earlier
single-repository recipe. The corrected plan combines4 exact weight files from
comfyicu/LTX-2.5 and only tokenizer.json from mlx-community/ltx-2.5-mlx. The same
provider is used for every source; revisions and per-file source identities
remain in provenance. Path collisions fail before download; all files are
published atomically and reuse verified blobs. Tiny local server tests passed
multi-source transfer, integrity, reuse and collision rejection. Actual App and
CLI metadata previews both showed5 files/39.36GB. H3 now selects onlyFL2VA and
FastMetal selects the releasedmlx_dit/supporting components. Their HF previews
succeeded at81 files/144.05GB and17 files/13.40GB respectively. No model weights
were downloaded. Evidence: `outputs/open-source-installation-20260908/`.

The scoped Swift/App build, packaged CLI inspection, signature verification,
helper transport, model capability/search tests and `make test` passed (the
baseline optional fixture/NumPy skips remain). The newest single-file-link and
diagnostic-cache changes still require the next build/package pass.

### Final package and UI follow-up

The subsequent complete Swift build and final scoped audio-gate build passed.
Installation, library transport/store, multi-source tiny downloads, diagnostic
cache, Studio behavior and model capability tests passed. The packaged App and
CLI contain the locally prepared MIT project license; this is a reviewable
default, not a claim that the owner explicitly selected MIT. Third-party and
model license terms remain separate. Strict App signature verification and
`git diff --check` passed. No commit or publication was made.

The final App's diagnostic-cache UI enrolled a synthetic temporary directory,
reported one 77-byte eligible tensor, deleted that tensor and preserved the
other file. Removing the registration retained the directory. Read-only CLI
checks confirmed no enrolled directories and retention disabled afterward.
No original model, media or research tensor was enrolled or deleted.

The final LTX page now has no audio toggle and explicitly says its executor
outputs video without an audio track. Its image-input controls are disabled;
the default video geometry is704×448,97 frames at24FPS with GPU selected.
The final Studio test rejects unsupported audio requests. These are UI and
request-validation checks, not a local video inference qualification: no
compatible LTX/H3/FastMetal weights were found, and none were downloaded.

The original Z-Image draft was restored through App configuration import and
its temporary prompt cleared through the UI. JSON comparison against the
pre-test backup confirmed exact equality. The App is unloaded, on the Z-Image
creation page, with512×512, GPU+ANE, random seed and LoRA unchecked at1.0.
Both base and LoRA flexible compiled manifests remain configured.

Current outstanding runtime checks are an actual App video generation using
an existing local model path and the optional App Z-Image installation probe
after the OS file-open wait is resolved. Earlier “pending final package/UI”
notes above are historical and are superseded by this follow-up. The current
requirement audit is `docs/status/open-source-product-2026-09-08.md`.
