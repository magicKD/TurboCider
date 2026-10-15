# Shared Qwen Playground workflows — 2026-10-02

This stage continues `3907626` on the M4 Pro / 48 GiB laptop, using the existing
Qwen Image 2.1 and Viggle v0.2.1 r128 six-step LoRA. No model was downloaded.
GPU remains the default. This is a bounded App/API delivery, not qualification
of every workflow's image quality or a new Runtime ANE speedup.

## App and local AI interfaces

`native/workflows/image_workflows.json` is the single catalog for outfit,
identity, face, outpaint and transparent workflows. The build embeds it in the
native library; the Swift App reads the same catalog and request composer as
the C ABI, CLI, Unix socket service and Python client.

- `workflows` discovers role names, order, requirements, modes and limitations.
- `workflow_request` constructs a native v1/v2 request from named local paths.
  It preserves dimensions, sampling, LoRA, execution and cache settings.
- Composition does not read images, open a model, validate weight files or
  submit work. Call `plan`, then `submit`, then poll the returned job ID.
- Instructions are preserved verbatim. Missing required roles, unknown roles,
  bad types, relative paths, NUL strings, duplicate JSON keys and oversized
  envelopes are rejected. The App's copied AI instructions include the new API.

The compact Playground picker has five workflows. Face identity and target
roles are distinct. Transparent without a reference uses text generation;
adding a reference switches to extraction and changes only an untouched default
instruction. User instructions remain intact. Result reuse targets each
workflow's appropriate role. The reference-encoding control is hidden when
text generation has no reference to encode.

Schema-1 two-template drafts migrate in memory to schema 2, preserving the old
drafts and reference bindings. Loading alone does not overwrite the file.
Invalid files remain protected. Expansion is persisted as one of 1.25/1.5/2.

Real UI inspection found parent accessibility identifiers overriding child
buttons and the picker. Explicit accessibility containers now preserve distinct
identifiers, including `playgroundTemplatePicker`, `playgroundGenerate` and
`playgroundChoose.source`. This was verified in the running rebuilt App.

## Validation and quality boundary

Evidence: `outputs/qwen-workflows-20261002/`.

- Complete native package and Swift App builds passed.
- 11 native ABI/CLI workflow methods, 5 real local-service methods and 12
  Python client methods passed. Service checks use local sockets and no weights.
- The expanded Swift Playground suite passed: role ordering, independent
  drafts, cancellation/import rollback, resize/original ownership, transparent
  mode/default transitions, outpaint bounds, migration and persistence.
- Two real requests were exported by `PlaygroundState`, rediscovered/composed
  through the local API, compared exactly with the App request, planned,
  submitted, polled to success and followed by service shutdown. Both used
  512×512, six steps, strength 1, runtime LoRA, GPU, component-staged residency
  and DiT cache off. Both reported 227 LoRA-applied projections.

| Workflow | Request wall | Denoising | Observed result |
|---|---:|---:|---|
| Transparent fox sticker | 24.899 s | 17.408 s | RGBA, clean isolated subject when alpha-composited |
| Prompt-only 2× outpaint | 28.312 s | 20.299 s | Executed correctly, but **did not substantially widen the scene** |

These are functional examples, not a paired speed benchmark. Transparent alpha
ranges from 0 to 255: 49.21% of pixels are below 10, 48.71% above 245 and 2.08%
intermediate. Gray-background inspection shows a coherent sticker and clean
edges. Stored RGB under almost-transparent pixels is noisy; viewing RGB without
alpha is not a valid output-quality check. Near-zero background alpha is not
claimed to be exactly zero. Extraction quality is not covered by this sample.

Outpaint is currently prompt-guided recomposition. The multiplier does not pad
the input, enforce exact framing or lock source pixels. The failed visual sample
is retained; task success is not outpaint quality acceptance. A separate
transparent-padding pilot under `outpaint-guide-pilot/` investigates a stronger
geometric guide. It is not silently used by the shipped App/API composer.
That one additional six-step request took 25.471 s (17.912 s denoising). The
subject retained the smaller center framing, but the outer transparent border
was not filled: this pilot also failed outpaint quality. The stage therefore
used three serial model requests, in a two-request group and a later single
pilot. The App labels the range experimental and explains that requested framing
may not be achieved. This limitation is not hidden behind successful job status.

Running-App checks passed for five-template discovery, face role labels,
missing-reference blockers, transparent generation enabled without a reference,
reference-control switching, outpaint multiplier-to-prompt binding, local API
start/discovery/composition/stop and independent accessibility IDs. Packaged API
composition preserved existing API history and did not open a model session.

The native Open panel opened, but computer automation repeatedly reset its Go
to Folder field while confirming a file. This stage therefore does not claim a
new UI file-picker/import or mouse-drag pass. Earlier import acceptance and the
current CPU importer tests remain separate evidence. The 107 existing history
objects and Creation draft are semantically unchanged; App serialization changed
JSON byte order. The original Playground draft was restored byte-for-byte after
the UI checks, and the test draft was retained separately.

## Runtime ownership and disk

The runtime graph exporter now publishes with macOS `renamex_np(RENAME_EXCL)`.
A concurrently created destination cannot be overwritten, including an empty
directory. Three new CPU regressions plus the existing projection-layout check
passed in 0.107 s; the old code failed the publication negative control.
The production numerical kernels and runtime scheduling are unchanged.

Removed six verified, rebuildable old Swift/Clang module-cache directories:
1,288,288,960 logical bytes (about 1.20 GiB). APFS reclaimed physical space is
not inferred from that number. Models, generated/history images, the current
build and rollback packages were retained. Reports live under
`outputs/runtime-export-publication-20261002/`.

Runtime weights still do not imply disk-free Core ML. Verified private graph
leases are released after their worker and Core ML owners; exported source
graphs, OS caches and conservatively retained crash remnants have separate
lifetimes. This stage does not promise zero disk I/O or zero crash residue.

Next Runtime candidate: overlap only verified graph/model preparation with GPU
conditioning, leaving large slots and self-test until DiT admission. Existing
timings combine several setup operations, so new phase instrumentation and a
matched control are required. The 64 GPU probe blocks do useful denoising work;
their time is not removable overhead. Six-step scheduler state is often not
mature enough to restore wholesale. Current Runtime remains FP16; W8A8 and
encoder ANE are unqualified research paths. See the existing
[Runtime lifecycle measurements](2026-10-02-runtime-staged-lifecycle.md) and
[laptop ANE decision](2026-10-01-runtime-ane-decision.md).
