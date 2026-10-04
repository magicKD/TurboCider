# App, local API and runtime ANE source milestone

The previously delivered App/archive remains the `975aa91` build. The changes
below are a subsequent source milestone on `dev-local`; they have not been
repackaged or qualified with model inference. No model was downloaded.

## Reference images

The reference menu exposes reversible preparation: original, longest side at
most 1024, and aspect-preserving fits within 512×512, 512×768 or 768×512.
Preparation does not crop, pad or enlarge. It uses ImageIO orientation and RGBA
PNG derivatives; a no-resize operation retains the original bytes. The original
managed copy is kept. Repeated resizing starts from that baseline, not an
already-downsampled derivative. A whole batch commits atomically and keeps
logical IDs, order, the selected initial image and undo state.

Annotation replacements now retain their logical ID, preventing the preview
from jumping to reference one. An annotated image becomes its own preparation
baseline so a subsequent resize does not remove the annotation. Undo restores
the previous asset. Existing draft records without preparation metadata remain
decodable. History reuse currently restores the submitted input file; it does
not reconstruct preparation provenance from older job records.

Reference thumbnails support drag ordering within the same workspace, with
the existing move menus available as well. External multi-file import retains
its existing ordered, atomic staging behavior. Preparation changes input files,
not output canvas dimensions or Qwen's internal reference encoding area.
Shrinking a reference file to 512² does not itself establish faster inference.

## Local AI interface

`capabilities` advertises the protocol, action envelope schemas, limits and job
lifecycle without loading weights. The same schema drives envelope validation.
Native request compatibility remains the responsibility of `plan` and the model
catalog. The service and CLI reject duplicate decoded JSON keys before they can
be folded, malformed types, unknown fields, embedded NULs in envelope strings,
out-of-range pages and fractions hidden by Foundation numeric rounding.

The standard-library Python client keeps job IDs, bounds message sizes and does
not retry ambiguous submissions. Its timeout leaves the service-owned job
running for explicit status/cancellation. The App can copy the actual socket
and AI connection instructions. A local Qwen Turbo example plans by default;
`--run` explicitly submits sequential stages with ordered references and unique
outputs. See [Local API](../public/LOCAL_API.md).

Installation discovery, shared API/App history and a persisted workflow
scheduler are still open. The `models` action is not an installation inventory.

## Runtime ANE lifecycle

Current runtime graphs receive model-dependent weights through memory slots,
but their compiled Core ML graphs and private verified snapshots use disk.
The runtime is FP16 with a token-row FFN partition. Quantized source weights
are decoded into those slots; this is not runtime W8A8/Hadamard computation.
Qwen runtime currently covers DiT FFN; its encoders remain on GPU. Historical
M4 Max measurements are not evidence for this M4 Pro laptop or this revision.

New private leases use held kernel locks and directory/marker inode binding.
Creation attempts bounded recovery of recognized abandoned snapshots, skips
live leases and refuses unknown identities or links. Snapshot directories and
files have explicit private permissions. Worker/Core ML destruction still
precedes lease destruction. Cleanup failures produce diagnostics; a recoverable
final directory-removal failure restores a locked marker for retry.

This does not guarantee zero disk remnants after every failure. Old or damaged
markers, unrecognized entries, scan/tree limits, and the short creation/final
removal crash windows are conservatively retained. Source artifacts and Core ML
system caches are outside this private-lease cleanup. No inference precision,
partition policy or GPU default changed, and no new speedup is claimed.

## Validation and remaining acceptance

Completed in this work session:

- App-wide Swift type checking, including the preparation view and drag payload.
- CLI compilation/linking with strict RPC validation and capability discovery.
- Objective-C++ runtime syntax checking and C++ lease regression-source syntax
  checking; shell and Python source syntax checks.
- Static review of atomic asset replacement, immutable input ownership,
  annotation provenance, exact page parsing and lease failure paths.

These are build/static checks, not behavioral acceptance. The user requested a
stop to testing; no model, GPU, Core ML or API test service was run. CPU regression
sources were added but remain unexecuted. The previous release's test results
must not be attributed to these changes.

Before promoting this source milestone, run the focused reference, local-client,
RPC and lease regressions once permitted, then inspect the preparation sheet
and drag interactions in a separate App state. Full model experiments remain
stopped. `TURBOCIDER_BUILD_APP_ONLY=1 make build-app` skips compiling Swift test
targets when only the App and model-library helper need rebuilding; it is not
a replacement for acceptance tests.

The overall goal remains open: Playground templates, output-size matching,
complete AI workflow integration, runtime ANE scheduling/W8A8 exploration and
matched laptop performance/quality evidence are not complete in this milestone.
