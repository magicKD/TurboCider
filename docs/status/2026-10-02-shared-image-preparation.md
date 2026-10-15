# Shared App and AI reference preparation — 2026-10-02

Reference downsizing now uses one CPU implementation for the App and external
agents. An AI can discover the presets, explicitly prepare a reference, and
pass the returned `image_path` into the same workflow composer used by the App.
No inference session, model download or generation job is needed for resizing.

## Delivered behavior

- Five presets: original, automatic (1024 bounds), fit512, portrait512 and
  landscape512. Preserve aspect ratio, EXIF orientation and alpha; never crop,
  pad, upscale or overwrite the source. Small images keep their original bytes.
- The App uses the shared C ABI renderer while retaining its existing original
  copies, batch rollback, undo, annotation baselines and history metadata.
- CLI `prepare-image`, RPC `image_prepare`, Python `Client.prepare_image`,
  capability schemas and App AI instructions expose the operation explicitly.
- File output is an exclusive atomic PNG publication. Existing files,
  directories and symlinks are preserved. Failed publication cleans only the
  owned temporary inode; competing requests cannot overwrite each other.
- The caller owns created derivatives. A no-op returns the source path instead.
  Lost replies are marked separately from ambiguous generation submission;
  neither operation is automatically retried.

See `docs/public/LOCAL_API.md` for schemas, examples and cleanup rules.
The operation is synchronous on serial RPC dispatch, so a large decode may
hold up other replies while leaving the inference worker unlocked. It has no
immediate cancellation guarantee. The existing one-frame/80-million-pixel
import limit remains in force.

## Acceptance

Frozen native and Swift App build succeeded. All 434 native manifest inputs
match the tested source tree.

| Check | Result |
|---|---|
| C ABI and CLI preparation | 9 test methods passed |
| Live local RPC and discovery | 7 test methods passed |
| Python client and lost-reply semantics | 17 test methods passed |
| Shared workflow composition | 11 test methods passed |
| Swift reference/import/history integration | Passed |
| Former Swift renderer versus native renderer | 10 cases identical |

The independent renderer comparison used an odd-size RGBA image with varied
transparency and an EXIF-6 TIFF, across all five presets. All eight resized PNGs
were byte-identical; both original cases returned no derivative. The Swift
suite additionally verifies alpha pixels, batch rollback/context changes,
undo, restoration, annotation baselines, reordering and historical reuse.
The initial sandboxed Python-client attempt could not bind Unix sockets; the
same suite passed outside that sandbox. No product defect was inferred from
that environment restriction.

Evidence: `outputs/shared-image-preparation-20261002/final-validation.json`,
its referenced logs and `legacy-comparison.json`.
The desktop remains locked: fresh UI interaction and replacement of the
running installation are pending. Installed App remains `30d25a4`; the previous
`ec64c63` package is preserved. User draft, Playground and job files remain
byte-identical to the preceding stage's backup.

## Separate outpaint pilot

One serial 512×512, six-step Viggle Turbo GPU request tested an edge-padded
scene guide. Request time was 25.908 s, denoising 18.375 s. The result has one
subject, a coherent opaque surrounding scene and no transparent outer hole.
However, the subject shrank further than the guide: exact expansion geometry
and preservation of original pixels are not qualified. This pilot is not
integrated into the production workflow.

Evidence: `outputs/qwen-outpaint-edge-20261002/verification-report.json`.
No additional model was downloaded. No new Runtime ANE benchmark was performed
for this preparation change; prior Runtime results and limitations continue to
apply.
