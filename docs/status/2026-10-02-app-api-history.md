# App and local AI task visibility — 2026-10-02

The local API page now shows paginated jobs with status, prompt, time, errors
and image results. Completed images can be previewed, opened, located in Finder
or explicitly imported into the current Creation draft. Imports preserve the
draft's model, prompt and generation settings. API jobs stay in their own
history; listing jobs neither loads weights nor runs inference.

Failed reads and stopped services retain the last successful page with an
outdated-data warning and refresh timestamp. Response validation rejects
malformed pages, duplicate IDs and invalid result metadata. Only a successful
result's absolute local output path enables result actions; the request's
proposed output never does. Lifecycle and request identities prevent slow or
old service responses from overwriting the current page.

History loads on page entry/service start and explicit refresh. It does not
add a continuous history poll. Thumbnails are decoded off the main thread at
a bounded size. Separate accessibility containers keep child actions reachable.

## Acceptance

- App and focused Swift integration executable compile successfully.
- Integration tests pass against an isolated local service and protocol
  fixtures: service ownership/lifecycle, empty and populated history, valid and
  invalid result paths/schema, pagination bounds, duplicate pages, retained
  stale data, slower page responses and stop/restart isolation.
- Actual App: start local service, read an existing completed image job,
  preview the image, import it as a reference, stop service and retain history
  with a stale warning. No generation request was submitted.
- Actual Creation UI: paste two local files together; retain all three
  reference identities; resize the 1024² reference to 512²; restore its
  original 1024² copy; move it forward using its menu while preserving the
  selected canvas image. Collapse and expand the right settings panel.
- Playground renders its independent workflow and role inputs. This stage
  does not repeat model-quality tests for each template. Scene expansion is
  prompt-guided recomposition, and transparent extraction still needs output
  alpha/quality inspection as documented in the public API guide.
- Native mouse dragging was attempted but not successfully exercised through
  desktop automation; it is not counted as a passed UI check. Multi-file paste
  and menu reordering were exercised successfully. Open-in-external-app and
  Finder buttons were not separately exercised in this stage.
- The user's Creation draft, Playground draft and Creation history were
  compared before restoration: only test references/selection had changed;
  other state was semantically identical. All three files were restored
  byte-for-byte. Four owned test inputs were archived into project evidence
  and removed from the App's inputs folder; original images remain unchanged.

Evidence is in `outputs/api-task-history-20261002/`: `api-tests.log`, frozen
build hashes, UI draft snapshots, `user-state-restored.json` and
`final-validation.json`. Swift sources and all 434 native manifest sources
are checked against the tested binaries before packaging. Distribution
verification checks executable sections, strict local code signing and ZIP
file contents; installation receipts live alongside the stage evidence.

No models were downloaded and no inference or Core ML export ran in this App
acceptance stage. Per the user's revised scope, further work is focused on
App usability and local AI integration. The preceding single Runtime/GPU
comparison showed no speed advantage, so GPU remains the default; see
`2026-10-02-runtime-fixed-two.md` for the qualified result and cleanup evidence.
