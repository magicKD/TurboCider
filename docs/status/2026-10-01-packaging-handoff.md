# Packaging handoff after the testing stop

The user requested that testing stop and the current code be organized and
packaged. No more feature work, model inference or behavioral regression runs
are part of this handoff. There are no model downloads.

The package includes the committed reference preparation/ordering, local API
discovery/validation and runtime ANE private-lease changes described in
[the source milestone](2026-10-01-app-api-runtime-source.md), alongside the
earlier Qwen GPU/LoRA/DiT-cache and inspector/history work.

The first archive (`d312dcf`) predates Playground and reference-based output
canvas sizing. The final handoff also includes those committed features from
`cd6cf58`, described in [the source notes](2026-10-01-playground-source.md), and
the in-progress local installation discovery work has been closed out for
packaging. Earlier archives are retained. There is no further feature expansion
in this handoff. ANE scheduling and W8A8 work remain deferred; no new speed or
image-quality result is claimed.

The `installations` RPC reads the typed model-library registry through the
service's own sibling helper. It returns registered model/LoRA/ANE paths with
`files_verified: false`, rather than scanning directories or opening weights.
The helper is bounded by time and output size, uses the service's settings and
does not create missing library directories. The Python client and App's copied
AI instructions document the query. It still needs behavioral acceptance when
testing resumes.

Build the package binaries without compiling test/auxiliary probe targets:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
TURBOCIDER_BUILD_PACKAGE_ONLY=1 \
TURBOCIDER_BUILD_OUTPUT_DIR=build/package-20261001 \
tools/native/build.sh
```

Use `TURBOCIDER_PACKAGE_OUTPUT_DIR=dist/release-<commit>` with
`tools/native/package.sh` and the same build output directory to avoid replacing
the App currently open at `dist/TurboCider.app`. The normal default output remains
unchanged. Package-only mode retains native build-identity/catalog checks and
uses `TURBOCIDER_BUILD_APP_ONLY=1` for Swift compilation. These commands do not
start the App, an API service or an inference process.

For the final handoff, existing model-runtime objects can be reused only after
their source digests in the prior build manifest still match. Rebuild the CLI
with the build script's compiler flags and command. Since the runtime identity
also includes service sources, regenerate its manifest and bundled catalog,
recompile both generated-header consumers and relink the native library.
Rebuild the Swift App and model-library helper with
`TURBOCIDER_BUILD_APP_ONLY=1`. The archive verifier
then compares executable sections in every packaged project binary to the build
output, checks signing and compares every archived file with the App bundle.
This avoids rebuilding unchanged model runtimes and does not run test targets.

The delivery JSON next to the archive records the exact commit, archive hash,
signature/integrity checks and zero new test/model runs. These packaging checks
are not behavioral acceptance. The newer reference/API/ANE changes still need
focused regression and UI verification when testing resumes. The previous
archive and its validation evidence are retained separately.
