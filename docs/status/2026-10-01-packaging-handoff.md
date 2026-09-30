# Packaging handoff after the testing stop

The user requested that testing stop and the current code be organized and
packaged. No more feature work, model inference or behavioral regression runs
are part of this handoff. There are no model downloads.

The package includes the committed reference preparation/ordering, local API
discovery/validation and runtime ANE private-lease changes described in
[the source milestone](2026-10-01-app-api-runtime-source.md), alongside the
earlier Qwen GPU/LoRA/DiT-cache and inspector/history work.

Playground and reference-based output-canvas matching are not integrated. The
unintegrated canvas helper and test-source draft are preserved locally in
`outputs/deferred-20261001/`, outside the shipping source. Further ANE scheduling,
W8A8 and local-installation discovery proposals remain deferred. No new speed or
image-quality result is claimed.

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

The delivery JSON next to the archive records the exact commit, archive hash,
signature/integrity checks and zero new test/model runs. These packaging checks
are not behavioral acceptance. The newer reference/API/ANE changes still need
focused regression and UI verification when testing resumes. The previous
archive and its validation evidence are retained separately.
