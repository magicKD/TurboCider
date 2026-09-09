# Creation-first UI and Turbo Drop — 2026-09-09

## Scope

- Replace the previous identity with the user-supplied Turbo Drop concept:
  committed PNG master, reproducible sidebar/icon assets and README logo.
  See `assets/branding/README.md` for the ImageGen edit provenance.
- Keep image/video creation visible before model selection. Operation choices
  come from executable catalog capabilities; changing operation automatically
  selects a compatible model, preferring configured paths. Model-path validity
  remains a separate installation check. Unsupported operations do not mutate
  the draft. Automatic model switches retain prompts/assets and explicitly
  warn that model-specific parameters, LoRA and acceleration have reset.
- Filter the model picker by creation type and preserve a compatible operation
  when choosing another model.
- Expose sampling steps in the primary parameter panel. Z-Image Turbo and GGUF
  accept 1–50 steps across App/native request validation and default to 9.
  Reset restores the catalog default. Reselecting the same model and preparing
  a session preserve custom steps; drafts persist them.
- Keep the measured-workload automatic ANE gates unchanged. The sampler already
  builds its schedule and reports progress using the requested step count.

## Validation

- Native engine/CLI build completed; final complete Swift App/integration-test
  build completed. Used `DEVELOPER_DIR=/Library/Developer/CommandLineTools` and
  `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` because the local
  Xcode installation reports a CoreDevice/Mercury loader mismatch.
- `make test` passed with that environment outside the sandbox (system video
  encoding is unavailable inside the sandbox). Five optional cases were skipped:
  three missing real-checkpoint fixtures and two NumPy-dependent cases. Both
  NumPy-dependent suites were rerun with `.venv/bin/python` and passed with no
  skips. Missing real-checkpoint cases are not counted as passes.
- Final `make test-app` passed, including operation-first selection, unsupported
  operation rejection, custom-step persistence, request propagation and range
  validation. Native contract coverage includes both Z-Image formats, default 9,
  steps 1/8/9/20/50 and rejection of 0/51.
- `make test-library` passed with tiny loopback fixtures, no model downloads.
- `make test-api` passed, including App service lifecycle, execution ownership,
  duplicate-socket isolation and failure recovery.
- Packaging and `codesign --verify --deep --strict dist/TurboCider.app` passed.
  This is an ad-hoc local package, not a notarized release; its dependency-derived
  minimum macOS version on this machine is 26.2.
- Inspected a separately identified test App using a temporary draft directory:
  new logo visible; image/video entries visible; selecting video automatically
  selected H3; image model menu excluded video models; Z-Image initially showed
  editable 9 steps. Changing it to 12 was confirmed in the persisted test draft.
  The computer-use connection closed after that edit, so the reset-button click
  was not visually rechecked. Test App processes were stopped. The user's
  existing App/draft were not edited.
- `git diff --check` passed.

## Merge assessment

At the refreshed remote check, `origin/main` was `f0201d5` and the working branch
was `dev-verify` at `b3b5e29`, ahead by 14 commits and behind by zero. Main is an
ancestor, so the committed branch can fast-forward without text merge conflicts.
This change is still uncommitted; no commit, merge or push was performed.

The scoped UI/step changes are suitable for development-main review given the
checks above. This does not certify every pre-existing change in the branch or
constitute a production-quality release gate. No real-model generation at custom
step counts, GPU/ANE image parity matrix, full video-model regression, multi-Mac
performance sweep or notarization was performed in this pass. Non-default step
quality and speed remain unqualified. Preserve those limits in release notes.
