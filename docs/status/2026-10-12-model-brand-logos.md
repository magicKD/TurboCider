# Model hub publisher logos

Accepted on 2026-10-12, following the model hub/component/LoRA milestone
`c7bf3f4`. This update is prepared for integration into `dev-verify`.

## Delivered behavior

- Model rows, detail headers and download sheets use seven publisher logos:
  Qwen, Black Forest Labs, Lightricks, MiniMax, Wan, Tongyi-MAI and inclusionAI.
  The original PNG/JPEG bytes came from the publishers' official GitHub
  organization avatars. No model weights were downloaded.
- The seven images total 60,176 bytes. They ship inside the App, are read once
  from its bundle and require no runtime network access. A white tile keeps
  the original marks readable; missing assets retain the native badge fallback.
- LLaDA now identifies its publisher as inclusionAI. The former generic
  fallback incorrectly labeled it Real-ESRGAN.
- Source URLs, retrieval date, byte lengths, checksums and attribution are in
  [the asset manifest](../../assets/branding/model-brands/SOURCES.json) and
  [the asset README](../../assets/branding/model-brands/README.md).

## Acceptance

- The App and model-library helper were rebuilt with Command Line Tools and
  SDK 15.2. The updated package passed strict deep code signature verification.
  All seven packaged images and the provenance files match the repository bytes.
- `make test-app` passed all 16 executable suites after rebuilding them from
  the current sources. Coverage includes reference resizing, edit canvas,
  Playground, Turbo/LoRA state, persistence, history, worker transactions,
  streaming selection, model library and cache cleanup. These are CPU/tiny
  fixture checks, not real-model inference or an ANE export.
- `make test-library` passed all eight stages, including component inventory,
  model hub presentation, LoRA management, installation checks, helper transport
  and tiny loopback download fixtures. Repository layout and independence
  checks passed all 15 cases.
- Native UI inspection confirmed publisher logos in model rows and headers,
  Qwen's download header, and the corrected LLaDA publisher. LoRA navigation
  retained the four existing registrations. The App was left at the Qwen model
  page with no model loaded and no download started.

The initial App test attempt found a missing executable and then a stale
pre-merge upscaler test binary. Rebuilding all 16 App test executables resolved
both; no product assertion was weakened. Diagnostic logs are retained separately
from the successful current-source run.

Local logs and package/source checksums are under
`outputs/model-hub-logo-20261012/`, including `app-regression-current-sources.log`,
`library-regression.log`, `app-build-final.log`, `package-final.log` and
`acceptance.json`. The deliverable is `dist/TurboCider.app`; generated binaries
and local receipts are excluded from Git.

This visual update retains the inference compatibility boundaries and the
external Z-Image download-sheet automation limitation documented in
[the preceding acceptance report](2026-10-10-model-hub-components.md).
