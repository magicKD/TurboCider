# Single-image upscaling and unified model library

The sidebar now contains one creation entry. Existing-image upscaling is under **创作 → 单图修改 → 图像超分**, alongside content editing. Switching tools preserves the selected source; generation still offers **不超分 / 超分 ×2 / 超分 ×4** beside Generate. The image-upscaling route does not call a diffusion engine or require a generation-model download.

The model center includes x2plus/x4plus auxiliary entries and an **超分** filter. **管理模型** opens the selected upscaler. Download links open the upstream ZIP in the browser; the UI explains extraction, import and native Core ML compilation. No silent download or Python conversion is performed. Model import validates the actual scale before registration, so an x4 package cannot be assigned to x2. Registration records an external path in the existing `library.json`; removing it retains the source files. Unified configuration import/export includes the upscalers without adding them to the generation-engine picker or enabling post-generation upscaling. Existing App paths migrate into the same index.

## Verification

- Swift App and native model-library helper compile successfully.
- Model-library store tests pass, including package validation, idempotent registration, durable index storage, locking and removal without deleting weights.
- Synthetic upscaler tests pass, including explicit 2×/4×/off selection and auxiliary path restoration without changing the generation engine.
- Real x2plus tests pass for GPU/CPU+ANE cache changes, image export, cancellation, history restart, and preload/release. Model-library tests with the real package reject the wrong scale, then register/export/remove/import the correct one while preserving opt-out.
- Studio configuration and worker-query regressions pass. System permission was required for Core ML's temporary compilation cache and test-only trash operations; the authorized reruns passed.
- Actual UI: the separate sidebar entry is absent; **单图修改 → 图像超分** produces 1536×768 from a 768×384 input using preloaded CPU/ANE. Mode selection remains stable through refreshes. The model center shows both local models, the correct download/processing links, and the native file chooser reports successful validation/registration.
- Actual UI: returning to text-to-image and choosing Z-Image Turbo produces a 512×512 image at 8 steps, followed by automatic cached x2 super-resolution to 1024×1024. Both outputs are preserved.

Local evidence in `outputs/dev-verify-20260928`: `integrated-real-tests.log`, `integrated-studio-tests.log`, `integrated-tests.log` (store results plus the initial sandbox cache failure), `ui-integrated-release-build.log`, `ui-integrated-result.png`, `ui-integrated-library.png`, `ui-integrated-import.txt`, `ui-integrated-generation.png`. Tests use isolated draft/history and library directories; no other model downloads were performed.

The final local App package passes strict signature validation and matches the built executable. Startup of the actual App registered both existing packages in its persistent model library; its model-center filter and configured status were verified (`ui-integrated-actual-app.txt`).

This change does not promote experimental streaming configurations or qualify absent optional models. Those retain their existing gates and static/contract coverage.
