# Creation navigation regression correction

The earlier dev-verify validation missed the sidebar return from SeedVR2 or
Real-ESRGAN to creation. The selected sidebar row changed, but the shared draft
still selected SeedVR2 (or the auxiliary-upscaler view remained active). Its
upscale-only capabilities hid generation and editing operations.

Returning to creation now restores the last creation draft, including model,
operation, prompt, inputs and sampling settings. A separate local creation
snapshot survives App restarts. Model registrations and auxiliary-upscaler
preferences remain current. Older installations with only a SeedVR2 draft
recover a compatible configured generation model without downloading weights.
New creation also exits upscale mode; the creation model picker excludes
upscale-only models. SeedVR2 controls are confined to the upscale page, and
the video-upscale operation routes to that page.

Validation on the packaged native App:

- Opened the existing affected SeedVR2 draft and clicked creation: image/video,
  generate, single-image modification and reference-editing controls returned.
- Imported an existing 512 × 512 image, selected reference editing and 7 steps;
  visited SeedVR2 and returned with the same image, prompt, model and steps.
- Used the creation upscale shortcut and switched Real-ESRGAN ×2 → SeedVR2 →
  Real-ESRGAN ×4; returning restored the reference-editing draft.
- Switched to video creation: text-to-video, image-to-video, keyframes,
  reference video and video upscale were visible. Video upscale and new
  creation returned to the original video model and generation settings.
- Quit while in SeedVR2, restarted, visited the model center, then returned to
  creation: the saved FLUX creation draft and prompt were restored.
- Visually checked the complete creation page and left the App on image
  generation with a configured model.

Checks: rebuilt the App and affected Swift test executables; `make test` and
`make test-app` passed, including regression coverage for restart, input-import
locks, registration retention, image/video round trips, new creation and legacy
draft recovery. Packaged with `tools/native/package.sh`; deep strict code-sign
verification passed. No model downloads or new inference runs were needed for
this navigation correction.
