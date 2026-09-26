# Qwen-Image-2.1 in TurboCider

TurboCider runs Comfy-Org/Qwen-Image-2.1 natively through C++/MLX/Metal. Python
is used only for offline validators and development diagnostics; inference is
performed by the native runtime.

## Supported App/CLI paths

- Text-to-image and image editing with the BF16 Comfy checkpoint.
- RGBA generation, transparent subject extraction, and transparent-layer edits.
- One to ten ordered image references. References are encoded as visual
  conditions; a separate black/white mask is an additional visual reference,
  not hard pixel-preserving inpainting.
- Ellipse/brush annotation references, request-owned prefix KV caching, staged
  or resident GPU execution, and native App job persistence/reuse.
- Experimental PE-T2I and explicitly opt-in PE-I2I FP32 visual conditioning.

The App exposes Qwen-specific examples, ten-reference import limits, RGBA
handling, mask/annotation authoring, and the PE-I2I warning. Use the GPU route
for normal operation. The experimental GPU+ANE route is restricted to the
validated 512² text-to-image configuration and is not hardware-placement
proof.

## 512² validation snapshot

The maintained regression scope is 512². Evidence includes native GPU text
generation. App model selection and prompt examples default to 512×512;
reference encoding geometry is independent of that output canvas. Coverage includes
RGBA export, prefix-cache/session lifecycle, one-step progress
events (`0/steps` through `steps/steps`), and App transport/workflow tests.
The matched 512² GPU comparison against mflux is recorded in
`results/qwen21/matched-refresh-mflux-512-report.json`; native cached denoise
was measured at approximately 1.068× the mflux denoise speed in that matched
workload, with latent relative RMSE 0.007073 and RGB correlation 0.999890.

Additional completed samples exercise 1024² edits, transparent extraction,
annotation/mask edits, three-reference composition, and the ten-reference
runtime boundary. They are representative evidence, not a blanket perceptual
quality gate.

## Correctness and limitations

Run `make test-qwen21` after building the native/App targets. The target runs
focused report, mask-analysis, native contract, and App workflow checks without
starting inference. Full reports and experiment history are in
`docs/status/qwen21-convergence-2026-09-22.md` and the local
`notes/2026-09-21-qwen-image-21.md`. NumPy is an offline-test dependency;
Pillow is only needed for the image-report CLIs, not these unit tests.

Known limitations are intentional and explicit: BF16 Qwen3.5 visual parity
does not pass the strict reference gate, PE-I2I remains experimental FP32,
edit preservation is not pixel-locked, JPEG decoder parity has small byte
differences, and actual ANE occupancy cannot be proven without privileged
hardware tracing. The 2K path is exposed by the model contract but is outside
the maintained test scope; do not use its incomplete experiment as evidence
for production quality or performance.
