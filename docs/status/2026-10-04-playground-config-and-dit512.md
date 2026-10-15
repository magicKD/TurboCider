# Playground configuration and Qwen reference-512 DiT — 2026-10-04

This fixes the installed client's configuration flow on `dev-local` (baseline
`1ad0cfa`). `dev-verify` was an older checkout without the installed Playground
implementation and was left unchanged. No model was downloaded and no Core ML
model was exported. Experiments used the local Qwen-Image-2.1 checkpoint, serially.

## Corrected behavior

- 512×512 **output** was already supported. The defect was an unnecessarily
  restrictive combination guard: reference encoding size 512 and named DiT cache
  presets could not be used together. Swift admission, native admission and
  discovery metadata now agree for GPU base or one ordinary runtime Transformer
  LoRA, 20–40 steps, 512×512 output and 1–3 references.
- This is an approximation opt-in. Reference encoding and resizing an imported
  image remain separate from output dimensions. Viggle six-step remains excluded
  from DiT cache; the existing r128 fast512 route stays six steps/strength1/cacheoff.
  r256 fast512 and ANE combinations are not newly enabled.
- New image-model canvases and new Playground templates start at 512×512. Video
  retains model defaults. Saved drafts/history and explicit custom sizes survive.
- Playground has a collapsible right inspector with local model path, dimensions,
  ratio presets, steps, seed, residency, LoRA enable/strength/strategy, DiT presets
  and reference encoding. Settings are isolated per template and autosaved.
  Creation synchronization is explicit and retains template inputs/instruction.
- Adding/enabling Turbo does not silently rewrite sampling parameters. An explicit
  preset resolves the mismatch. Invalid cache/size/LoRA combinations explain the
  issue and provide recovery controls. Edits are locked during import/submission.
- New Playground setup from a non-Qwen Creation draft previously inherited generic
  `resident` memory policy. It now applies Qwen `component_staged` defaults. Changing
  a legacy non-Qwen template also clears incompatible video, streaming, device,
  prompt-enhancement and cache settings while preserving its role inputs,
  instruction, seed, model directories and per-model adapters.

## Memory behavior

This machine is an M4 Pro with 48 GiB memory. Creation and all five existing
Playground drafts already use `component_staged`; they were preserved. New Qwen
model selection also uses staged residency. The new inspector makes it visible.

`native/models/qwen21/pipeline.cpp` releases previous DiT/VAE before text encoding
(lines326–330), evaluates and releases local text/vision encoder weights
(lines332–356), then loads DiT/VAE (line441). Full staged generation releases DiT
before decode (lines1062–1074) and VAE afterward (lines1105–1109); exception cleanup
honors the same policy. VAE may coexist with DiT during denoising; bounded
conditioning/KV caches may remain. Explicit `prepare(false)` retains prepared
DiT/VAE by contract. This is not a claim that every allocation is always zero.

## Automated and UI acceptance

The final native manifest matches 434 source inputs. Build used Command Line
Tools and macOS15.2 SDK because the configured Xcode launcher has a CoreDevice
linker error. The package minimum OS remains inherited from the existing MLX
binary (26.2). Final Swift App and selected tests compiled successfully.

- 7 native contract methods, including 144 positive v1/v2 base/ordinary LoRA,
  cache-mode, step-count and reference-count combinations plus negative guards.
- 1 real isolated RPC reference512 admission test.
- 1 actual tiny MLX GPU Transformer lifecycle test: base and a real tiny LoRA,
  1–3 reference geometries, three cache presets, warmup/skip/full-step control,
  reference content/order/shape reset and post-destruction memory recovery.
- 47 additional low-load tests: shared workflow C ABI/CLI, offline discovery,
  Python client, real RPC and CPU image preparation, installations/helper
  lifecycle, real service queue/history with test-only CPU engines. Zero skips.
- Four Swift executables pass: Qwen reference sizes/default canvases; Playground
  independent drafts, five workflows, import rollback/cancellation, resizing,
  migration/persistence and editable configuration; Qwen LoRA/DiT/history and
  annotation contracts; editing canvas selection/alignment/locking.
- Real final App UI: staged memory label; imported-image preview; fast512 + cache
  +25-step submission readiness; Turbo mismatch blocker and explicit recovery.
  Inspector collapse/expand, per-template custom-size persistence and incompatible
  cache recovery were also exercised. Large and narrow layouts were inspected.

UI verification did not submit a new generation. The same native build is used
for the separate full-model CLI pair below. User drafts were restored after
checking that only known test edits differed. User jobs were unchanged and the
single owned imported test image was archived in the verification directory.

Evidence directory: `outputs/playground-config-20261004/`, including original
failure logs and corrected final runs. Initial fixture failures (base request
incorrectly declaring LoRA strategy and invalid reversed text slots) were fixed
in tests; they were not suppressed. The first package build's source-identity
check detected concurrent metadata changes; affected native objects were rebuilt,
relinked and the final source manifest was verified before acceptance.

## Full-model pair

The serial 25-step base edit pair passed: 512×512 output, fast512 reference
encoding, seed42, GPU, staged residency and one identical red-teapot input.
Both results report 25 actual steps and 1024 reference tokens. Each request ran
in a fresh process after compilation/tests finished; OS/file caches were not
flushed, so this is a single sequential timing screen.

| Mode | Request wall | Denoise | MLX peak | MLX active after request |
| --- | ---: | ---: | ---: | ---: |
| Off | 67.217 s | 58.575 s | 15.81 GiB | 0.547 MiB |
| Balanced | 46.864 s | 39.572 s | 15.81 GiB | 0.547 MiB |

Balanced reused 11 steps and skipped 264 middle-block evaluations (warmup8,
front8, threshold0.25, max-consecutive2). End-to-end ratio was 1.434×; denoise
ratio 1.480×. RGB MAE was 1.100/255 and PSNR42.275dB versus cache-off. Source and
both outputs were inspected: both recolor the teapot dark cobalt/navy, with
similar silhouette/composition and small differences in texture/highlights.
Both change some wall/table/material texture relative to the source; this
comparison does not establish exact editing preservation.

MLX memory statistics exclude Core ML, OS and file caches. The first output
inspection harness failed because the project Python lacked Pillow after
successful generation. The completed output was retained; the already-installed
bundled Python inspected it and ran only the remaining balanced case. There
were exactly two full-model generations, without dependency downloads.

## Limits

This acceptance targets the changed Qwen/Playground configuration and shared API
routes, not every model, video route, OS or possible input. The tiny-LoRA GPU and
full-model base pair do not independently qualify every ordinary LoRA file at
fast512. Fast reference encoding and DiT caching can change details. A single
image comparison cannot establish broad fidelity or a universal speedup.
