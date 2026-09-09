# Native framework refactor summary — 2026-09-09

## Scope delivered

- Production App/CLI inference uses C++ / Objective-C++ / MLX / Core ML.
  Python exporters, merge tools and reference workers remain development-only;
  sd.cpp is confined to `tools/validation/sd_cpp` and excluded from release targets.
- Shared Qwen3 implementation serves FLUX Klein and Z-Image with explicit
  model-specific hidden-state taps and precision policies. No shared prompt
  cache is inferred between models.
- `native/components/text` contains Qwen3/UMT5, `components/diffusion` contains
  Wan sampling/geometry/noise, `components/vae` contains TAEHV, and
  `components/weights` contains affine weight operations. Model orchestration
  remains under `native/models`; Apple runtime/media adapters stay separate.
- Wan 2.1 1.3B QAD replaces the FastMetal production Python session and naming.
  Old ABI/schema strings are read-only compatibility aliases, not external
  runtime discovery. Unsupported formats/options fail explicitly.
- H3/LTX/Wan use verified premerged LoRA artifacts. FLUX/Z-Image retain their
  supported native in-memory/low-rank LoRA modes. This is not a universal ban
  on runtime LoRA.
- Native GGUF support is limited to implemented floating/Q8_0/Q4_0/Q4_1
  tensors; old sd.cpp streaming and mixed K-quant support are not advertised.

## Evidence and limits

- Native library, CLI, Swift App/integration tests build; repository/contract
  and model-independent tests run. System-Python NumPy skips were rerun with
  the development Python: both modules passed (10 tests, 2 subtests).
- Actual ad-hoc signed App/CLI package contains no Python scripts. Its CLI
  generated an exact-match LLaDA PNG outside the repository cwd with cleared
  environment variables. The host/model files were not a clean-machine setup.
- Wan real checkpoint: all 46 small-fixture DiT/sampling stages exact against
  Python MLX, including sequential three-step eager/compiled rollouts.
- Full native 832x480x81 generation succeeds. Eager 75.112 s, compiled
  73.2908 s; decoded outputs meet existing aligned-RGB thresholds but are not
  pixel-exact. INT8 hybrid 78.5605 s uses a lower MLX allocator peak but is not
  a speedup; it remains explicit and approximation-opt-in.
- Repeated 5-frame ABBA warm medians: eager 2.96943 s, compiled 2.71741 s.
  These compare current modes on one Mac, not every pre/post-refactor model.
- UMT5 normalized conditioning differs from PyTorch (cosine ~0.999825).
  Small tensor parity, high cosine and successful generation alone do not
  establish universal video quality parity.
- Full old-Python-reference versus native eager validation also passed the
  existing aligned-RGB gate at 832x480x81, same prompt/seed: mean correlation
  0.998410, minimum frame correlation 0.992720, mean cosine 0.999415,
  mean MAE 3.08766/255, maximum motion relative error 0.127310. This qualifies
  this workload, not all prompts or models. Reference took 78.9991 s and the
  final native run 76.0669 s; these single runs are not a general speed claim.
- Fixed AVFoundation's default 600-tick timebase rounding 16 FPS timestamps.
  Native output now has exact 16/1 FPS, 81 frames and 5.0625 s duration.
  Native-writer regression passed all six combinations of 16/24/30 FPS and
  5/81 frames. The test requires host media access (sandbox writer startup
  failed; the same test passed outside the sandbox).

## Handoff boundary

This is a coherent native-framework refactor suitable for a development commit,
not a claim of universal performance non-regression, all-LoRA qualification,
notarization or new-machine GUI acceptance. Model weights/build outputs are
not committed. See `wan-native-session-2026-09-09.md` for scoped measurements
and `production-tool-boundary-2026-09-09.md` for packaging evidence.
