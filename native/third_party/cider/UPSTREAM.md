# Cider kernel attribution

Source: https://github.com/Mininglamp-AI/cider

Pinned commit: `4d91fcee9439f7aea17ae6e965271d9536c604a0` (MIT; see LICENSE).

This directory vendors the per-channel W8A8 primitive and its Metal kernels. The primitive's file reader uses embedded kernel sources so installed applications need no source checkout or environment variable. It is compiled without Objective-C ARC, matching upstream.

TurboCider's adapter pads single-row inputs to 32 rows to keep activation quantization enabled, packs weights with symmetric FP32 absmax scaling, and owns checkpoint-bound persistent caches outside the upstream code.

The separate original-code ConvRot integration adds `ConvrotW8A8Linear` and
`convrot_quantize.metal`. It consumes the checkpoint's signed INT8 codes and
FP32 row coefficients without requantizing weights. Activation quantization
accepts BF16/FP16/FP32 directly, after the caller's Comfy H256 transform. The
existing INT8 TensorOps GEMM is templated to add an FP32, bias-free epilogue;
historical FP16 entry points retain their original behavior. The new route
also quantizes single-row activations, pads physical M/N tails before GEMM,
and exposes monotonic GPU-encoding counters. The request owner selects this
route by hardware capability (automatic on M5 and later with macOS 26.4+), with explicit
overrides for comparison. The integration alone is not a claim of whole-model
quality or speed qualification.

The integer TensorOps OS boundary follows Apple's
[M5/A19 GPU Tech Talk](https://developer.apple.com/videos/play/tech-talks/111432/).
M4 and earlier, or older supported macOS releases, retain the floating-activation
ConvRot path. `TURBOCIDER_QWEN21_CONVROT_W8A8=0` selects that reference explicitly;
`=1` requires integer TensorOps. `_ENCODER` and `_DIT` component overrides allow
independent ablations. An explicit request on unsupported hardware fails instead
of silently claiming W8A8 execution.
