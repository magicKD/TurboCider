# Cider kernel attribution

Source: https://github.com/Mininglamp-AI/cider

Pinned commit: `4d91fcee9439f7aea17ae6e965271d9536c604a0` (MIT; see LICENSE).

This directory vendors only the per-channel W8A8 primitive and its three Metal kernels. The kernels and public primitive header are unchanged. The primitive's file reader uses embedded kernel sources so installed applications need no source checkout or environment variable. It is compiled without Objective-C ARC, matching upstream.

TurboCider's adapter pads single-row inputs to 32 rows to keep activation quantization enabled, packs weights with symmetric FP32 absmax scaling, and owns checkpoint-bound persistent caches outside the upstream code.
