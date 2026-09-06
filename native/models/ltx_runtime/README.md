# LTX runtime provenance

This directory vendors the reusable native runtime from the sibling
`ltx-mac` project, based on commit
`f1a47138ea18355df8f3df5d8d2d19cd012dd2ad` plus the synchronized shared-file
changes validated by `tests/native/test_contract.py`. It is intentionally a
mixed-language library, not an Objective-C rewrite:

- C: schedule, Transformer blocks, conditioning, connector, latent I/O;
- Objective-C/Metal: Metal and MPSGraph kernels plus Core ML/ANE bridges;
- C++/MLX: latent upsampler and media codecs.

`ltx_session.mm` owns only the TurboCider ABI, request validation, cache
identity, cancellation, lifecycle and media handoff. Model math should remain
in this runtime. Changes to the shared denoiser path should be implemented in
`ltx-mac` first (or upstreamed there immediately) and synchronized here; do
not add a second TurboCider-specific implementation of the same operator.

The contract test compares the byte-identical shared hot-path sources whenever
the sibling `ltx-mac` checkout is available. TurboCider-only additions, such as
the native Gemma encoder APIs and runtime environment isolation, are tested
separately.

The sibling checkout may be dirty while experiments are in progress. The
vendored directory is the release input; a TurboCider commit must therefore
include this directory and its tests rather than relying on the sibling
working tree at runtime.
