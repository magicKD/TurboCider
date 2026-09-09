# LTX runtime provenance

This directory vendors the reusable native runtime from the reference LTX
runtime, based on commit
`f1a47138ea18355df8f3df5d8d2d19cd012dd2ad` plus the synchronized shared-file
changes validated by `tests/native/test_contract.py`. It is intentionally a
mixed-language library, not an Objective-C rewrite:

- C: schedule, Transformer blocks, conditioning, connector, latent I/O;
- Objective-C/Metal: Metal and MPSGraph kernels plus Core ML/ANE bridges;
- C++/MLX: latent upsampler and media codecs.

`ltx_session.mm` owns only the TurboCider ABI, request validation, cache
identity, cancellation, lifecycle and media handoff. Model math should remain
in this runtime. Changes to the shared denoiser path should be implemented in
the reference runtime first (or upstreamed there immediately) and synchronized here; do
not add a second TurboCider-specific implementation of the same operator.

The default contract test verifies the committed shared hot-path snapshot
against `SOURCE_MANIFEST.json`. This records both the upstream base and the
TurboCider snapshot, including its adaptations; it is not a claim of exact
upstream parity. When synchronizing source changes, update the manifest in the
same reviewed change and run the corresponding numerical tests. Developers can
also set `TURBOCIDER_LTX_REFERENCE=/absolute/reference-checkout` to require a
byte-identical comparison against that explicit checkout. TurboCider-only additions, such as
the native Gemma encoder APIs and runtime environment isolation, are tested
separately.

The sibling checkout may be dirty while experiments are in progress. The
vendored directory is the release input; a TurboCider commit must therefore
include this directory and its tests rather than relying on the sibling
working tree at runtime.
