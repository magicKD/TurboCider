# SeedVR / SeedVR2

The native SeedVR2 model geometry and computation in
`native/models/seedvr2/` adapt the published ByteDance model architecture to
C++/MLX. The files are materially changed: native GPU operations replace
PyTorch, attention windows are batched by shape, and causal VAE histories use
owned channel-last MLX tensors. The low-precision interface is independently
checked against an external ComfyUI installation.

Upstream: https://github.com/ByteDance-Seed/SeedVR

Architecture reference revision:
`e4de8c24441a67e1b7df56abea10645059bb1185`.

Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.

Licensed under the Apache License, Version 2.0. A copy is included in
[`Apache-2.0.txt`](Apache-2.0.txt).

Model weights are downloaded separately and retain their upstream license.
`Comfy-Org/SeedVR2` repackages the original ByteDance weights and conditioning
for ComfyUI-compatible safetensors storage. No ComfyUI runtime code is bundled
with the production native executor. ComfyUI is used as an independent,
separately installed validation oracle only.
