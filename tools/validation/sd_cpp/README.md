# Developer-only stable-diffusion.cpp comparisons

These optional tools are not dependencies of TurboCider, its build, or its App.
No sd.cpp binary or library is packaged. The installer retains a pinned,
SHA-256-verified reference in the ignored `.deps/stable-diffusion-cpp` tree.

- `install.py`: install the optional reference; preserves the upstream LICENSE.
- `benchmark_reference.py`: compare native safetensors models against sd.cpp.
- `compare_native.py`: resident native MLX GGUF versus the reference server.

Example (developer environment only):

```sh
python3 tools/validation/sd_cpp/install.py
python3 tools/validation/sd_cpp/compare_native.py \
  --model-root /path/to/checkpoints --variant Q8_0 \
  --component-root /path/to/components \
  --library build/native/libturbocider.dylib \
  --server .deps/stable-diffusion-cpp/current/sd-server \
  --output /tmp/tc-native-gguf-comparison
```

The component root must contain tokenizer/, split_files/text_encoders/ and
split_files/vae/. The comparator creates a temporary explicit model layout;
it does not inject production backend-selection environment variables.

The old TurboCider sd.cpp resident/streaming bridge and its benchmark were
removed. Historical validation JSON remains in docs/design/validation as
historical evidence, not a claim about the current native runtime.

Reference provenance: stable-diffusion.cpp, MIT, Copyright (c) 2023 leejet;
pinned Unsloth build metadata and archive checksum are in install.py.
