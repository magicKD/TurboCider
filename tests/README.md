# Testing TurboCider

Run commands from the repository root. Start with the
[environment setup](../docs/public/ENVIRONMENT_SETUP.md) and
[contribution guide](../CONTRIBUTING.md). `make help` lists the maintained targets;
[the Makefile](../Makefile) defines their exact commands and prerequisites.

## Test layout

| Location | What belongs here |
| --- | --- |
| [repository/](repository/) | Layout, dependency isolation, build entry points and C/C++ boundaries |
| [native/](native/) | Python contracts, host fixtures, C/C++/Objective-C probes and opt-in model/device checks |
| [integration/](integration/) | Swift App, CLI, library and service behavior tests with native bridges |
| [comfyui/](comfyui/) | Gateway fixtures and node contracts using real CPU tensors |
| [fixtures/](fixtures/) | Small portable inputs; no model weights or generated media |

A file under `native/` can require a built library, optional Python packages,
Metal access or real weights. Read its prerequisites and choose the relevant
Make target or focused test; the directory name alone does not imply CPU-only.

## Build and run

After setting up dependencies:

```sh
make build
make test
```

`make test` combines repository, host and native contract checks. It requires the
project dependency environment and native build; it is broader than the quick
repository check below. For Swift changes also run `make test-app`; for shipping
changes follow the packaging requirements in the contribution guide.

Make prefers `.venv/bin/python3`, then `.deps/bin/python3`, and otherwise the
active Python. Set `PYTHON=/path/to/python3` explicitly when using another local
environment. Missing packages or unavailable hardware are validation gaps,
not passing tests; report failures and skips with their reasons.

## Choose a focused check

| Area | Entry point | Prerequisites and scope |
| --- | --- | --- |
| Repository structure | `python3 -B -m unittest discover -s tests/repository` | Git and Clang C/C++ toolchain; compiles tiny policy fixtures, no model weights |
| H3 download helpers | `make test-h3-download` | Local bytes and mocked responses; no model downloads |
| Validation file and metric helpers | `make test-validation-helpers` | Standard library and NumPy; SeedVR2 file I/O and H3 VAE arithmetic, no models or GPU |
| Streaming evidence and campaigns | `make test-streaming-campaign` | Standard library, temporary files and synthetic worker processes using local sockets; no model weights |
| Streaming layout and ownership | `make test-streaming-host` | Project environment and compiler; synthetic fixtures, no full model inference |
| Streaming API contracts | `make test-streaming-contract` | Native build and its dependencies |
| Runtime ANE host contracts | `make test-runtime-ane-host` | Project environment and host fixtures; no model inference |
| App state and interaction | `make test-app` | Built Swift test executables, macOS session and clipboard access |
| Model library and downloads | `make test-library` | Built test executables and tiny local HTTP fixtures |
| Local API lifecycle | `make test-api` | Built executables, local Unix sockets and child processes |
| ComfyUI nodes | `make test-comfyui` | Existing torch/NumPy/Pillow environment; use `COMFYUI_PYTHON` for node tests |

Native contract tests that support `TURBOCIDER_TEST_NATIVE_DIR`, such as
[test_contract.py](native/test_contract.py), can select an isolated matching build:

```sh
TURBOCIDER_TEST_NATIVE_DIR=build/native \
  .venv/bin/python3 -B tests/native/test_contract.py
```

Keep the selected library, its manifest and source revision together. Even tests
using MLX CPU tensors may need Metal device visibility during initialization;
`test-runtime-ane-cpu-bridge` documents that boundary in the Makefile.

For an isolated App build, use the same directory for compilation and behavior
tests. The default remains `build/native`:

```sh
TURBOCIDER_BUILD_OUTPUT_DIR=build/release-check make build-app
make test-app NATIVE_TEST_OUT=build/release-check
```

LTX snapshot, streaming-contract and finalizer-envelope tests also honor
`TURBOCIDER_TEST_NATIVE_DIR`; the finalizer alone can be overridden with
`TURBOCIDER_LTX_FINALIZER_BINARY`. This avoids testing an older helper left in
the default build directory. `make test-ltx-optimization` checks PrunaVAED
convolution/fusion math, Sol route accounting and spatial K/V pooling with
small fixtures. `tests/native/test_ltx_w8a8.py` additionally checks projection
workspace lifetimes, fused normalization/RoPE and rotation/quantization against
independent oracles. These GPU checks establish correctness, not model speed.

## Device and model checks

`make test-runtime-ane` runs small synthetic Core ML/MLX device work.
`make test-streaming-metal` and `make test-tiny-vae-metal` also exercise real
Metal resources; their success does not establish model quality or performance.
Real model acceptance is explicit, for example:

```sh
make test-model MODEL=/absolute/FLUX.2-klein-4B OUTPUT=/absolute/new-test-output
```

Use a fresh output directory and the fixture/model expected by the target.
Record the device, source/library identity, command, test count, failures and
skips. Keep raw logs, tensors and media in ignored local output directories;
portable experiment narratives belong in
[docs/experiments/](../docs/experiments/README.md). Performance work additionally
follows the [measurement methodology](../docs/public/PERFORMANCE.md).
