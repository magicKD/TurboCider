# Local environment setup

[Documentation](README.md) · [Getting started](GETTING_STARTED.md) · [FLUX preparation](FLUX_PREPARATION.md)

Use this guide before the quick start on a Mac without a configured development
environment. Run commands from the repository root in the same terminal session.
`make setup` requires an existing **arm64 CPython 3.11**. It installs pinned
Python packages; it does not install Python, Apple developer tools, FFmpeg,
model weights or ANE artifacts.

## 1. Check Apple development tools

The native build requires Apple silicon and a macOS SDK with Swift and Clang.
Check the selected tools before installing Python packages:

```sh
uname -m
sw_vers
xcode-select -p
xcrun --sdk macosx --show-sdk-path
xcrun swift --version
xcrun clang --version
df -h .
```

`uname -m` must report `arm64`; use a native terminal rather than an x86_64
interpreter under Rosetta. If Apple developer tools are missing, run
`xcode-select --install` and complete the macOS installer. A working Command
Line Tools installation was sufficient for the build and tests recorded below;
full Xcode was not required. To explicitly select installed Command Line Tools:

```sh
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
```

The **SDK version and minimum runtime OS are different requirements**. The build
reads the deployment target from `libmlx.dylib` and uses it for the native
package. The MLX wheel used in the recorded setup required macOS 26.2 even
though the SDK was 26.5. Do not lower `TURBOCIDER_DEPLOYMENT_TARGET` to work
around a dependency built for a newer OS; that does not make the library
compatible with an older system.

## 2. Provide Python 3.11

If a native CPython 3.11 is already installed, check it and proceed to setup:

```sh
python3.11 -I -c 'import platform, sys; print(sys.version); print(platform.machine())'
make setup PYTHON="$(command -v python3.11)"
```

The system `python3` may be 3.9, which the setup script rejects. An optional
bootstrap below keeps Python under the ignored `.deps/` directory and does not
replace the system interpreter. It requires an existing `python3` with pip and
network access; the recorded machine used `/usr/bin/python3` for this step.
If pip is unavailable, install uv through its
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
and use that `uv` executable in the second command.

```sh
python3 -m pip --isolated install --disable-pip-version-check \
  --no-cache-dir --target .deps/bootstrap uv==0.12.13

UV_PYTHON_INSTALL_DIR="$PWD/.deps/python" \
UV_PYTHON_BIN_DIR="$PWD/.deps/bin" \
UV_CACHE_DIR="$PWD/.deps/uv-cache" \
  .deps/bootstrap/bin/uv python install 3.11

.deps/bin/python3.11 -I -c 'import platform, sys; print(sys.version); print(platform.machine())'
make setup PYTHON="$PWD/.deps/bin/python3.11"
```

uv downloads a standalone CPython distribution; the `3.11` selector may resolve
to a newer patch release over time. See uv's
[Python installation guide](https://docs.astral.sh/uv/guides/install-python/) and
[directory settings](https://docs.astral.sh/uv/reference/environment/#uv_python_install_dir).
uv is only a bootstrap option, not an inference dependency.

Subsequent Make targets prefer `.venv/bin/python3`, then `.deps/bin/python3.11`,
then `python3.11` on `PATH`. `PYTHON=/absolute/path/to/python3.11` overrides this
choice. For commands run directly outside Make, use the explicit `.venv` Python
or set the terminal's path:

```sh
export PATH="$PWD/.venv/bin:$PWD/.deps/bin:$PATH"
```

## 3. Understand the managed environments

Setup creates two environments and runs `pip check` in both:

| Location | Contents and purpose |
|---|---|
| `.venv/` | Packages pinned in `tools/dependencies/build.lock.txt`; native MLX headers/libraries, Python development tests and offline tools |
| `~/Library/Application Support/TurboCiderNative/toolchains/coreml/` | Packages pinned in `tools/dependencies/coreml.lock.txt`; separate offline Core ML export environment, without MLX |

The lock files are the authoritative version lists. Keep their pins together;
installing the latest `mlx` or `coremltools` separately can change compatibility.
The packages have these roles:

| Package | Role in this setup |
|---|---|
| `mlx`, `mlx-metal` | MLX C++ headers, native libraries and Metal backend; also used by offline tensor tools |
| `coremltools` | Build MIL graphs, compress weights and export Core ML partitions offline |
| `numpy` | Offline tensor manipulation, weight conversion and numerical checks |
| `protobuf` | Core ML model representation and serialization used by coremltools |
| `sympy`, `mpmath` | Symbolic shape expressions and supporting mathematics used by coremltools |
| `attrs`, `cattrs` | Structured Python data and conversion support pulled in by coremltools |
| `packaging` | Version and compatibility handling for coremltools |
| `pyaml`, `PyYAML` | YAML serialization support pulled in by coremltools |
| `tqdm` | Progress reporting in offline tools and their dependencies |
| `typing_extensions` | Python typing compatibility support required by cattrs |

The native App/CLI uses the linked MLX and Apple frameworks, not a Python model
worker. Core ML, Metal, Foundation, AVFoundation and the other Apple frameworks
come from macOS / its SDK; they are not pip packages. PyTorch, Diffusers, mflux,
ComfyUI and a Hugging Face Python client are not required for this native FLUX
preparation path. Optional external comparison tools have separate requirements.

Check the environments without loading a model or starting a GPU workload:

```sh
.venv/bin/python3 -m pip check
"$HOME/Library/Application Support/TurboCiderNative/toolchains/coreml/bin/python3" -m pip check
TC_MLX_ROOT="$(.venv/bin/python3 -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/mlx")')"
test -f "$TC_MLX_ROOT/include/mlx/mlx.h"
test -f "$TC_MLX_ROOT/lib/libmlx.dylib"
test -f "$TC_MLX_ROOT/lib/libjaccl.dylib"
test -f "$TC_MLX_ROOT/lib/mlx.metallib"
otool -l "$TC_MLX_ROOT/lib/libmlx.dylib" | sed -n '/LC_BUILD_VERSION/,+5p'
```

Build scripts discover MLX from `.venv` unless `MLX_ROOT` explicitly selects
another compatible installation. If using an override, keep its headers,
libraries and Metal resources from the same build.

For a different offline-export location, invoke setup directly with
`--app-toolchain /absolute/toolchains/coreml`; use that environment's Python
explicitly for export. This does not configure a Python runtime inside the App.

For installation without network access, first download both lock files' wheels
on a compatible Apple silicon Mac with CPython 3.11, then transfer the wheelhouse:

```sh
python3.11 -m pip download --only-binary=:all: \
  --dest .deps/wheelhouse \
  -r tools/dependencies/build.lock.txt \
  -r tools/dependencies/coreml.lock.txt

python3.11 tools/setup_dependencies.py --offline --wheelhouse .deps/wheelhouse
```

The offline target still needs Python and Apple tools already installed. Model
downloads and ANE preparation remain separate steps.

## 4. Add media tools when needed

`make setup` does not install **FFmpeg or ffprobe**. The video quality tests need
both; the video timing test needs ffprobe and Apple developer tools. Native
Apple video output uses AVFoundation, while H3's external media paths also
invoke FFmpeg/ffprobe. They are not needed for basic FLUX image generation.

If you already use Homebrew, its
[FFmpeg formula](https://formulae.brew.sh/formula/ffmpeg) provides an installation
route:

```sh
brew install ffmpeg
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders
```

For the repository's generated video fixtures, verify `libx264` encoding, H.264
decoding, MP4/MOV containers, RGB24 raw video, and the lavfi color/concat/format/
scale filters. A binary named `ffmpeg` alone does not guarantee these features.
The recorded setup used a local FFmpeg 9.0.1 build with static x264 and exposed
both binaries through `.deps/bin`. That minimal build covered these tests;
audio and other codecs need a build with the corresponding features. x264 and
pkgconf were build dependencies of that local FFmpeg, not Python dependencies
or additional requirements for building the TurboCider image engine.

Make prepends `.deps/bin` to `PATH`. For direct CLI runs, set the terminal path
as shown above and check `command -v ffmpeg` and `command -v ffprobe`.
Finder-launched Apps do not inherit a terminal's path changes; configuring one
shell is not sufficient to make an external media tool discoverable in every
launch context.

## 5. Build, package and verify in layers

```sh
make package
dist/cli/turbocider doctor
dist/cli/turbocider self-test
make test
make test-app
make test-library
make test-api
open dist/TurboCider.app
```

`make package` includes the build. It produces `dist/TurboCider.app` and
`dist/cli/`; retain the complete CLI folder, including its dynamic libraries,
Metal resources and helpers. Packages are locally ad-hoc signed, not notarized.
`make build` alone produces the development executables under `build/native/`.
`make test` does not build these executables or run the separate App, library
and API targets; keep the build-before-test order above.

Use a macOS session with Metal and AVFoundation access. App tests exercise the
clipboard; library and API tests use local sockets. Permission or device-access
failures in a restricted runner should be diagnosed separately from missing
packages. Read test output for skips. With the pinned packages and media tools
installed, these extra fixtures remain outside the basic FLUX preparation:

| Optional check | Additional requirement |
|---|---|
| Wan LoRA preflight fixture | `TURBOCIDER_WAN_TEST_MODEL` directory, or default `models/Wan2.1-1.3B-QAD`, containing `mlx_dit.safetensors` and `mlx_dit.json` |
| Gemma4 tokenizer | `models/LTX-2.5/gemma4-12b-ltx-v1/tokenizer.json` |
| Gemma4 checkpoint | `models/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| NVFP4 GPU test | `TURBOCIDER_TEST_GPU=1`, numpy, the optional `safetensors` Python package, Metal access and a separately built `build/native/nvfp4-probe`; the default build does not create this probe |

Neither `doctor`, `self-test` nor the default test targets prove that a real
model generates successfully. Model-backed validation (`make test-model`) needs
weights and runs actual inference. Continue with
[FLUX model and ANE preparation](FLUX_PREPARATION.md), then establish a GPU
output before comparing hybrid inference.

## Troubleshooting gaps found during setup

| Symptom | Cause and action |
|---|---|
| `python3.11` missing, or setup rejects Python 3.9 | Install CPython 3.11 first; pass `PYTHON` explicitly. Make does not bootstrap an interpreter. |
| `No module named encodings`, with a standalone Python reporting `/install` as its prefix | Copying the interpreter into a venv can lose its standard-library location. Setup now creates venvs with `symlinks=True`; it does not rebuild an existing venv. Inspect the failed environment and recreate only the confirmed project-owned environment with a working base Python and `--symlinks`, then rerun setup. Do not clear an unrelated environment. |
| An environment breaks after moving the checkout | Venv executables and configuration refer to the base interpreter. If Python lives in `.deps`, moving or deleting it can also break the separate Core ML environment. Recreate affected owned environments at the final location. |
| Setup refuses a nonempty directory | It will not adopt a non-venv directory or a symlink destination. Choose an empty dedicated toolchain directory rather than overwriting existing contents. |
| MLX headers or dylibs missing | Run setup with the correct interpreter; inspect `.venv` or the explicit `MLX_ROOT`. Python importability alone is not the C++ build check. |
| `stdlib.h` / `dirent.h` missing in a C test despite installed tools | Resolve Clang and the macOS SDK through `xcrun`; the H3 quant-cache test now passes the SDK explicitly with `-isysroot`. Installing unrelated Python packages will not fix SDK selection. |
| Linker cannot find `video.o` / `audio.o` while building LTX helpers | The build now references the flattened names `native_media_video.o` / `native_media_audio.o`. Use the corrected build script and rebuild; this is not a missing media package. |
| Downloaded weights exist but MLX rejects a file symlink | Managed downloads now publish regular-file hard links to verified blobs. Reinstall through the updated library to reuse verified cached blobs; see [model library storage](MODEL_LIBRARY.md#location-and-registration). |
| Video checks are skipped | Install both media tools and check their visibility and codec features; also inspect any separately reported missing model fixtures. |

## Recorded preparation, not an inference benchmark

The preparation below was recorded on 2026-09-14 against `main` at `d742f7d`
with the setup and download fixes described in this guide. It used an Apple M5 Pro with 24 GiB unified
memory, macOS 26.4.1, Command Line Tools, SDK 26.5, Swift 6.3.2, Clang 21,
CPython 3.11.16 and the checked-in MLX 0.32.0 / coremltools 8.3.0 lock files.
Build, packaging, CLI doctor/self-test, a small Metal matrix operation and a
small Core ML export/compile check passed. The recorded `make test` run had
126 passing unittest cases and four optional skips, plus passing standalone
native checks; App, library and API test targets passed too.

The FLUX weights and 20 ANE partitions were also prepared and their requests
passed planning. No real model generation, measured peak memory, image-quality
comparison or M5 performance result is established by that preparation. Disk
space for weights/artifacts is separate from the RAM needed during inference.
