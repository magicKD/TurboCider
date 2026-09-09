# Production / offline tool boundary — 2026-09-09

The native Core ML resource API no longer launches Python. It retains inventory,
native compilation and cache cleanup. An `export` request returns an offline-only
error. Studio selects exported manifests instead of Python executables. Legacy
saved exporter settings remain decodable but are not forwarded to native APIs.

The CLI no longer discovers or launches `prepare_lora.py`; the retired command
returns a migration message. Use `python3 tools/native/prepare_lora.py ...` in a
development environment. Both Core ML exporters and H3/LTX LoRA merge tools remain
in the repository, including provenance and output-identity checks.

The packaging recipe copies neither Core ML Python exporters nor LoRA preparation
scripts to the App or CLI release. Native helper executables for media handling
and H3 quantization remain; removing Python does not mean removing native helpers
or system frameworks. Model weights are still supplied separately.

This supersedes the transitional CLI-script packaging and App-export observations
in `lora-native-boundary-2026-09-09.md`. It is a source/build boundary change, not
evidence of notarization, fresh-machine GUI qualification, or all-model numerical
and performance parity. Those requirements remain open.

## Actual package smoke

Rebuilt App/CLI package passed ad-hoc codesign verification; minimum macOS 26.2.
Inspection of both release trees found no Python scripts. Packaging regenerated
the existing `dist/TurboCider.app` and `dist/cli` build artifacts.

From `/private/tmp`, with `env -i PATH=/usr/bin:/bin`, the App's bundled CLI
generated the LLaDA fox fixture (256x256, seed 42, four steps), using the explicitly
supplied model directory. Request wall was 11.93938425 seconds, MLX peak
45,973,308,210 bytes, backend `mlx_cpp_metal`, graph `compiled_single_blocks`.
GPU access required sandbox escalation. This is a one-shot smoke, not a paired
performance benchmark; it does not prove no regression.

The host still has development Python installed, and model files were supplied
from the existing repository model directory. This does not replace a clean-Mac
GUI test or fully copied model-package validation.
