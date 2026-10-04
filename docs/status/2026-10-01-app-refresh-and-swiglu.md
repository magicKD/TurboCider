# App refresh and complete SwiGLU candidate — 2026-10-01

This continues the laptop goal after `1fe3b49`; it does not close the remaining
Runtime ANE/GPU A8W8 or encoder optimization work. Existing Qwen/Viggle weights
only; one actual App generation and two small synthetic Core ML geometries ran
serially. No checkpoint or dependency was downloaded.

## Editing preferences no longer block text generation

Previously, selecting fast 512 reference encoding in editing and then switching
to text generation left an irrelevant validation error and disabled Generate.
The draft now retains that editing preference while generation requests use
the standard reference field, no reference inputs, and no approximation flag
caused by an inactive editing preference. The inactive preference no longer
locks the canvas, sampling hint, import limit or DiT cache. Switching back to
editing restores the preference and its original compatibility checks.

The focused Swift regression covers mode round trips, old drafts, persistence,
configuration import, history reuse, invalid saved editing values, independent
Playground requests and the native v1/v2 request fields. It passed. The selector
is hidden during generation; explicit recovery remains available in editing.
Isolated native UI checks confirmed enabled Generate in both modes, preserved
512 selection on return to editing, and settings collapse/reopen.

A real click on Generate after the mode switch produced a coherent red ceramic
teapot on a pale wooden table: 512×512, six-step Viggle r128, strength 1, GPU,
27.08 s App elapsed. The saved request has `qwen21_reference_size=1024` and no
inputs; the draft still has the editing preference 512. Completion restored the
controls and showed the output/history thumbnail. This is a functional request,
not a new speed benchmark. Evidence: `outputs/app-refresh-20261001/`.

## LoRA registration survives a busy library

The creation page could add a LoRA to its draft while a library inspection or
refresh was running. The controller previously discarded the registration at
its busy guard, so the App could use a file that the AI `installations` action
could not discover in the registry.

LoRA additions now enter a FIFO and run serially through registration and index
refresh. Equal model/path requests are coalesced while pending or active; the
same path under a different model remains a distinct registration. Failure does
not discard later items or hide earlier registration errors. An item binds to
the current root when it begins; additions during a library switch follow the
resulting root. Cancel clears pending registrations and reports that incomplete
items may be added again; already completed registrations remain.

`ModelLibraryTests.swift` contains suspended metadata-runner fixtures for the
busy/failure race, ordering, duplicate/model identity, index failures, directory
switching, cancellation and retry. No model inference or external registry is
needed for these controller checks. Exact build/run logs and final installation
checks are recorded in `outputs/app-refresh-20261001/`.
The focused controller target and its existing catalog/model-switch checks
passed; the final Swift App and model-library helper build also passed.

## Complete dynamic SwiGLU research graph

The isolated candidate now includes gate, up and down projections with runtime
weights, shared input rotation, per-row normalization, the nonlinear hidden,
and all three LoRA branches. Gate/up corrections use original X before SiLU;
down LoRA uses the corrected hidden in its original basis. The down base
projection applies the same Hadamard transform to hidden and weights after the
nonlinearity. No LoRA is merged into a checkpoint or runtime weight slot.

FP16 and Q/DQ variants share FP32 restoration, nonlinear/hidden normalization,
and unquantized down-LoRA accumulation. The original hidden is also returned
for auditing. CPU gate/up low-rank preparation, all input rotation/normalization,
Python bridging and the complete prediction are included in measured total cost.
The candidate remains outside the App/native runtime admission contract.

Five CPU mathematical tests passed, including an independent float64 oracle,
incorrect LoRA-order/basis counterexamples and a finite hidden whose rotated
scale exceeds FP16. Three additional bridge tests passed. The first Core ML
attempt failed because the probe incorrectly required NumPy FP16 output.
Apple's [Python bridge implementation](https://github.com/apple/coremltools/blob/main/coremlpython/CoreMLPythonUtils.mm)
converts an FP16 MLMultiArray into a float32 NumPy array. The corrected probe
requires both the model description and MIL outputs to remain FP16 and requires
every returned float32 value, including signed zero, to round-trip through FP16
exactly. Nonrepresentable float32, float64, nonfinite values or wider declared
interfaces still fail; no numerical tolerance was relaxed. The failed report
was retained, and retry records proved exact promotion for both outputs.

| Rows / hidden / width; group; rank | FP16 total | Q/DQ total | FP16 / Q/DQ prediction | Maximum output relative L2 vs unquantized math (FP16 / QDQ) |
|---|---:|---:|---:|---:|
| 32 / 64 / 96; 32; 4 | 0.710 ms | 0.653 ms | 0.284 / 0.261 ms | 0.227% / 1.538% |
| 128 / 512 / 768; 64; 8 | 18.125 ms | 18.051 ms | 3.329 / 3.420 ms | 0.643% / 1.971% |

Each variant has only two warmed ABBA timing samples. These diagnostic figures
do not establish a stable speedup. All five MatMul operations in each variant
preferred CPU in the compute plan; a preference is not a hardware trace. The
medium case spent about 81% of total time in CPU preparation, and its 1.004×
total ratio provides no useful acceleration evidence. A/B/A runtime weight and
adapter switching returned identical A outputs; base and zero cases passed.
This is numerical/component evidence, not image quality or hardware INT8 proof.

The two complete probes took 1.37 s and 2.07 s including conversion/compilation,
numerical cases and timing. Their explicit temporary graphs were removed and
source hashes stayed unchanged. Raw evidence:

- `outputs/runtime-ane-swiglu-int8-tiny-20261001/` — original bridge-check failure.
- `outputs/runtime-ane-swiglu-int8-tiny-retry-20261001/` — complete tiny comparison.
- `outputs/runtime-ane-swiglu-int8-medium-20261001/` — complete medium comparison.

Next performance work must address preparation and FP32 device boundaries and
compare against GPU execution, before claiming an ANE W8A8 benefit. This round
does not implement the GPU A8W8 backend or move Qwen encoders to ANE.

## App delivery

The previously running `dist/TurboCider.app` was older than the delivered 0f8dfa7
package and lacked Playground. The refresh uses the validated native runtime,
rebuilds Swift App/helper sources, and keeps the original App and state backups
under `outputs/app-refresh-20261001/`. Release metadata identifies the exact
code revision; `update-report.json` records the installed bundle identity and
preservation checks. This update does not alter user model weights or import
the isolated test jobs into the user's history.
