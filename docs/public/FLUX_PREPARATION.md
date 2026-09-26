# Prepare FLUX.2 Klein 4B locally

[Documentation](README.md) · [Environment setup](ENVIRONMENT_SETUP.md) · [Getting started](GETTING_STARTED.md) · [Model library](MODEL_LIBRARY.md)

This guide prepares the complete model, exports optional Core ML partitions,
compiles them for the current Mac, and checks generation requests. Run the
commands from the repository root in the same shell. Complete the
[environment, build and packaging setup](ENVIRONMENT_SETUP.md)
first: the examples use `dist/cli/turbocider` and the project's Python 3.11
environment at `.venv/bin/python3`.

The preparation recorded on September 13–14, 2026 used an **M5 Pro with 24 GiB
unified memory, macOS 26.4.1 and MLX 0.32.0**. Environment setup, model preparation,
20 partition exports, local compilation, registration and GPU/hybrid request
planning completed. **No real model generation or performance comparison was
run.** The 512×512, dynamic-text, 1088-row, 6144-channel configuration below is
the manually prepared experiment, not a measured or recommended M5 policy.

**September 16 update:** M5 Pro 24 GiB now has a measured automatic case for
512×512 / 4 steps / resident / no LoRA / 1088 rows / 6144 ANE channels. See the
[M5 adaptation report](../design/m5-ane-adaptation.md) for repeated GPU/hybrid
measurements, quality checks and the exact scope. The historical preparation
record below remains useful for exporting and compiling the artifacts.

## 1. Choose a library and obtain the complete model

Use a repository-local library, shared by subsequent CLI commands:

```sh
TC_ROOT="$PWD"
TC_CLI="$TC_ROOT/dist/cli/turbocider"
TC_PYTHON="$TC_ROOT/.venv/bin/python3"
TC_PREP="$TC_ROOT/outputs/flux-preparation"
export TURBOCIDER_MODEL_LIBRARY="$TC_ROOT/models/library"
mkdir -p "$TC_PREP"
"$TC_CLI" library location
```

The environment variable selects this library for the current shell and its
children. To make the App and future CLI launches use the same directory, set
that directory in the App or run `"$TC_CLI" library configure
"$TURBOCIDER_MODEL_LIBRARY"`; this updates the shared saved library setting.

The native download recipe needs the transformer, text encoder, tokenizer, VAE,
scheduler and model index. A standalone transformer checkpoint is insufficient.
`make setup` does not fetch weights. The recorded complete selection contained
18 files totaling 15,980,131,745 bytes (about 15.98 GB decimal), before Core ML
exports, compiled caches and generated outputs. The recorded 20-partition
source set and compiled set each occupied about 1.1 GiB. Allow additional free
space for temporary downloads, export/compilation work and outputs; the final
weight size alone is not a sufficient disk-space budget. Preview the actual
selection and available disk space before downloading.

Create a request for the pinned ModelScope snapshot used during preparation:

```sh
cat > "$TC_PREP/download.json" <<'JSON'
{
  "modelID": "flux2-klein-4b",
  "repository": "black-forest-labs/FLUX.2-klein-4B",
  "provider": "modelscope",
  "revision": "a6e056113f18cb00eb3d9639b5c67fe60c32dd80",
  "include": [
    "model_index.json", "scheduler/", "transformer/",
    "vae/", "text_encoder/", "tokenizer/"
  ]
}
JSON
"$TC_CLI" library plan "$TC_PREP/download.json" > "$TC_PREP/download-plan.json"
cat "$TC_PREP/download-plan.json"
df -h "$TC_ROOT"
```

`library plan` fetches remote metadata, not model weights. After reviewing its
selected files, `downloadBytes` and reusable `cachedBytes`, download them:

```sh
"$TC_CLI" library download "$TC_PREP/download.json" \
  > "$TC_PREP/download-result.json" \
  2> "$TC_PREP/download-events.jsonl"
```

For Hugging Face, use `"provider": "huggingface"` and the corresponding recorded
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. Provider revision IDs are not
interchangeable. Preserve the selected revision and downloader provenance when
comparing runs; a different snapshot requires fresh compatibility checks.

If a transfer fails, rerun the same download request. Completed, verified files
in the blob cache can be reused, but the native downloader does not guarantee
byte-range resume within an interrupted file. If repeated large-file transfers
fail, use the provider's official client with resume support to obtain the
complete pinned model folder, then register it as below. Keep the required
directory layout and verify completion and integrity; `.partial` or otherwise
unfinished downloads are not model files ready for registration and use.

If a complete installation already exists, skip downloading and register its
directory, for example:

```sh
"$TC_CLI" library register flux2-klein-4b "$TC_ROOT/models/FLUX.2-klein-4B"
```

Change the last path to the actual existing directory. Registration keeps its
files in place. Native downloads register their installation automatically.

## 2. Resolve and inspect the installation

Managed installations receive generated directory names. Resolve the registered
path instead of copying a machine's installation ID into scripts:

```sh
"$TC_CLI" library resolve flux2-klein-4b > "$TC_PREP/resolved-model.json"
TC_MODEL="$("$TC_PYTHON" -c \
  'import json, pathlib, sys; r=json.load(open(sys.argv[1])); assert r["ok"]; print(pathlib.Path(r["result"]["path"]).resolve(strict=True))' \
  "$TC_PREP/resolved-model.json")"
"$TC_CLI" library inspect flux2-klein-4b "$TC_MODEL" \
  > "$TC_PREP/model-inspection.json"
cat "$TC_PREP/model-inspection.json"
```

`resolve` selects the most recently registered installation whose path still
exists. Check the returned path if several versions are registered.
`files_present` means the component, index and tensor-file structure checks
passed. It does not load the model, hash every weight or prove generation works.
The downloader performs its separate declared size/hash checks. The recorded
transformer SHA-256 was
`9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e`.
To record the selected transformer's full hash:

```sh
shasum -a 256 "$TC_MODEL/transformer/diffusion_pytorch_model.safetensors" \
  > "$TC_PREP/transformer.sha256"
```

## 3. Export the optional Core ML partitions offline

GPU generation does not require this step. The Python exporter reads the local
official Klein 4B `transformer/config.json` and
`transformer/diffusion_pytorch_model.safetensors`; it does not download or run
the complete model. It produces 20 per-channel INT8 MLP packages with FP16
inputs/outputs. This is approximate inference, not a bit-exact GPU equivalent.

Reuse the dedicated Core ML environment created by `make setup`, with the
separately pinned export dependencies (`coremltools==8.3.0`, `numpy==2.0.2`,
validated with Python 3.11):

```sh
TC_COREML_PYTHON="$HOME/Library/Application Support/TurboCiderNative/toolchains/coreml/bin/python3"
"$TC_COREML_PYTHON" -I -m pip check
TC_ANE="$TC_ROOT/artifacts/flux2-klein-4b/a6144-b1088"
"$TC_COREML_PYTHON" -I "$TC_ROOT/tools/coreml/export_flux2.py" \
  --model "$TC_MODEL" \
  --output "$TC_ANE/source" \
  --bucket 1088 \
  --ane-mlp-width 6144 \
  > "$TC_PREP/export.log" 2>&1
```

If setup used `--app-toolchain`, set `TC_COREML_PYTHON` to that environment's
Python instead; see [managed environments](ENVIRONMENT_SETUP.md#3-understand-the-managed-environments).
The export environment is a build tool; production inference does not start
Python workers. `-I` isolates export from inherited Python paths and user-site
packages.

The split assigns MLP channels `[0, 6144)` to the Core ML branch and
`[6144, 9216)` to the GPU. The export identity records the resolved checkpoint
path, size and SHA-256, bucket, variant, coremltools/NumPy versions, recipe and
any LoRA or nondefault channel-width settings. A changed identity requires a
new output directory. For an identical export, completed packages are reused
only when their receipt-listed file hashes pass. A corrupt package or an
existing package without its receipt is rejected rather than automatically
repaired. Without an identity marker, the exporter accepts only an empty
directory apart from its `.export.lock` and optional `export.log`.

For this text-to-image experiment, 512×512 contributes
`(512 / 16) × (512 / 16) = 1024` image tokens. Bucket 1088 leaves capacity for
64 encoded text tokens, including the chat template. Check the actual prompt:

```sh
"$TC_CLI" tokenize "$TC_MODEL" "A red fox in a snowy forest." \
  > "$TC_PREP/tokens.json"
cat "$TC_PREP/tokens.json"
```

The recorded tokenizer returned `valid: 20`, giving 1044 total tokens.
The requests below explicitly set `dynamic_text: true`. Disabling it pads FLUX
text to 512 tokens, so this image would require 1536 rows and would not fit the
1088-row artifact. Longer prompts, larger images and reference-image editing
also need their own capacity check and potentially another export. This FLUX
exporter uses a fixed bucket; Z-Image's enumerated-shape instructions do not
apply to it.

The example has no LoRA. A LoRA experiment needs partitions exported with the
same adapter, role and strength used by the generation request. Base-only
partitions cannot stand in for LoRA-bound partitions.

## 4. Compile and register the real manifest paths

Register the source manifest, then generate the native compilation request with
absolute paths derived from this checkout:

```sh
"$TC_CLI" library register-ane flux2-klein-4b "$TC_ANE/source/manifest.json"
"$TC_PYTHON" - "$TC_ANE" "$TC_PREP/compile.json" <<'PY'
import json
import pathlib
import sys

artifacts = pathlib.Path(sys.argv[1]).resolve()
request = {
    "action": "compile",
    "model": "flux2-klein-4b",
    "source_manifest": str(artifacts / "source/manifest.json"),
    "cache": str(artifacts / "compiled"),
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(request, indent=2) + "\n")
PY
"$TC_CLI" coreml "$TC_PREP/compile.json" \
  > "$TC_PREP/compile-result.json" \
  2> "$TC_PREP/compile-events.jsonl"
TC_MANIFEST="$("$TC_PYTHON" -c \
  'import json, pathlib, sys; r=json.load(open(sys.argv[1])); assert r["partitions"] == 20; print(pathlib.Path(r["manifest"]).resolve(strict=True))' \
  "$TC_PREP/compile-result.json")"
"$TC_CLI" library register-ane flux2-klein-4b "$TC_MANIFEST"
```

Compilation returns a `manifest` field pointing to `manifest-HASH.json`.
Use that returned value; do not assume `compiled/manifest.json` exists. The
native compiler stores `.mlmodelc` packages and identities tied to source
content, macOS build, architecture and GPU. Recompile for a different machine
or changed system identity. First-use device specialization can still occur
after this compilation finishes.

The App can discover the registered source and compiled partitions when using
the same library. Registration checks manifest metadata and referenced package
directories; loading performs the final checkpoint, LoRA, capacity and compiled
identity checks. Registering artifacts does not itself enable ANE. Hardware
example profiles are disabled templates with example paths, not bundled
partitions or a shortcut around these checks.

## 5. Prepare and check GPU and manual hybrid requests

Create paired requests with the same prompt, image size, steps, seed and
residency, using the compiled path returned above:

```sh
"$TC_PYTHON" - "$TC_PREP" "$TC_MANIFEST" <<'PY'
import copy
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1]).resolve()
gpu = {
    "model": "flux2-klein-4b",
    "operation": "image.generate",
    "prompt": "A red fox in a snowy forest.",
    "width": 512,
    "height": 512,
    "steps": 4,
    "seed": 42,
    "dynamic_text": True,
    "execution": "gpu",
    "residency": "resident",
    "output": str(output / "gpu.png"),
}
hybrid = copy.deepcopy(gpu)
hybrid.update(
    execution="gpu_ane",
    allow_approximation=True,
    ane_manifest=str(pathlib.Path(sys.argv[2]).resolve(strict=True)),
    output=str(output / "hybrid.png"),
)
for name, request in [("gpu", gpu), ("hybrid", hybrid)]:
    (output / f"{name}.json").write_text(json.dumps(request, indent=2) + "\n")
PY
"$TC_CLI" plan "$TC_PREP/gpu.json" > "$TC_PREP/gpu-plan.json"
"$TC_CLI" plan "$TC_PREP/hybrid.json" > "$TC_PREP/hybrid-plan.json"
cat "$TC_PREP/gpu-plan.json"
cat "$TC_PREP/hybrid-plan.json"
```

Planning validates request structure and supported execution stages and reports
a memory estimate. For FLUX it does not load weights or partitions, tokenize
the prompt, measure peak memory or establish successful generation. At 512×512,
the current heuristics report 14 GiB for GPU and 18 GiB for hybrid; these are
not measured peaks or hard memory limits. The recorded 24 GiB machine passed
these planning checks, but actual runtime memory and speed remain unverified.

M5 Pro **24 GiB** can now select automatic hybrid for the measured 512×512,
4-step, resident, base-model request with 1025–1088 total tokens and a6144
partitions. Change `execution` to `auto` to use that policy with opted-in
artifacts. Other hardware/configurations still fall back to GPU. The explicit
`gpu_ane` request above bypasses automatic performance selection; a successful
plan alone does not establish runtime performance.

## 6. Proceed to generation and measurement separately

The preparation session stopped before this step. To validate a real output,
run GPU first and inspect its result before the hybrid request:

```sh
"$TC_CLI" generate "$TC_MODEL" "$TC_PREP/gpu.json" \
  > "$TC_PREP/gpu-result.json" 2> "$TC_PREP/gpu-events.jsonl"
```

After the GPU request succeeds, run the manual hybrid request and compare the
images and reported route:

```sh
"$TC_CLI" generate "$TC_MODEL" "$TC_PREP/hybrid.json" \
  > "$TC_PREP/hybrid-result.json" 2> "$TC_PREP/hybrid-events.jsonl"
```

After both PNG files exist, run the repository's numerical comparison:

```sh
"$TC_PYTHON" -I "$TC_ROOT/tools/native/quality_gate.py" \
  "$TC_PREP/gpu.png" "$TC_PREP/hybrid.png" \
  > "$TC_PREP/quality.json"
cat "$TC_PREP/quality.json"
```

The report includes shape, finite-value and pixel-similarity checks and a
pass/fail result using the tool's default thresholds. Inspect the images too;
one image pair does not establish quality across prompts.

These one-shot processes do not establish warm performance. For a resident
session comparison, use `tools/native/benchmark_native.py` with matching GPU
and hybrid requests. Record the first request separately, preserve repeated
warm samples, alternate route order, avoid concurrent GPU work, and compare
output quality as well as complete request time including PNG export. The
external `flux2-engine` and `mflux` checkouts are only needed by the separate
legacy-engine comparison scripts, not by TurboCider generation or its native
GPU/hybrid benchmark. See [performance boundaries](PERFORMANCE.md) and
[the recorded benchmark conditions](BENCHMARKS.md).
