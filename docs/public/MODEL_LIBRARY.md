# Model library

[Documentation](README.md) · [Getting started](GETTING_STARTED.md)

The native `turbocider library` helper provides model registration and verified
downloads without Python or a GPU. App and CLI share its index and configured
root. The App offers search, registration, file preview, downloads, progress
and cancellation; existing draft paths are registered without moving weights.

## Location and registration

The default managed root is
`~/Library/Application Support/TurboCider/Models`. Select another root with
`turbocider library configure DIRECTORY` or the managed-directory control in the App. The shared
setting lives in `~/Library/Application Support/TurboCider/library-settings.json`.
`TURBOCIDER_MODEL_LIBRARY` overrides it; `--root DIRECTORY` overrides one command.
An existing external installation can remain in its original directory:

```sh
turbocider library register z-image-turbo /absolute/existing-model
turbocider library list
turbocider library location
turbocider library resolve z-image-turbo
turbocider library remove INSTALLATION_ID
```

`remove` only removes the registration. It retains external and managed files.
Use `turbocider library import paths.json` with a `modelPaths` object to register
existing paths. CLI `generate` and `batch` accept `@model-id` in place of a
model directory; the most recently registered existing installation is selected.
The alias must match the request model. `library catalog` shows download recipes.

Registration does not prove that the weights can run; native model loading
still validates the actual layout and tensors.

Use the installation-inspection control beside the selected path, or:

```sh
turbocider library inspect z-image-turbo /absolute/existing-model
```

The read-only report lists missing components, broken links, missing indexed
shards, malformed safetensors headers and tensor byte ranges beyond the file
length. FLUX checks also distinguish the 4B/9B architecture configurations.
`files_present` means this file check passed, not that a generation was run;
`incomplete` lists file problems, and `needs_preparation` lists remaining runtime
setup. App downloads run this check after publishing the selected files, so a
partial selection is not described as a ready model. Use the folder picker to repair a
moved installation, then remove the obsolete registration without deleting files.
The byte count covers referenced weight files, including shared files; it is
not additional disk usage. Reports are timestamped snapshots; recheck after
changing files. No tensor data is loaded, full weight hashes are not computed,
and ANE compilation is not triggered by inspection.

The library stores an atomic `library.json` index, `installations/` links,
content-addressed `blobs/`, provenance under `manifests/`, and unpublished
`staging/` directories. A process lock serializes App/CLI writers. Two files
with the same verified SHA-256 reuse the same stored bytes, even across sources.

## Preview before downloading

### Z-Image precision choices

The App defaults Z-Image downloads to `Comfy-Org/z_image_turbo` on ModelScope.
Select one diffusion checkpoint independently from the text component:

| Diffusion checkpoint | File size (decimal GB) | Runtime |
|---|---:|---|
| `z_image_turbo_bf16.safetensors` | 12.31 | BF16 GPU, qualified ANE profiles |
| `z_image_turbo_int8_convrot.safetensors` | 6.20 | Native ConvRot + MLX packed Q8 |
| `z_image_turbo_nvfp4.safetensors` | 4.51 | Experimental native MLX W4A16; GPU only, no LoRA/ANE |

NVFP4 files require high/low nibble and tiled-scale reordering, which the native
loader now performs without requantizing weights. Activations remain BF16;
this is **not** NVIDIA W4A4 execution. Smaller weights do not guarantee faster
inference or identical images. Inspection checks files, not numerical quality.

The Comfy repository has no tokenizer. When downloading Qwen BF16, the App
adds the two tokenizer JSON files from `Tongyi-MAI/Z-Image-Turbo` with their own
source revision/integrity record. See `examples/requests/download-z-image-int8.json`.
Reusing a local encoder skips both its weights and tokenizer downloads.

Qwen3-4B choices are BF16 (8.04 GB), locally converted MLX affine Q4
(3.05 GB), or Q8 (4.87 GB). These conversions retain dense BF16 embeddings,
use group size 32 and are not interchangeable with GGUF or Comfy FP4/FP8 mixed.
Q4/Q8 currently require this **offline developer conversion**, not an in-App
quantizer or a one-click download of a third-party quantized repository:

```sh
Python/bin/python3 tools/convert/qwen3_affine.py \
  --source models/Comfy-Org-z_image_turbo \
  --weights models/Comfy-Org-z_image_turbo/split_files/text_encoders/qwen_3_4b.safetensors \
  --bits 4 --output models/qwen3-4b-tc-q4
```

Use a Python environment with MLX (the example is this development workspace's
environment); `--bits 8` produces Q8. Existing output directories are refused.
Select Q4/Q8 in the download dialog, then choose the converted component root
under shared text. The native runtime does not invoke Python. The conversion
records source SHA-256 and MLX version. Preserve the original model license.

Each downloaded precision combination receives its own installation and name.
Use **Use this installation** to switch; content-addressed VAE/tokenizer files
are shared. For a manually assembled Comfy directory containing several DiT
files, discovery prioritizes BF16, then INT8, then NVFP4. Use separate
installation directories to select another precision explicitly. A root-level
`text_encoder/` binding overrides the Comfy BF16 text file, allowing Q4/Q8 reuse.

Disk weight totals and physical RAM are displayed separately. Runtime memory
also depends on resolution, activation buffers and staged release of the text
encoder. See [local quantization measurements](QUANTIZATION_2026-09-10.md).

### Generic download request

Create `download.json`:

```json
{
  "modelID": "z-image-turbo",
  "repository": "Tongyi-MAI/Z-Image-Turbo",
  "provider": "modelscope",
  "include": ["model_index.json", "scheduler/", "transformer/", "vae/", "text_encoder/", "tokenizer/"]
}
```

```sh
turbocider library plan download.json
turbocider library download download.json
```

`plan` reads remote metadata and checks existing verified blobs; it does not
download model weights. The result includes required bytes, reusable bytes and
selected files. `download` is the action that fetches model files. Omitted
`provider` defaults to ModelScope. Use `huggingface` to select Hugging Face;
default revisions are respectively `master` and `main`.

Some installations need small files from a second repository. `supplements`
uses the same selected provider and requires exact paths, for example:

```json
{
  "modelID": "ltx-2.5-distilled",
  "repository": "comfyicu/LTX-2.5",
  "provider": "huggingface",
  "include": [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "gemma4-12b-ltx-v1/tokenizer.json"
  ],
  "supplements": [{
    "repository": "mlx-community/ltx-2.5-mlx",
    "include": ["gemma4-12b-ltx-v1/tokenizer.json"]
  }]
}
```

The App's LTX recipe includes this tokenizer in the same preview and atomic
installation. It does not download the supplementary repository's encoder
weights. Each file retains its source repository, pinned revision and integrity
metadata. Conflicting selected paths fail before downloading. Components and
verified blobs retain their reuse behavior. If either repository is unavailable
on ModelScope, select Hugging Face and review both source licenses.

These paths were checked against the [LTX weight source](https://huggingface.co/comfyicu/LTX-2.5/tree/main)
and [tokenizer source](https://huggingface.co/mlx-community/ltx-2.5-mlx/tree/main/gemma4-12b-ltx-v1)
on 2026-09-08 using metadata only. Native tensor compatibility remains a separate
runtime check. H3 defaults to `FL2VA/`, with a separate Turbo LoRA preparation
step; ordered-reference requests require their own Ref2VA installation and
provenance. Wan defaults to the released `mlx_dit` and supporting components;
its native decoder requires offline-converted `vae/taew2_1.safetensors`. No Python worker is used. See [usage](USAGE.md).

Exact file paths and directory prefixes ending in `/` are supported by
`include`. An empty list selects the complete repository, which may include
documentation assets and multiple weight variants. Prefer model-specific
selections, inspect size, and observe the upstream license before downloading.

For gated/private models, accept the upstream terms on the provider site and
set `HF_TOKEN` or `MODELSCOPE_API_TOKEN` in the launching environment. Tokens
are not written into library metadata or forwarded to another origin on a CDN
redirect. ModelScope's native client uses its authenticated API convention;
Hugging Face revisions are resolved before paginated file listing.

## Reuse local components

Add explicit components to the same request, for example:

```json
{
  "components": {
    "text_encoder": {
      "path": "/absolute/compatible-model/text_encoder",
      "compatibility": "qwen3-4b-z-image"
    },
    "tokenizer": {
      "path": "/absolute/compatible-model/tokenizer",
      "compatibility": "qwen3-4b-z-image"
    }
  }
}
```

This is a request fragment, not a standalone request. Corresponding remote
files are skipped and links are created instead. For Z-Image, both components
are required together. `library shared-text DIRECTORY` and the App picker check
the Qwen3 architecture dimensions, tokenizer configuration and presence of
referenced encoder shards. Metadata checks establish architectural compatibility,
not numerical equivalence of arbitrary fine-tuned encoders. The native executor
performs final tensor validation. Caller-provided compatibility labels alone
do not bypass these checks.

## Failure and cancellation

The downloader verifies declared file lengths, LFS/ModelScope SHA-256 hashes,
and Git blob identities for Hugging Face non-LFS files. It publishes a model
only after the selected downloads complete. Cancellation removes that run's
partial staging directory; completed verified blobs remain reusable on retry.
The current transfer retries at file granularity; it does not promise byte-range
resume. Forced process termination may leave unpublished staging data.

After a download, inspect the installation before generating. A file preview
or successful transfer alone does not establish numerical model compatibility.
Use the [getting-started workflow](GETTING_STARTED.md) for the first GPU request.

API references: [Hugging Face Hub API](https://huggingface.co/docs/hub/en/api),
[ModelScope Hub API implementation](https://github.com/modelscope/modelscope_hub/blob/main/src/modelscope_hub/_legacy_api.py).

## Model and LoRA configuration

The App's **管理目录 → 导出模型与 LoRA 配置** writes a portable JSON
configuration. Import it with **导入模型与 LoRA 配置** or `library import`:

```json
{
  "schemaVersion": 1,
  "modelPaths": {
    "z-image-turbo": "/absolute/model-library/bindings/z-image-installation",
    "flux2-klein-4b": "/absolute/external/FLUX.2-klein-4B"
  },
  "loras": [
    {"modelID": "z-image-turbo", "path": "/absolute/external/loras/style.safetensors"}
  ]
}
```

Paths must exist on the destination machine. Import changes successfully
registered model selections, retains unrelated entries, and reports missing
paths individually. It does not change prompts or enable imported adapters.
`library.json` remains the shared inventory of installations, components,
provenance and model-associated LoRA records; old indexes without `loras` remain
readable. External paths are references, so weights are never copied just to
register them. LoRA aliases resolve to a canonical path and deduplicate per
model. CLI commands are `register-lora MODEL_ID FILE` and `remove-lora ID`.
Removing registration always retains the original weights.

Each model's **LoRA 模型库** supports registering files, scanning its `loras/`,
`split_files/loras/` and `models/loras/` directories, and adding a registered
adapter to the creation draft. Registration records the intended base model;
it does not prove tensor compatibility. Discovered adapters are not automatically
enabled. Switching models preserves each model's draft adapters and strengths.
The execution strategy resets to automatic when switching models.

Z-Image bindings now live under the configured library's `bindings/` directory.
On startup, readable legacy bindings with `installation.json` outside this root
are recreated there, with links directly to the resolved source directories.
The old output directory can then be removed without breaking the new binding;
the actual source weights must remain available. Existing registrations are
retained for review. App session opening checks Z-Image component files, shard
indexes and safetensors ranges before invoking the engine, and reports broken
links or missing files with their component paths.

## ANE partitions in the shared library

The model page includes **ANE 分区** for Z-Image Turbo and FLUX.2 Klein 4B.
Register either a source `.mlpackage` manifest or a compiled `.mlmodelc`
manifest. Registration reads schema 2 metadata and checks the architecture,
shape mode, sorted capacity buckets, complete block list, LoRA identities and
referenced artifact directories. A manifest left behind after package deletion
is rejected with the missing block path; registration does not regenerate it.
Files remain in place. Removing a registration retains the files.

`library.json` now includes optional `anePartitions` records: model ID, canonical
manifest path, source/compiled kind, fixed/enumerated/range mode, buckets,
checkpoint, LoRA file/strength/role, block count and linked source manifest.
Re-registering the same path refreshes its metadata and retains its ID. Old
indexes and configurations without this field remain compatible.

```sh
turbocider library register-ane z-image-turbo /absolute/source/manifest.json
turbocider library register-ane z-image-turbo /absolute/cache/manifest-HASH.json
turbocider library remove-ane PARTITION_ID
```

The App's unified import/export includes manifest references:

```json
{
  "modelPaths": {},
  "anePartitions": [
    {"modelID": "z-image-turbo", "path": "/absolute/source/manifest.json"}
  ]
}
```

The App discovers registered partitions on each explicit ANE request, even when
no partition path is saved in the draft. It rechecks checkpoint identity, LoRA
identity/strength, token capacity, artifact existence and compiled-cache machine
identity. Registry snapshots never bypass these checks. Registered source
partitions can be compiled on demand; the resulting compiled manifest is also
registered. **编译并登记** in the library writes to `<model-root>/ane-cache` and
retains the source relationship. Existing default App caches remain supported.
An explicitly selected compatible manifest takes precedence over other matches.
Hardware-specific automatic performance gates remain unchanged.

Variable-length support must exist in the exported source model. The Z-Image
exporter still defaults to fixed shape; explicitly request `--shape-mode
enumerated` or `--shape-mode range`. For 512×512 output, `--min-bucket 1056
--bucket 1536 --bucket-step 32` covers 1024 image rows plus up to 512 text rows.
Changing output resolution can require a larger export. Checking ANE in the App
compiles an existing source; it does not run the offline Python exporter.
