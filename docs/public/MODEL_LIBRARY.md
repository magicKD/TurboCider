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
