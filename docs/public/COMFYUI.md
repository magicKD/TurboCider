# ComfyUI / Comfy Desktop backend

[中文使用教程：App 接口、Desktop 连接、模型与 LoRA](COMFYUI_TUTORIAL.zh-CN.md)

TurboCider owns model weights and inference. ComfyUI sends requests over a local
HTTP API, receives ordinary `IMAGE` / `MASK` tensors, and can use its existing
preview, save, resize, composite and image-batch nodes. No checkpoints, CLIP,
VAE or LoRA weights need to be installed in ComfyUI. The node installer copies
Python code only and never invokes pip or a model downloader.

## Start

Use an existing ComfyUI installation (including one managed by Desktop):

```sh
python3 tools/comfyui/install.py /path/to/ComfyUI
python3 tools/comfyui/start.py --cli dist/cli/turbocider
```

The second command starts the native Unix API and HTTP gateway on
`http://127.0.0.1:8765`. Keep the terminal running. Ctrl-C stops both owned
processes. Existing model registrations come from the TurboCider library.
`--state-dir`, `--socket` and `--port` allow separate instances; state directories
and sockets must not be shared between running services.

To start both services and an existing ComfyUI together, use:

```sh
python3 tools/comfyui/start.py --comfyui /path/to/ComfyUI \
  --comfy-python /path/to/comfy/python
```

This launcher sets offline model-library environment flags and stops its owned
processes together on Ctrl-C. It never installs ComfyUI or its dependencies.
Alternatively, restart ComfyUI yourself. For a dedicated API-only installation:


```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /path/to/comfy/python \
  /path/to/ComfyUI/main.py --cpu --offline --listen 127.0.0.1 --port 8188
```

`--cpu` controls ComfyUI's own model placement; TurboCider still uses Metal GPU.
`--offline` is supported by the tested ComfyUI version; older versions can use
`--disable-api-nodes`. These workflows never use ComfyUI model loaders.

In **Comfy Desktop**, add a **Remote Connection** named `TurboCider Local`, with
ComfyUI URL `http://127.0.0.1:8188`. This URL is the ComfyUI server, not port 8765.
A local remote connection installs no models. Keep both local services running.
Comfy Cloud cannot reach your Mac's loopback API; use the local connection.

Packaged releases include the same files under `dist/comfyui/`; from that folder,
run `python3 tools/comfyui/start.py` (finds the sibling `../cli/turbocider`).

The App's **本地 API / Local API** page starts the native service and a built-in
HTTP gateway together. Choose the port (default `8765`), click **启动服务 / Start**,
and copy the displayed HTTP URL into the Model node. The App includes expandable
curl/Python examples and a route reference. No Python runtime or external gateway
script is required for this mode; start ComfyUI separately after installing nodes.
Do not run `start.py` or `serve.py` on the same port alongside the App.

Stopping the App service or quitting the App closes its HTTP listener and stops
its native worker. An occupied port fails startup and rolls back the owned native
service. The independent CLI launcher remains available when App ownership is
not desired. For an older App or a separately managed Unix service, the standalone
adapter can still be started with `serve.py --socket ... --data-dir ...`.

## Nodes

| Node | Behavior |
|---|---|
| **Model (API)** | Select a registered installation; returns a backend handle and native capability JSON. Loads no weights in ComfyUI. |
| **LoRA (API)** | Select a registered adapter, strength and strategy. Chain nodes for ordered adapters; the native plan checks support. Creates a new handle without mutating branches. |
| **Generate Image** | Prompt, dimensions, seed, steps, batch size and advanced native options. Returns RGB images, inverse-alpha masks and job receipts. |
| **Edit / Transform Image** | Accepts an IMAGE batch as ordered references; `image.transform` applies strength to each init-image input. Model-specific admission remains in TurboCider. |
| **Native Request / Video** | Complete native schema-1 request for any registered executable model, including video. Returns a downloaded local artifact path and receipt; optional IMAGE batch becomes ordered inputs. |

Generate ready-to-import workflows for your existing registrations:

```sh
python3 tools/comfyui/workflows.py --output-dir outputs/comfyui-workflows
```

Open a generated JSON in ComfyUI / Comfy Desktop with Cmd-O (Ctrl-O).

Connect **Model → Generate Image → Save Image**. Put **LoRA** between Model and
generation when needed. Refresh node definitions after starting the API or
changing registrations. `TURBOCIDER_API_URL` chooses the default discovery URL;
for another endpoint, set the Model node URL and its optional installation ID.
LoRA also accepts an explicit registration ID.

Steps `0` uses the backend model default. Batch generation submits sequential
jobs with `seed + index`, wrapping at native schema's maximum `2147483647`.
Queueing the graph again executes it again, even with an unchanged seed.
ComfyUI's interrupt cancels the currently submitted native job; native cancellation
finishes at a safe boundary. Timeout also requests cancellation. Keep the job ID
from logs/receipt when diagnosing an interrupted client or lost reply.

RGB output connects to standard ComfyUI nodes. To preserve transparency when
saving, combine RGB and `alpha_mask` with ComfyUI's **Join Image with Alpha**
node before saving. A generated alpha channel does not guarantee semantic
background removal. Reference batches represent multiple references for one
edit, not independent editing jobs; use ComfyUI list iteration for the latter.

### Model and LoRA settings

- FLUX.2 Klein 4B: generate, edit and transform; typical 512×512 / 4 steps.
  Native LoRA strategy is `in_memory_merge`.
- Z-Image Turbo: text-to-image; typical 512×512 / 8 steps. Both
  `in_memory_merge` and `inference_time` are passed through. It does not expose
  image editing; unsupported operations fail before generation.
- Qwen Image 2.1 base / ordinary LoRA: use 20–40 steps. Viggle v0.2.1 r128/r256
  is a special six-step adapter requiring its native pinned identity, strength
  1 and `{"allow_approximation":true}`. Select the existing Viggle adapter and
  set steps to **6** explicitly. Reduced reference encoding with the r128
  adapter uses `{"allow_approximation":true,"qwen21_reference_size":512}`
  at 512×512 and 1–3 references. DiT cache stays off for the six-step route.
  The native runtime's experimental editing-quality limitations still apply.
- Video models are exposed only when registered and executable. Offline-premerged
  video LoRA is not interchangeable with runtime image LoRA. The integration
  never merges on disk, invents weights or substitutes a different model.

Advanced options are a JSON object of native schema-1 fields, e.g.
`{"residency":"component_staged","qwen21_dit_cache":"off"}`. Node-owned
model/prompt/shape/seed/output/inputs/LoRA fields cannot be overridden there.
Unsupported fields and combinations are rejected by native `plan`. For the
advanced Request node, omit `output`; the gateway allocates a unique output.
Native local paths in advanced requests refer to the API host, never the
ComfyUI host. Ordinary image nodes transfer PNGs and need no shared paths.

Standard `KSampler`, `MODEL`, `CLIP`, `VAE` and `LATENT` nodes operate on locally
loaded Comfy models and are not backend handles. Compose workflows at the
image boundary; TurboCider owns sampling and encoding. Arbitrary ControlNet,
hard-mask inpainting or sampler substitution is not advertised by this pack.

## HTTP contract

All routes are under `/v1`. JSON requests require `Content-Type: application/json`.

| Method / route | Body / result |
|---|---|
| `GET /health` | Native service status; no model load |
| `GET /catalog` | Executable installed models, native capabilities, registered LoRAs, Qwen workflow catalog |
| `POST /assets` | Raw `image/png`, up to 64 MiB / 80 million pixels → `id` |
| `POST /plan` | `{"installation_id":"…","request":{…}}` → native plan |
| `POST /jobs` | Same as plan → `{"id":"…"}`; never automatically retried |
| `GET /jobs` | Latest 100 native jobs for recovery/diagnosis |
| `GET /jobs/{id}` | Gateway job status, phase progress, terminal result/error |
| `POST /jobs/{id}/cancel` | `{}` → cancellation status; poll for terminal state |
| `GET /jobs/{id}/artifact` | Successful job's PNG or MP4 only |

Use `{"kind":"image","role":"reference","asset_id":"…"}` for uploaded
references, and `{"id":"registered-lora-id","role":"transformer","strength":1}`
for registered LoRAs. The gateway maps these to the service's existing paths;
`plan` is the only execution admission authority. Stale registrations report
an error rather than scanning directories or downloading replacements.

The HTTP service listens only on IPv4 loopback, rejects browser Origin headers
and non-loopback Host headers, and exposes no arbitrary file-download route.
This is a trusted local automation API, not a multi-user public hosting service:
native advanced options can name local input/adapter/cache paths. The App gateway has no bearer-auth setting. For optional
bearer auth with the standalone Python gateway, set `TURBOCIDER_API_TOKEN` in both gateway and ComfyUI environments.
The Model node stores an environment-variable name, not a credential in the
workflow. Do not expose this trusted API directly to an untrusted network.

Uploaded references, native history and generated outputs persist in the state
directory so queue execution, cancellation and history inspection remain safe.
They are not deleted automatically. Stop jobs/services before manually removing
an old state directory. An HTTP disconnect does not cancel a native job. A lost
submission reply may follow acceptance: inspect `/jobs` before submitting again.
The native Unix API remains available and supports both native schemas 1 and 2.

## Verification

```sh
make test-comfyui COMFYUI_PYTHON=/path/to/comfy/python
make test-api
python3 tools/comfyui/validate.py --output-dir outputs/comfyui-e2e
# Run only while both queues are idle:
python3 tools/comfyui/validate_lifecycle.py --output-dir outputs/comfyui-lifecycle
```

The opt-in E2E runner uses the real ComfyUI `/prompt` queue and existing
registrations. It covers FLUX batch/edit/transform, Z-Image base/runtime-LoRA/
merged-LoRA/base-again, Qwen base/Viggle/one-and-two-reference editing/ordinary
LoRA/ordinary-LoRA editing/base-again. No model is downloaded; missing registrations are reported as
skipped. Requests, history, timings and output images are retained. Use `--case`
(repeatable) for targeted reruns, keeping the output directory's FLUX references.
Read the dated validation report for exact tested versions and limitations.
