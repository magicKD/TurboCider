# Local API

[Documentation](README.md) · [Usage reference](USAGE.md)

TurboCider exposes a persistent job service over a user-only Unix socket.
The App's local API page starts an owned service and shows its socket path.
It releases the embedded model session first. Stop the API to resume generation
inside the App. Quit or unexpected App termination stops its owned service;
an independently launched CLI service has its own lifecycle.

## CLI lifecycle

Inspect the protocol before starting a service:

```sh
dist/cli/turbocider --help
dist/cli/turbocider capabilities
```

The second command returns the same JSON object as a service's `capabilities`
result, using the installed CLI version. It needs no socket, model or registry
and creates no service state. Query the live service when checking its actual
version; use `service_status` to check whether it is running.

Keep `dist/cli/` intact, then run:

```sh
dist/cli/turbocider serve /tmp/turbocider.sock /absolute/service-state
```

The service stays in the foreground. Ctrl-C or SIGTERM cancels running work,
marks pending work interrupted, releases its model session and removes the
socket. Generation cancellation waits for a safe runtime boundary. A crash may
leave a socket; the next start recovers an owned, inactive socket and marks
unfinished history interrupted. Concurrent servers cannot share the same
state directory or socket.

Do not put `TURBOCIDER_SERVICE_PARENT_PID` in a shell profile. The App sets this
internally to bind its child's lifetime to the App; standalone `serve` does
not need it.

## RPC fields

Send one newline-terminated JSON object per connection. Responses use
`{"ok": true, "result": ...}` or `{"ok": false, "error": ...}`.

| Action | Additional fields |
|---|---|
| `submit` | `model_path` and complete `request`; returns a job ID |
| `status`, `cancel` | `id` |
| `jobs` | Optional `offset`, `limit` (default 20, maximum 100) |
| `plan` | `request` |
| `workflow_request` | `input`: workflow ID, role paths, optional instruction/expansion and native request settings |
| `image_prepare` | `input`: absolute `source_path`, `preset`, optional new absolute `.png` `output_path`; explicit CPU downsizing |
| `workflows` | None; shared App/API templates, role order, prompts and construction schema |
| `capabilities`, `models`, `installations`, `doctor`, `service_status` | None |

`capabilities` is the discovery entry point for scripts and local AI agents.
It returns the protocol version, action descriptions and JSON Schemas for each
RPC envelope, response envelopes, job states, transport/queue limits and
workflow rules. It does not load weights. The embedded `request` is validated
by the native `plan` action, not by a duplicate schema maintained by the client.
`models` lists registered capabilities, not installed weights. Supply an
existing native-compatible model path when submitting.

Successful generation results include `service_execution_path` and
`service_session_reused`. The latter means the service reused its existing
engine for the same model and path, including Qwen and other native models.
It does not guarantee that component weights stayed resident or that prompt
conditioning was cached; those have separate runtime fields. Disposable
workers always report false.

Qwen-Image-2.1's model entry includes `reference_encoding`: the default 1024px
processing, field locations for native request schemas 1/2, base 256/512px
approximation and the stricter 512px r128 GPU editing opt-in. Separate
`base_constraints` and `viggle_r128_gpu_edit_constraints` describe those routes;
`ordinary_lora_reference_size` retains the legacy default of 1024;
`ordinary_lora_supported_reference_sizes` and `ordinary_lora_constraints`
describe explicit 512px editing support. `dit_cache_supported_reference_sizes`
reports both 512 and 1024. Named DiT presets can combine 512px reference encoding
with 512×512 GPU edits, 20–40 steps and 1–3 references, on base or ordinary
runtime LoRA. The six-step Viggle route still requires DiT cache off.
Reduced reference encoding requires
`allow_approximation: true`; use `plan` for authoritative validation and see
the [Qwen request guide](QWEN_IMAGE_21.md). `plan` does not load or verify the
adapter; its pinned contents are checked when the model binds it.

`installations` returns the service's local model-library registry, including
registered model components, LoRAs and ANE partitions. The response contains
`schema_version: 1`, `root`, `index`, `scope: "registered_metadata"` and
`files_verified: false`. Registrations may be stale; this query does not read
weights, scan model directories, download files or establish compatibility.
Choose a registration explicitly and use its absolute path in a request.

The service invokes its sibling `turbocider-library inventory` helper using its
own environment and library settings. A client's environment cannot select a
different library through this RPC. Missing metadata returns an empty registry
without creating a directory; unreadable or invalid metadata returns an error.
The index is limited to 4 MiB and settings to 1 MiB. The helper has a 10-second
deadline, 8 MiB stdout limit and 64 KiB diagnostic limit. Inventory dispatch can
delay other RPC replies until it finishes, but does not lock the inference
worker. Older services/helpers must be updated together; clients must not fall
back to guessing paths or scanning the caller's default model directory.

Unknown action fields and duplicate JSON keys (including escaped spellings and
keys inside the native request) are rejected. `offset` must be an integer in
0–2147483647; `limit` must be an integer in 1–100. Booleans, numeric strings,
fractions, nulls and out-of-range values are errors, not implicit conversions.
Request JSON is limited to 1 MiB, excluding the terminating newline.

For the CLI client, save an action such as `{"action":"models"}` to
`rpc.json`, then run `dist/cli/turbocider rpc /tmp/turbocider.sock rpc.json`.

## Python client

The repository includes a dependency-free client in
[`bindings/python/turbocider_local.py`](../../bindings/python/turbocider_local.py).
Add `bindings/python` to `PYTHONPATH`, or copy that single file into your local
automation project. In the App, **复制 AI 接入说明** copies the actual socket path,
discovery request and lifecycle rules for Codex or another local agent.

```python
from turbocider_local import Client, JobTimeout, TransportError

client = Client('/the/socket/path/shown/in/the/App')
capabilities = client.capabilities()
models = client.models()
inventory = client.installations()  # service-side registered paths; files unverified
plan = client.plan(request)  # complete native schema 1 or 2 request
job_id = client.submit('/absolute/local/model', request)
print(job_id, flush=True)   # keep this ID even if waiting is interrupted
job = client.wait(job_id, timeout=1800)
```

The client never starts a service, downloads files or retries a submission.
`JobTimeout` preserves `job_id` and the last observed job; it does not cancel the
job. `JobFailed` carries the terminal job record. A transport error after sending
`submit` has `submission_may_have_succeeded=True`: inspect `jobs` and match the
unique requested output path before deciding whether to submit again. A lost
reply is not evidence that the job failed to enter the queue. Cancellation also
requires polling until a terminal state.

`image_prepare` also has no automatic retry. A transport failure after sending
it has `file_write_may_have_succeeded=True`, while
`submission_may_have_succeeded` stays false. Inspect the requested output before
deciding whether to retry; the response may have been lost after a successful
exclusive file creation. A connection failure before sending has both flags
false. The new flag defaults to false for other operations and preserves the
existing submission flag's meaning.

The Python client's default transport timeout is 30 seconds. If overriding it,
allow more than 10 seconds for `installations`; the CLI gives that query
12 seconds and a larger response limit. Neither client retries a submission.

The following inline alternative needs only the Python standard library:

Only the standard library is needed. Set the socket to the App page's value
or your CLI service's socket. Model paths are absolute native-compatible
installations; output paths should be unique. This example uses local FLUX
weights and saves a 512×512 image:

```python
import json
import socket
import time

SOCKET = '/tmp/turbocider.sock'

def rpc(action, **fields):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(30)
        client.connect(SOCKET)
        client.sendall(json.dumps({'action': action, **fields}).encode() + b'\n')
        with client.makefile('rb') as stream:
            response = json.loads(stream.readline())
    if not response['ok']:
        raise RuntimeError(response['error'])
    return response['result']

print(rpc('models'))
print(rpc('service_status'))
request = {
    'model': 'flux2-klein-4b',
    'prompt': 'A red fox in a snowy forest, soft morning light.',
    'width': 512, 'height': 512, 'steps': 4, 'seed': 42,
    'output': '/absolute/outputs/fox-42.png',
}
print(rpc('plan', request=request))
job_id = rpc('submit', model_path='/absolute/FLUX.2-klein-4B',
             request=request)['id']
while True:
    job = rpc('status', id=job_id)
    print(job['state'], job.get('progress', {}))
    if job['state'] in {'succeeded', 'failed', 'cancelled', 'interrupted'}:
        print(job.get('result', job.get('error')))
        break
    time.sleep(1)

# Cancellation: rpc('cancel', id=job_id)
# Paged history: rpc('jobs', offset=0, limit=20)
```

The same schema accepts native image/video requests; executor capability and
installed artifacts still apply. `plan` checks the request, not every weight
file. Progress contains a real phase, completed work and phase total; a denoise
step count is not a percentage of whole-job wall time.

`service_status` returns the service PID, active job ID (empty when idle),
history count, closing state, open session's model ID and external-worker state.
An open session does not establish that all component weights are resident.
`status` returns job details and final metrics.
For repeated prompts, submit another request with a new seed and output path.
A compatible resident session reuses conditioning; check `prompt_cache_hit`
in the result. Changing model or relevant input identity invalidates reuse.
The service admits up to 32 pending jobs; history is capped at 10,000 records.

## Explicit reference-image downsizing

`image_prepare` reads a local image using CPU only, without loading a model or
creating a generation job. It preserves aspect ratio and transparency, never
crops, pads or upscales, and does not modify the source file. Source images must
contain one frame and at most 80 million pixels.

The input is a strict object with `source_path` and `preset`, optional
`schema_version: 1` and optional `output_path`. Both paths must be absolute and
contain no embedded NUL. An output path must end in lowercase `.png` even when
the operation needs no resizing. Unknown fields, duplicate JSON keys, unsupported
versions and invalid presets are rejected.

| Preset | Bounding dimensions |
|---|---|
| `original` | Return the original file without resizing |
| `automatic` | 1024×1024; longest side at most 1024 |
| `fit512` | 512×512 |
| `portrait512` | 512×768 |
| `landscape512` | 768×512 |

These are fit bounds, not a request for a forced aspect ratio. For example, a
2048×1024 source becomes 512×256 with `fit512`. If resizing is needed,
`output_path` is required, its parent directory must already exist, and the
output must be a new file. Publication is atomic and exclusive; an existing
file, symlink or directory is not overwritten. If no resizing is needed,
`output_path` is not written and `image_path` is the original source path.
**Always use the returned `image_path`; do not assume `output_path` exists.**

When `output_created` is true, the caller owns that PNG; the API does not
register or automatically delete it. Keep it until every task that references
it has reached a terminal state, and longer if it is needed for history replay
or another workflow. When `output_created` is false, `image_path` is the source
image, not a disposable derivative. After a lost reply or client timeout,
confirm that preparation has finished and establish file ownership before
removing anything; a client timeout does not cancel the operation.

```python
prepared = client.prepare_image({
    'schema_version': 1,
    'source_path': '/absolute/reference.png',
    'preset': 'fit512',
    'output_path': '/absolute/prepared/unique-reference.png',
})
reference_path = prepared['image_path']
# Use reference_path in inputs or workflow_request.role_paths.
# Keep the requested generation width/height separate from this input resize.
```

The result contains `schema_version`, `source_path`, `preset`, `image_path`,
`output_created`, `original_width`, `original_height`, `width`, `height` and
`changed`. Discover the exact input/result schemas and restrictions through
`capabilities` → `image_prepare` and `image_preparation`.

The equivalent CLI command is:

```sh
dist/cli/turbocider prepare-image image-preparation.json
```

For raw RPC, send
`{"action":"image_prepare","input":{"source_path":"/absolute/reference.png","preset":"original"}}`.
This action is marked `mutates: true` because a resize can create a file. It is
synchronous on serial RPC dispatch and can delay other RPC replies until the
CPU/file operation finishes, but it does not lock the inference worker.
There is no immediate cancellation guarantee or `cancel` job ID for this call.
Client transport timeouts do not prove that file work stopped. Do not blindly
retry a lost reply; inspect the output and choose a fresh path when appropriate.

## Ordered editing workflows

The App and API share five Qwen templates: `playground.outfit`,
`playground.identity`, `playground.face`, `playground.outpaint` and
`playground.transparent`. Query `workflows` for their titles, ordered roles,
required inputs, default instructions, prompt modes and `input_schema`.
The catalog is compiled from one JSON source; the App does not maintain a
separate copy of the prompt prefixes.

`workflow_request` is **pure composition**. It does not read images, check
weights, resize or pad files, load a model, call `plan`, submit work or create
service history. It returns `request`, `operation`, `prompt` and an ordered
`roles` receipt. It replaces the base request's operation, prompt and inputs;
dimensions, seed, steps, execution, LoRA and cache settings remain unchanged.
Other native settings pass through for authoritative validation by `plan`.
Use explicit settings for generation; a minimal schema-1 request can use native
defaults when constructing a preview. Schema 2 is also accepted, with one
text/prompt input followed by the declared image references.

```python
catalog = client.workflows()
composed = client.workflow_request({
    'workflow_id': 'playground.face',
    'role_paths': {'person': '/absolute/identity.png',
                   'target': '/absolute/target.png'},
    'instruction': 'Keep the target lighting and expression.',
    'request': {'schema_version': 1, 'model': 'qwen-image-2.1',
                'width': 512, 'height': 512, 'steps': 25, 'seed': 42,
                'execution': 'gpu', 'residency': 'component_staged',
                'qwen21_dit_cache': 'off', 'prompt_enhance': False,
                'output': '/absolute/results/unique-face-result.png'},
})
plan = client.plan(composed['request'])
job_id = client.submit('/absolute/Qwen-Image-2.1', composed['request'])
print(job_id, flush=True)
job = client.wait(job_id, on_progress=lambda job: print(job.get('progress', {})))
# Need to stop? client.cancel(job_id), then keep querying status until terminal.
output_path = job['result']['output']  # Local artifact; no upload/download API.
```

For the CLI, `turbocider workflows` returns the catalog and
`turbocider workflow-request workflow-input.json` returns the same composition
envelope. Save its `request` object separately for `turbocider plan`,
`generate`, or an API `submit` request. Progress is available on CLI stderr or
by polling API `status`; the service does not expose an event subscription.

`outfit` requires `person` and `clothing`; `identity` requires `person` and
accepts an optional `scene`; `face` requires `person` (facial identity) and
`target` (the image to edit). `outpaint` requires `source` and accepts only
`expansion` 1.25, 1.5 or 2, defaulting to 1.5. Expansion guides scene
recomposition in the prompt: it does not change the requested output dimensions
or guarantee fixed borders or preserved source pixels. Actual canvas dimensions
still need to satisfy the selected model/LoRA/cache combination in `plan`.

For `transparent`, omit `source` to generate a transparent subject, or supply
`source` to extract the subject with `image.edit`. When instruction is omitted,
generation uses the catalog's concrete subject example and extraction uses its
source-preserving mode default. An explicit instruction is appended verbatim.
Qwen exports RGBA PNG, but success alone does not establish correct subject
extraction or a nonopaque alpha channel. Inspect the local image/alpha before
using it as a transparent artifact. Masks remain visual references rather than
hard pixel-preserving inpainting.

API inputs are existing local files. Each `inputs` reference position determines
the image number used by the model. Keep identity/outfit/scene roles in your
workflow state, then emit references in that declared order. Do not reorder
them after constructing a prompt containing image numbers.

[`qwen_reference_workflow.py`](../../examples/local_api/qwen_reference_workflow.py)
demonstrates one- to three-reference Qwen Turbo editing at 512×512, optionally
followed by another edit using the first output. It uses an existing Viggle
six-step adapter at strength 1, GPU, and DiT cache off. This is a pinned Turbo
example, not the schedule for ordinary LoRA adapters. Reference encoding
defaults to 1024. Select `--reference-size 512` for the explicit approximate
r128 GPU edit option; this does not change the 512×512 output canvas.

```sh
python3 examples/local_api/qwen_reference_workflow.py \
  --socket /the/app/socket \
  --model /absolute/Qwen-Image-2.1 \
  --turbo-lora /absolute/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors \
  --reference-size 512 \
  --reference /absolute/person.png --reference /absolute/outfit.png \
  --prompt 'Put the outfit from image 2 on the person in image 1, preserving their identity.' \
  --then 'Keep the same person and outfit; change the background to a garden.' \
  --output-dir /absolute/results
```

Without `--run` it only discovers the API and plans the first step. Add `--run`
to submit; the second step is planned and submitted only after the first output
exists and its job has succeeded. Each invocation creates unique output names
and prints each job ID immediately. A failed stage stops the workflow. The
service does not provide an atomic workflow transaction or automatically undo
earlier successful jobs. A standalone CLI service can outlive the App.

The App's local API page provides a read-only, paged task list with status,
errors, creation time and result timing. Completed local image results can be
previewed, opened, located in Finder or imported into the current creation
draft as references. Videos and other local results provide file location.
Importing a reference preserves the draft's model, prompt and generation
settings; stop the API service before generating from the creation page.
API jobs remain in their service history and are not automatically inserted
into the App's creation history or copied into its materials. The task list
loads when the page opens or the service starts, and on manual refresh; it does
not continuously poll history. If the service stops or a refresh fails, the
last successful page remains visible and is explicitly marked as possibly
outdated, with its last refresh time. A failed read is not shown as empty
history. Installation discovery is available through
`installations` for registered metadata; it does not verify
files or scan for unregistered models. Shared App history and a persisted
workflow scheduler are not part of this protocol revision.

## Troubleshooting

Start with `models` and `service_status` to check connectivity without loading
weights. Use `plan` to inspect a request before submitting it. If the socket is
unavailable, confirm the service is running and use the path shown in the App.
Another process cannot use the same state directory or socket concurrently.

Actual generation requires compatible local weights and macOS permission to
access Metal. A successful connection or plan does not certify model output.
Use unique output paths; stop an App-owned service through its local API page.
