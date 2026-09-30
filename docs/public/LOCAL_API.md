# Local API

[Documentation](README.md) · [Usage reference](USAGE.md)

TurboCider exposes a persistent job service over a user-only Unix socket.
The App's local API page starts an owned service and shows its socket path.
It releases the embedded model session first. Stop the API to resume generation
inside the App. Quit or unexpected App termination stops its owned service;
an independently launched CLI service has its own lifecycle.

## CLI lifecycle

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
| `capabilities`, `models`, `installations`, `doctor`, `service_status` | None |

`capabilities` is the discovery entry point for scripts and local AI agents.
It returns the protocol version, action descriptions and JSON Schemas for each
RPC envelope, response envelopes, job states, transport/queue limits and
workflow rules. It does not load weights. The embedded `request` is validated
by the native `plan` action, not by a duplicate schema maintained by the client.
`models` lists registered capabilities, not installed weights. Supply an
existing native-compatible model path when submitting.

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

## Ordered editing workflows

API inputs are existing local files. Each `inputs` reference position determines
the image number used by the model. Keep identity/outfit/scene roles in your
workflow state, then emit references in that declared order. Do not reorder
them after constructing a prompt containing image numbers.

[`qwen_reference_workflow.py`](../../examples/local_api/qwen_reference_workflow.py)
demonstrates one- to three-reference Qwen Turbo editing at 512×512, optionally
followed by another edit using the first output. It uses an existing Viggle
six-step adapter at strength 1, GPU, and DiT cache off. This is a pinned Turbo
example, not the schedule for ordinary LoRA adapters.

```sh
python3 examples/local_api/qwen_reference_workflow.py \
  --socket /the/app/socket \
  --model /absolute/Qwen-Image-2.1 \
  --turbo-lora /absolute/viggle-turbo-6step.safetensors \
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

API jobs currently remain in their service history; they are not automatically
inserted into the App's creation history. You can import a completed output as a
reference through the App. Automatic installation discovery, shared App history
and a persisted workflow scheduler are not part of this protocol revision.

## Troubleshooting

Start with `models` and `service_status` to check connectivity without loading
weights. Use `plan` to inspect a request before submitting it. If the socket is
unavailable, confirm the service is running and use the path shown in the App.
Another process cannot use the same state directory or socket concurrently.

Actual generation requires compatible local weights and macOS permission to
access Metal. A successful connection or plan does not certify model output.
Use unique output paths; stop an App-owned service through its local API page.
