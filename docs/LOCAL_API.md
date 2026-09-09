# Local API / 本地 API

TurboCider exposes a persistent job service over a user-only Unix socket.
The App's **本地 API** page starts an owned service and shows its socket path.
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

## Python client

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
    time.sleep(0.5)

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

## Validation

```sh
make build-app
make test-api
python3 tests/native/test_service.py --model /absolute/FLUX.2-klein-4B \
  --output /absolute/new-test-output
```

The first tests use temporary state and no model weights. The last test uses
existing local weights to verify real generation, changed-seed prompt reuse,
queue/active cancellation, GPU ownership and crash recovery. None downloads
models. macOS must permit local socket and Metal access.
