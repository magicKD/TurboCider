# Caches and memory

[Documentation](README.md) · [Usage reference](USAGE.md)

## Repeated prompts

FLUX and Z-Image keep text conditioning in a compatible resident session.
Changing only the seed does not require encoding again. CLI `batch` and the
local API preserve the session across requests; a one-shot process does not.
The native LTX route can also persist connected conditioning on disk with an
identity derived from model, checkpoint, tokenizer and prompt. Cache reuse is
reported by the executor rather than inferred from a faster request.

App tasks display a text-conditioning cache-hit message when the result reports a hit.
Z-Image layer callbacks update elapsed time without replacing the completed
denoise-step counter. Task details
and the model's recent-session sampling panel expose text/denoise timing, MLX memory
and available Core ML metrics. Core ML calls, prediction time and output-copy
bytes are cumulative within that session; they are not per-request counters
or an ANE occupancy measurement. MLX snapshots exclude Core ML and OS/file
caches. Peak memory is the allocator's recorded peak, not total physical RAM.

Z-Image releases encoder weights after encoding while retaining the resulting
conditioning tensor. the clear-memory-and-unload control in Settings closes the embedded
session and releases its model and prompt caches; the next request reloads.
App-owned API sessions are released by stopping the service. Original model
weights, media and source files remain on disk.

## Tensor retention

App Settings provide inventory, manual cleanup and retention of 7/30/90 days.
Automatic cleanup is **off by default**. When enabled, the running App checks
at startup and then hourly; it skips busy inference/API sessions and retries
on a later check. No background service or system scheduler is installed.
Age is measured from the newest file modification time inside an entry,
which ordinarily means generation time, not last access.

CLI equivalents:

```sh
turbocider cache inventory
turbocider cache settings
turbocider cache retain 30
turbocider cache prune 30
# Explicitly remove all recognized regenerable entries:
turbocider cache prune 0
# Disable automatic cleanup:
turbocider cache retain 0
```

`retain` saves policy shared with the App. A standalone CLI does not wake up
periodically; invoke `prune` when desired. Policy is stored in
`~/Library/Application Support/TurboCider/cache-settings.json`.

LTX connected text tensors are managed in:

- `~/Library/Caches/TurboCider/ltx-conditioning`, or an explicit
  `TURBOCIDER_LTX_CONDITIONING_CACHE_DIR`.
- The App service's `api-service/jobs/ltx-conditioning-cache` under its state
  directory (normally `~/Library/Application Support/TurboCiderNative`).

Each removable entry must have the native 64-character cache identity, matching
metadata and exactly the expected conditioning files. Unknown data, extra media
and symlink entries are skipped. Cleanup acquires the same cross-process lease
as native inference and refuses to run while the runtime holds it. Next use
regenerates removed tensors from the available original weights.

For optional FLUX/Z-Image diagnostic dumps, use the register-diagnostic-tensor-directory control in Settings
or explicitly enroll the actual dump directory:

```sh
turbocider cache register-dumps /absolute/outputs/my-run/dumps
turbocider cache dump-directories
turbocider cache inventory
# Remove this registration and retain every file:
turbocider cache forget-dumps /absolute/outputs/my-run/dumps
```

Enrollment itself deletes nothing. Registered diagnostic tensors share the
same retention policy and manual cleanup button. Enrollment is nonrecursive:
select the folder containing the tensors, not the whole research `outputs/`
tree. Only known native FLUX/Z-Image dump names (conditioning, image/initial/
step latents, noise and decoded tensors) with a valid single-`tensor`
safetensors payload are eligible. The header, dtype, shape, byte count and
absence of symlinks are checked again before removal. Missing or replaced
directory links do not silently enroll another target. Unrecognized files,
model checkpoints, nested directories, JSON records, images and videos remain.
Diagnostic tensor removal is permanent; reproducing those tensors requires
rerunning the original request. They are different from reusable prompt caches.

The enrollment list lives beside the policy in
`diagnostic-tensor-directories.json`. No existing output folder is enrolled
automatically. This command does not delete LoRA merged weights, Core ML
compilation, model-library blobs or arbitrary `outputs/` folders. Core ML
storage has separate preview/cleanup controls in the model preparation page.
