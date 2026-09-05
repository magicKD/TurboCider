# TurboCider Architecture

TurboCider separates the product into five stable layers:

1. The SwiftUI app and Swift SDK only speak the versioned local HTTP API.
2. The Python control plane validates requests, selects an execution plan, owns jobs, persistence, cancellation, and API security.
3. Engine adapters translate the common request into H3, LTX-2.5, FLUX.2, or FastMetal contracts without moving their optimized compute graphs into the control plane.
4. Native Metal, MLX, and Core ML runtimes own model execution. ANE placement is declared per model, shape, quality tier, and persistence requirement.
5. The model lifecycle layer pins upstream revisions, estimates disk writes, downloads only engine-compatible files, invokes the native repositories' conversion tools, creates stable compiled caches, and records a device-bound receipt.

FLUX.2 uses a JSON-lines worker process for persistent requests. One worker is keyed by model, precision, placement, and ANE manifest, so model weights and Core ML sessions survive across jobs. H3 and LTX remain isolated native subprocesses because their current engines already own the relevant Metal/Core ML lifetime and expose batch-oriented commands. FastMetal currently uses an isolated streaming worker around its MLX prompt-to-video entry point; the adapter captures engine phase metrics and preserves its fixed DMD schedule without moving the DiT into the control plane.

Non-persistent workers may emit one `turbocider_result=<json>` line. The process
runner captures that structured result without changing normal stdout logging,
allowing job records and benchmarks to expose engine-owned phase timings rather
than treating all subprocess wall time as model execution.

Explicit `gpu_ane` requests fail closed. `auto` only selects production plans whose files, shape gates, quality level, and persistence requirements match the request.

## Public contract and engine feature surface

The versioned request deliberately separates portable fields from native
engine controls. Task, prompt, input assets, output shape/media, sampling, and
execution policy have common typed representations in Python and Swift.
Advanced controls remain recursive JSON under an engine namespace. The H3
adapter maps common media inputs and frequently used optimization fields, then
passes `engine_options.h3.args` and `.env` through to the native executable;
LTX accepts its supported environment controls; FLUX.2 and FastMetal map their
runtime-specific precision, cache, attention, decode, and ANE settings. This
keeps old clients stable when an engine adds a new experimental option.

Adding another checkpoint that uses an existing engine normally requires only
a model pack with capabilities, relocatable configuration candidates,
preparation recipes, and execution plans. A new runtime family publishes an
`EngineAdapter` class through the `turbocider.adapters` Python entry-point
group. The App and SDK do not need an engine-specific transport because they
discover public model capabilities through `/v1/models` and plan eligibility
through `/v1/plans`. See `EXTENDING.md` for the extension and admission
contract.

The App uses model-pack recommendations, including persistent-worker policy,
as initial values rather than hidden hard-coded execution contracts. It
consumes `/events` as an SSE stream and
falls back to job polling if the stream is interrupted. The same recursive
engine-options object is available in Swift code and in the App's advanced JSON
panel.

## Jobs and process lifetime

The daemon owns job and worker lifetime. CLI `generate --detach` submits to an
already running daemon, so returning from the CLI cannot cancel the job.
Attached CLI, Swift SDK, and App clients all observe the same persisted record.
Progress updates are periodically written atomically; completed jobs survive a
daemon restart. A record that was queued or running when the daemon stopped is
converted to a failed `interrupted` terminal record during recovery, because a
native subprocess cannot safely be assumed to still belong to the new daemon.

Persistent workers are an execution-plan property rather than an API-client
property. FLUX.2 worker identity includes every option that changes model or
runtime state, preventing a warm worker configured for one precision,
attention implementation, cache policy, or ANE manifest from servicing an
incompatible request.

## Model and cache lifecycle

Preparation recipes live with model packs rather than in App code. The
control plane expands relocatable paths, selects a Python environment that
actually contains the dependencies required by the planned operations, and
executes ordered actions. Downloads use exact Hugging Face commit revisions.
Partial Hugging Face downloads remain resumable; conversion tools retain their
own identity and cache validation.

The four backends deliberately keep different artifact contracts:

- H3 merges its pinned Turbo LoRA into the released FL2VA transformer and
  exports request-shape-specific Core ML MLP partitions. `--h3-rows` is
  mandatory because a different sequence length is a different ABI.
- LTX exports compiled Stage-1/Stage-2 video MLP and text K/V blocks from the
  exact Comfy INT8/ConvRot transformer consumed by `ltx-mac`.
- FLUX.2 first creates `.mlpackage` blocks, then derives a runtime manifest of
  content-addressed `.mlmodelc` directories. Stable paths allow Core ML's
  device-specialization cache to survive process launches. The engine resolves
  the local checkpoint as 4B, 9B, or 9B-KV from its transformer configuration
  (or an explicit model-pack declaration) and rejects architecture conflicts.
  Exporters support single-file and indexed sharded safetensors, derive hidden
  and MLP widths plus the single-block count from tensor headers, and stamp the
  resulting dimensions into manifests. Runtime selection includes those
  dimensions and block coverage as ABI gates, not just sequence length.
- FastMetal downloads a pinned MLX checkpoint and compiles 30 fixed-shape Core
  ML FFN prefix models. Its validated hybrid ABI is 32,760 rows, hidden size
  1,536, 4,096 ANE intermediate channels, and 4,864 GPU channels. The
  remaining attention and FFN suffix stay in compiled MLX graphs.

FastMetal's hybrid plan is intentionally device- and shape-gated. It is only
eligible for the measured `apple-m4-max-40gpu-64gb` profile at 832×480×81;
`auto` falls back to the portable GPU plan elsewhere, while explicit
`gpu_ane` fails closed.

LTX and FastMetal hybrid plans also gate FPS as part of their fixed-shape ABI.
H3's explicit experimental hybrid route performs an adapter-level second
check: an unrelated environment variable cannot satisfy it, and the request
must select an existing Core ML/private-ANE artifact or a checkpoint-backed
private projection. GPU plans remove inherited `H3_COREML_ANE_*`,
`H3_PRIVATE_ANE_*`, and `LTX_ANE_*` variables before launch so a supposedly
pure-GPU benchmark cannot silently inherit hybrid placement.

`TURBOCIDER_MODELS_DIR` makes managed artifacts relocatable. The Swift app
continues to keep writable state outside its signed bundle; bundle resources
contain recipes and scripts, never model weights.

Architecture recognition does not itself make a model production-supported.
In particular, Klein 9B has no model pack until its gated license permits the
intended use and real 64 GiB GPU, GPU+ANE, parity, performance, and quality
admission checks have passed.

## Progress contract

Adapters declare weighted phases and expected cold/warm durations. Engine
stdout/stderr remains the source of truth for completed steps. Unstructured
diagnostic lines update the job log but do not replace the last recognized
phase. Jobs expose elapsed, estimated total, and estimated remaining seconds
through polling and SSE, so CLI, SDK, and App all observe the same state.

The benchmark layer supports media outputs, directory-valued native artifacts,
nested phase metrics, and opt-in external tensor comparators. This permits
exact same-backend checks and approximate cross-backend checks to share one
auditable JSON report without assuming every engine produces an MP4 or PNG.
