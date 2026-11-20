# ConvRot GPU generation without memory pre-wait

The user explicitly requested removing the remaining available-memory wait:
clicking Generate should start loading and inference and let macOS service
allocations. This supersedes the previous 120-second recovery policy documented
in `qwen21-convrot-memory-stability-2026-11-08.md`. Its earlier evidence remains
historical; this change does not relabel those experiments.

## Required allocations versus optional cache

Ordinary public ConvRot GPU streaming now selects
`StreamingAllocationAdmission::SystemManaged` for both encoder and DiT. It
allocates the bounded mandatory working set directly through MLX/macOS:

- Remove the recovery class, its polling loop and all request/phase/step calls.
  Native generation no longer emits `memory_wait` or `memory_ready`.
- Skip host-pressure, 4 GiB reserve and 512 MiB growth heuristics for required
  bank, slot, sparse embedding and Float-A coefficient allocations. Do not
  replace waiting with immediate heuristic rejection. The direct required check
  returns before collecting a host-memory observation.
- Preserve actual allocation failure handling, non-null buffer checks, typed
  tensor geometry, exact allocator capacity checks and managed weight budgets.
  No failed allocation is treated as successful output.
- Continue using memory observations for optional cached layers. Under warning,
  critical pressure, missing observations or insufficient estimated headroom,
  reduce/evict the optional prefix and continue with the original two slots.
  Optional cache allocations still use Guarded admission and never exceed the
  requested budget.
- Preserve source authentication/generation checks, original weights, M5+
  W8A8 math, older-chip Float-A math, GPU reader fences, cancellation and final
  owner/cache cleanup. Actual GPU/IO completion waits remain necessary; this
  change removes the extra memory-admission wait.

`Guarded` remains the default for other internal callers and the separate
private Runtime ANE diagnostic. The system-managed mode is explicitly selected
by the bounded public GPU route, whose existing request validation restricts
generation to at most 512×512 and the supported model recipe. Legacy App error
presentation remains available for old task records; it is not a new-generation
waiting path.

## Validation

The runtime built successfully as
`tc-runtime-build-v1-1cae1ffd2f04458d90f0a9238ad8d5316f889b87603f236ef7b1deb88331f9de`.
The independent package checks verified 684 native source files, 386 MLX
dependency bindings, Swift targets, 17 Mach-O files, signing and fresh-process
C API identities for raw, App and CLI libraries.

CPU policy tests cover normal/warning/critical pressure, unknown/missing/invalid
observations, reserve/growth shortage and invalid policy values. SystemManaged
permits required allocations without waiting; Guarded and optional caching keep
their original decisions. The runtime check is tested with maximum accounting
inputs without allocating payload. VAE metadata/ownership, source/cache plans,
Darwin observations, Studio, Tiny App contracts and RunInsights also passed.

Actual Metal synthetic tests passed Float-A and W8A8 streamed/resident equality,
cache eviction and complete reader retirement using the new required-allocation
policy. This is an M5 host test, not an M4 hardware qualification.

Real model tests used the original downloaded Turbo INT8 ConvRot DiT and
Qwen3VL INT8 ConvRot encoder, GPU, streaming, 512×512, eight steps, seed 73 and a
requested 2 GiB weight budget. No external normal-pressure wait was used:

| Check | Result |
| --- | --- |
| Controlled TF32=0 CLI | All four complete tensor payloads and final PNG equal the qualified original golden; request 11.8098 s. |
| Product-default App-state baseline | Succeeded in 10.9145 s and matched the qualified product-default PNG. |
| Original final VAE with Tiny preview | Succeeded in 10.3454 s; steps 2/4/6/8; final PNG unchanged. |
| Cancellation then another generation | Cancelled at preview step 2, then Tiny final generation succeeded in 9.2193 s. |
| Actual dist App via Computer Use | Three consecutive successes: 12.0589, 11.3314 and 11.2912 s. |
| Actual GUI correctness | Same-prompt repeats equal each other and the preceding App version; changed-prompt PNG equals the product-default reference. |
| Immediate progress | First GUI observation showed loading at displayed elapsed 0.1 s; subsequent clicks showed text encoding at 0.7 s. No memory wait appeared. |

The native receipt reports `allocation_policy: system_managed` independently
for both encoder and denoiser. Each successful full request executed 252 encoder
and 1536 DiT W8A8 projections. The App-state test explicitly rejects either
memory-wait event in its phase sequence and checks the actual allocation-policy
receipt. This verifies native behavior in addition to UI presentation.

The complete actual-GUI resource observation had 1783 samples, maximum gap
105.1 ms, peak physical footprint 4,107,372,200 B and zero observed new host
swap-in/out traffic. Pressure samples included normal and warning. Successful
requests retained only about 262,666 B of MLX payload after cleanup; this excludes
the App UI and driver. The CLI and App-state observations also saw no new swap
traffic. Existing swap allocation was not zero and is not a failure condition.

The validation observer retained the separate 8 GiB App footprint ceiling. Swap
traffic was recorded, without making it an admission condition or silently
changing the older performance campaigns. These are correctness/usability
measurements, not a new paired performance-speedup claim. The OS can still fail
an actual allocation; this change does not promise arbitrary workloads can fit.

## Delivery

The qualified package replaced `dist/TurboCider.app` and `dist/cli` using macOS
`ditto`, preserving nested signing metadata. The preceding distribution was
backed up. The final App was reopened with the user's original prompt, history,
Turbo model, streaming, 2 GiB budget and Tiny configuration retained. No other
App, user history or model weight was removed.

Local untracked evidence under `outputs/convrot-immediate-20261108/`:

- `attempt-1794120720609369000/DELIVERY.json`: build, CPU tests and package binding.
- `cli-attempt-1794121464987838000/report.json`: complete tensor/PNG equality.
- `app-state-attempt-1794121507581569000/report.json`: native policy, preview, cancellation and no-wait phases.
- `gui-attempt-1794121704945763000/report.json`: actual GUI jobs and independently verified resource samples.
- `installed-app.json`: installed native identity, backup and final GUI qualification.

This work is committed locally; no new changes were pushed.
