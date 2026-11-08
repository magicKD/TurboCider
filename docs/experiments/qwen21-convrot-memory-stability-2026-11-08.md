# ConvRot streaming generation under macOS memory warning

The desktop App previously collapsed Darwin warning and critical pressure into
one boolean and required normal pressure before continuing. Its 15-second wait
was shorter than the approximately 41.7-second recovery recorded in the source
hash campaign. Consequently, a request could fail before inference despite
having enough free and reclaimable file-backed memory. The user's actual
observation was free=1489 MiB, non-speculative file-backed=7385 MiB,
process=207 MiB, pressure elevated. This was an admission failure, not evidence
of malformed weights or a W8A8 numerical failure.

The user explicitly requested allowing macOS to manage ordinary memory warning
instead of waiting for normal pressure. This change supersedes the earlier
normal-only policy; the older performance campaigns and their original gates
remain unchanged.

## Implementation

- Decode Darwin flags separately as normal=1, warning=2 and critical=4.
  Unknown/unavailable observations remain unavailable. Do not use XNU's internal
  pressure enum as these flags. See Apple's [pressure notification description](https://github.com/apple-oss-distributions/xnu/blob/main/doc/vm/memorystatus_notify.md).
- Admit warning requests against the existing free plus bounded file-cache
  credit, with the full 4 GiB system reserve and actual allocation requirement.
  Optional cached DiT layers are disabled/evicted at warning; the original
  authenticated fixed weights and two refill buffers remain required.
- Recheck the selected optional prefix before its first allocation. If available
  memory changed after planning, fall back to the same minimum plan and source
  lease. The mandatory bank must still pass its own physical admission.
- At each drained step boundary, recheck the complete 512 MiB growth requirement
  after optional eviction. Previously the follow-up check used zero growth.
- Only critical pressure or insufficient reserve/growth enters a cancellable
  recovery wait, bounded to 120 seconds using a monotonic clock. A recovered
  sample must remain admissible for 500 ms. An initially admissible warning
  starts immediately. Observation errors are not treated as recoverable pressure.
- Publish `memory_wait`/`memory_ready` events. The App displays Chinese waiting
  and timeout text while retaining native diagnostics and cancellation semantics.
- Destroy request-local arrays and Tiny preview owners before the final GPU
  drain/cache clear, on success, cancellation and failure. Restore the previous
  allocator cache limit afterward. Cleanup errors remain visible.
- Include the previously qualified decode-only Original VAE residency change.
  Validate all original metadata before dropping unused lazy owners; decoder
  arithmetic, source coefficients and final precision are unchanged.

Encoder and DiT still use original ConvRot W8A8 on M5+; Float-A remains available
for older hardware. No GU fusion or private ANE prototype was added here.

## Verification and limitations

Native/CLI and Swift App builds passed, along with VAE metadata/ownership tests,
streaming plan/source tests, native Darwin admission tests, Studio behavior,
Tiny App contracts and RunInsights tests. Deterministic CPU recovery cases
cover the user's warning sample, critical recovery, reserve/growth boundaries,
unknown flags, malformed observations and deadlines. Actual Metal synthetic
tests passed both Float-A and W8A8 streamed/resident numerical comparisons,
optional eviction and reader retirement. This is not an M4 hardware test.

The final runtime is
`tc-runtime-build-v1-d3ed43edff20d35f3b7b8c49301c5eae5ffc009e20f30f54f782f2d9386fbec5`.
All 685 native source inputs, 386 MLX dependency bindings, Swift inputs, 17 Mach-O
files, signing checks and fresh-process C API identities were verified.

Real-model checks used the downloaded original Turbo INT8 ConvRot DiT and
Qwen3VL INT8 ConvRot encoder, 512×512, Turbo8, seed 73, pure GPU, streaming and
a requested 2 GiB weight budget. They had no external normal-pressure wait:

| Check | Result |
| --- | --- |
| Controlled `MLX_ENABLE_TF32=0` CLI | Four complete tensor payloads and final PNG match the qualified original golden. |
| App-state lifecycle, pressure warning throughout | Baseline, preview, cancellation at preview step 2, then Tiny final generation all passed. |
| Original final VAE with Tiny preview | Preview steps 2/4/6/8; full final PNG unchanged. |
| Actual dist App, Computer Use | Three successful clicks: 12.1008, 11.3006 and 11.1274 seconds as recorded by App jobs. |
| Actual GUI repeat | Same Chinese prompt/seed produced identical complete PNGs with different admitted optional prefixes. |
| Product-default math | Actual GUI PNG equals a separate default-environment no-preview CLI PNG; preview on/off preserves all four tensor payloads and PNG. |
| Restart | New App reopened with all history and original user prompt preserved. |

The product-default check matters: the controlled benchmark explicitly disables
MLX TF32, whereas the normal App uses MLX's existing default. A cross-configuration
GUI/golden byte check failed (relative pixel L2 0.01007, correlation 0.999833).
That failed check is retained; it is not relabeled as byte-exact. The separate
same-configuration product-default check above passed. No global math default
was changed as part of this memory fix.

The App-state run sampled a 4,038,067,520 B peak physical footprint, 128 KiB
system swap-in and zero swap-out. The actual GUI observation sampled
4,497,639,032 B peak, 64 KiB system swap-in and zero swap-out; its maximum sample
gap was 105.1 ms. Completed requests retained about 262,666 B of MLX payload;
this excludes App UI, driver and system memory. The first separate CLI trial
observed 264,306,688 B of system swap-out. These are host-wide traffic counters,
not attributable App swap; this is not a zero-new-swap performance qualification.

This stability acceptance used the pre-existing App observer ceiling of 8 GiB
and 256 MiB global swap-out emergency threshold. It does not modify the separate
older 4/5 GiB CLI diagnostic campaigns. Timings above include their stated
request/UI work; they do not establish a new performance speedup.

## Delivery and preserved evidence

The tested App and CLI replaced `dist/TurboCider.app` and `dist/cli`. The previous
distribution was retained under the experiment output directory. App copying
uses macOS `ditto` so resource forks and nested data-file signing attributes
survive; an initial `shutil.copytree` signing failure was preserved before any
old distribution was moved. No user history, model weights or other App was
removed. The final App was left open.

Local evidence, intentionally untracked, is under
`outputs/convrot-stability-20261108/`:

- `attempt-1794105618986775000/DELIVERY.json`: build, source and package binding.
- `cli-attempt-1794106155254537000/report.json`: controlled CLI exact outputs.
- `app-state-attempt-1794106185016836000/report.json`: warning-only lifecycle.
- `gui-attempt-1794106448923529000/report.json`: actual GUI jobs and resource evidence.
- `product-default-attempt-1794106801483919000/report.json`: default math exactness.
- `installed-app.json`: installed runtime, backup path and GUI qualification.

No new goal work was pushed to the remote repository.
