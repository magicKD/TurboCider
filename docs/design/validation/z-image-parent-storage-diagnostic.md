# Independent parent and Metal storage controls

Read-only inference diagnostics, no production library or default changes.
Native add probe bypasses MLX/model code,32MiB input/output FP32,4 repeated
dispatches per buffer,3 excluded warmup buffers and5 measured buffers.
Reports median command-buffer GPU timestamps and validates every output.
These timings are not a Z-Image speedup qualification.

## Independent process parent

Direct probe4 from the current execution host remained slow: scalar
128/256/512-thread groups11.9572/12.1916/5.20397 ms per dispatch;
float4 at128/256/512 threads2.44000/3.21244/3.68682 ms.

Used a temporary user launchd label
`local.turbocider.gpu-diagnostic-parent-check` to launch the same binary.
`launchctl submit` unexpectedly relaunched the completed task; on observing
multiple output sets, removed exactly that registration with `launchctl
remove` and verified the service was absent and no diagnostic process remained.
No user data was deleted. Logs retained under
`/private/tmp/z-image-launchd-parent-check.{stdout,stderr}`.
Do not reuse submit as a presumed one-shot launcher.

First independent-parent output: scalar10.4009/10.4461/4.69884 ms,
float4 2.68224/2.44778/3.44237 ms. Subsequent relaunched sets remained in
the same broad range. Direct QoS21 versus launchd QoS17 were recorded, so
this is not a matched scheduling-policy comparison. No order-of-magnitude
recovery was observed, but this does not eliminate every process-policy cause
or establish full-model launchd performance. No whole model ran under launchd.

## Shared/private/shared storage screen

Extended standalone probe with a `private` argument. Private input upload
and output readback use blit commands outside timed compute, with successful
completion required. CPU verification reads back every result. Source/target
buffers remain alive for the entire probe; no production allocator changed.

Median GPU milliseconds per dispatch:

| Vector width | Threads | Shared before | Private | Shared after |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 11.9511 | 11.6986 | 11.2009 |
| 1 | 256 | 11.4672 | 11.2053 | 10.9456 |
| 1 | 512 | 4.94874 | 4.44401 | 4.66847 |
| 4 | 128 | 1.94765 | 2.41806 | 1.91370 |
| 4 | 256 | 3.19694 | 2.45588 | 2.21037 |
| 4 | 512 | 2.96046 | 2.43406 | 2.21497 |

All probes exited0 and validated outputs. No general private-storage win or
historical recovery is established; no allocator switch is justified.
No restart, administrator access, power changes or other app termination.
Historical same-machine regression remains unresolved; absolute effective
GPU clock/power telemetry and an authorized fresh-boot control remain useful
discriminators. Current relative kernel speedups remain separate evidence.
