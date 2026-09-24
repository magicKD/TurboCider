# Same-machine regression: read-only telemetry follow-up

> **Invalid for performance analysis.** The later fresh-boot control confirmed
> this trace belongs to the degraded pre-reboot system state. It remains useful
> only as diagnostic history; see `z-image-performance-ledger-2026-09-23.json`.

This is diagnostic evidence, not optimization qualification. No production
kernel, power setting, driver state, or process lifecycle was changed here.

## Throttle counters under real generation load

Trace: `/private/tmp/z-image-throttle-load-20260922/throttle.jsonl`.
55 samples span 54.515113 seconds, covering current-default 512x512/8-step
cold and warm requests. Every sample had the following unchanged values:

| Counter | First = last | Delta |
| --- | ---: | ---: |
| HWPI | 0 | 0 |
| DPE | 0 | 0 |
| Misc | 0 | 0 |
| IVDM | 11378492329 | 0 |
| Total | 11378671397 | 0 |
| IVDM Min Active Throttle Percent | 85 | 0 |
| IVDM Max Throttle Percent | 100 | 0 |

The 85/100 fields are not evidence of current 85% throttling. Counter units,
reset behavior, and private IOReport ABI semantics are not established.
Zero deltas provide no affirmative evidence of throttling during this run;
they do not prove that every possible throttling mechanism is absent.

Instrumented warm timing: denoise 14.729939 s, VAE 5.046113 s,
request wall 20.015772 s. These timings are not a fresh speedup qualification.

Earlier trace `/private/tmp/z-image-ioreport-load-20260922/gpu-states.jsonl`
showed only P15 residency advancing over 54.471091 seconds; idle also advanced
only P15. Do not translate the state name into a measured effective clock or
claim hardware is healthy solely from these counters.

## OS installation timeline

Read-only `system_profiler SPInstallHistoryDataType -json` reports macOS
26.6.2 installed August 31, 2026, before the September 7 reference run.
Command Line Tools 26.5 and 26.6 were recorded on August 11, 2026.
This does not support a simple intervening macOS-version-upgrade explanation.
Installation history alone does not establish the exact historical loaded
driver/runtime or exclude a subsequent runtime-state problem.

## Conclusion and remaining discriminator

Historical 1024x1024/9-step pure-GPU warm wall was 36.368471 s,
denoise approximately 35.44 s, VAE approximately 0.858 s. Retained old package
today measured 169.543168 s wall, 148.218079 s denoise, 20.213015 s VAE;
that 1024 run did not fingerprint all loaded dependencies. A separate old
512 audit did fingerprint old packaged native and MLX libraries and remained
slow. New DiT fusion alone cannot explain this evidence.

Root cause remains unresolved. Need independent effective GPU clock/power
telemetry and/or an authorized fresh-boot control with the same retained
binary, verified dependencies, weights, shape, steps and prompt. No reboot,
administrator operation, cache deletion or driver reset was performed.
