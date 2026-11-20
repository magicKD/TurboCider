# Z-Image streaming experiments

These historical records compare streaming routes on specific devices and
workloads. The original precision, budgets, admission rules and validation
limits are part of each result.

| Record | Scope |
| --- | --- |
| [BF16 explicit streaming](z-image-streaming-2026-09-15.md) | Original device, workload and memory ownership measurements, with a later link to GPU+ANE suffix reading |
| [M5 / 16 GiB INT8 GPU streaming](z-image-int8-m5-16-streaming-2026-10-28.md) | MacBook Air comparison of resident and 6/8 GiB streaming requests, with same-image output and observed paging |
| [M5 Pro / 24 GiB INT8 prefetch](../m5/z-image-int8-streaming-2026-09-20.md) | Budget and prefetch comparisons for the distinct M5 Pro profile |
| [M5 Pro generic streaming comparison](../m5/z-image-generic-int8-streaming-2026-09-20.md) | Frozen-library comparison and engine-lifecycle limits |

The original [BF16 JSON record](../../status/z-image-streaming-2026-09-15.json)
and [M5 / 16 GiB benchmark receipts](../../../results/benchmarks/z-image-int8-m5-16-streaming-2026-10-28/)
remain available at their recorded paths. See the [streaming design](../../design/streaming/README.md)
for the framework contract and the [user guides](../../public/README.md) for
current usage.
