# <Experiment question>

Date: <YYYY-MM-DD>
Status: <planned / measured / inconclusive / superseded>

## Question and decision

What hypothesis is being tested? What result would reject it? State the decision
and whether it affects a default, an opt-in route or research only.

## Reproduction context

| Field | Value |
| --- | --- |
| TurboCider commit and local changes | <full SHA; describe any uncommitted diff> |
| Hardware and memory | <chip, CPU/GPU cores, RAM> |
| OS, compiler, SDK and dependencies | <versions and source revisions> |
| Model, source revision and precision | <identity; no absolute model path> |
| Workload | <shape, steps, seed, public prompt or prompt identity> |
| Baseline and candidate | <one explicit difference; identify approximations> |
| Cache and residency | <cold/warm, retained workers, compilation included?> |
| Measurement boundary | <kernel / stage / complete request; synchronization> |
| Sampling | <warmups, repetitions, ordering, thermal/power conditions> |

## Commands and evidence

State the working directory and exact commands, using placeholders for external
weights. Link small committed evidence and identify ignored raw output under
`outputs/<topic>/<run>/`. State which artifacts are local-only and unavailable
to readers; do not link to files absent from the repository as if committed.

## Results

| Metric | Baseline | Candidate | Samples and uncertainty |
| --- | --- | --- | --- |
| <time, memory or quality metric> | <value/unit> | <value/unit> | <count/interval> |

Include quality thresholds and observed errors, memory and swap where relevant,
failures, exclusions and their reasons. Never infer whole-request speed from a
kernel measurement or merge unrelated campaigns into one comparison.

## Limits and follow-up

What remains unmeasured or unqualified? Link the relevant design, implementation,
tests and later corrections. Preserve this run's original conclusion when a
later experiment supersedes it.
