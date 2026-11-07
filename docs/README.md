# TurboCider documentation

Start with the [user guides](public/README.md) to build and run TurboCider.
This directory also holds architecture decisions, experiment records and
historical implementation reports.

| I want to… | Entry point |
| --- | --- |
| Build, install models or use the App, CLI and SDK | [User guides](public/README.md) |
| Compare published performance and its limits | [Benchmarks](benchmark/README.md) |
| Understand the architecture and proposed changes | [Design documents](design/README.md) |
| Find an experiment, reproduce it or record a result | [Experiment archive](experiments/README.md) |
| Check implementation history and qualification limits | [Status reports](status/README.md) |
| Learn optimization mechanisms and their tradeoffs | [Techniques knowledge base](../techniques/README.md) |
| Contribute a change | [Contributing](../CONTRIBUTING.md) |

## Where new documents belong

- `public/`: maintained instructions for users. Describe executable behavior.
- `design/`: architecture, interfaces and proposals. State what is proposed
  versus implemented; link to experiments for measurements.
- `experiments/`: the canonical home for new experiment narratives and small,
  portable evidence. Follow the [recording guide](experiments/README.md).
- `benchmark/`: curated comparisons published to users, with links to evidence.
- `status/`: dated implementation, integration and release checkpoints. Link
  experiment details instead of copying them into a second report.
- `../techniques/`: reusable explanations, references and small teaching examples;
  raw experimental data does not belong in this knowledge base.

Dates and hardware in historical records are part of their scope. A successful
old run does not qualify the current code, a different model or another device.
Large outputs, weights, traces and machine-local logs stay outside tracked
documentation, normally under the ignored root `outputs/` directory.

## Incremental archive migration

Existing records are being consolidated one topic at a time. The
[experiment archive](experiments/README.md) indexes records that still live in
their original directories. Move a report together with its small evidence and
update its links in the same PR. Keep a short forwarding page when older paths
have many incoming links. Do not delete negative results or rewrite historical
measurements as part of a directory cleanup.
