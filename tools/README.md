# Developer tools

Run commands from the repository root unless a tool says otherwise. Start with
`make help` for supported build and test entry points and the
[contribution guide](../CONTRIBUTING.md) for validation requirements.

| Directory or entry | Purpose |
| --- | --- |
| `setup_dependencies.py`, `dependencies/` | Set up the pinned build and Core ML environments |
| `native/` | Native build, packaging, model preparation and focused diagnostics |
| `coreml/` | Offline Core ML conversion and validation |
| `convert/`, `models/` | Model conversion and model metadata helpers |
| `h3/`, `ltx/` | Model-specific preparation and comparison tools |
| [profiling/](profiling/README.md) | Opt-in measurement campaigns and evidence collection |
| [validation/](validation/README.md) | Reference comparisons, evidence checks and experimental screens |
| `experimental/` | External reference-source audits and synthetic research fixtures |
| `comfyui/` | ComfyUI integration helpers |
| `branding/`, `metal/` | Branding generation and Metal diagnostics |

Placement in this directory does not make a tool part of the release package.
`native/build.sh` and `native/package.sh` explicitly select shipping sources and
resources. Research scripts can require local weights, optional dependencies or
specific hardware; read their arguments and assumptions before running them.

## Retired entry points

The old `experimental/build_h3.sh` and `experimental/build_ltx.sh` built vendor
snapshots under `experimental/video/`, which is no longer in the repository.
Use `make build` for the maintained native runtimes. The remaining
`experimental/audit_video_sources.py` and `experimental/ltx_gemma_fixture.py`
accept an explicit external reference location and remain available.

The old `cleanup.sh` removed tracked design, status and experiment directories.
It is retired without a replacement: generated build output and historical
research records have different lifecycles. Remove only a known generated
output directory when needed; preserve reviewed experiment records.
