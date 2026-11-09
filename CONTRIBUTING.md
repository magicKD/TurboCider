# Contributing

TurboCider is a native Apple silicon inference project. Start with the
[README](README.md) and [build guide](docs/public/GETTING_STARTED.md). Original project
code uses the [MIT license](LICENSE); retained third-party notices govern the
corresponding derived components. Model weights have separate upstream terms.

## Development workflow

1. Use an Apple silicon Mac and the pinned dependencies from `make setup`.
2. Create a focused branch. Preserve unrelated changes and historical documents.
3. Put inference changes in `native/`; keep model-independent orchestration
   separate from model executors. Foundation bridges belong in Apple platform
   or API layers, not portable runtime code.
4. Keep App, CLI and SDK request capabilities consistent. Advertise only
   executable operations, not unvalidated upstream features.
5. Run `make test`. For Swift changes, run `make build-app` and `make test-app`.
   Model-library/download changes also require `make test-library` (tiny local
   HTTP fixtures); local API lifecycle changes require `make test-api` (local
   Unix sockets and child processes). Build the test executables first.
   Use actual model tests when changing inference, cancellation or residency.
6. Package with `make package` before reviewing a shipping change. Verify the
   App and CLI bundle resources, not only the development executable.

## Tests and benchmarks

Use the [test guide](tests/README.md) to choose a focused suite and distinguish
host fixtures, native contracts, App checks and model/device acceptance.

Do not require full model downloads in unit tests. Use tiny local fixtures for
model installation, download, cache and request contracts. Mark tests that need
real weights, Metal, clipboard access or a specific chip explicitly.

Performance changes should identify the workload, device, precision, cache
state and baseline. Preserve raw samples locally and commit portable summaries.
Consult [performance methodology](docs/public/PERFORMANCE.md); do not convert a block
microbenchmark into an end-to-end claim.

## Repository hygiene

- Do not commit weights, generated media, access tokens, build products or local
  absolute model paths. Use sample configuration with placeholder paths.
- Keep `docs/status/` and existing research records. Add a new dated correction
  instead of deleting useful history.
- Runtime failures must explain the unsupported input or missing artifact;
  never return fabricated media or silently substitute another model.
- Changes that import third-party code must include its license and provenance
  in `native/THIRD_PARTY_NOTICES.md`.
- Regenerate branding with `swift tools/branding/generate.swift` on macOS.

## Focused cleanup and documentation changes

Work one directory or closely related component at a time. Split independent
cleanup into separate PRs; include another directory only when the same change
needs a consumer, test, build registration or documentation link updated.

Before removing code, check build/package lists, imports, dynamic entry points,
tests and documented commands. A lack of direct textual references alone does
not prove a file is unused. Preserve third-party provenance and do not mix
behavior changes with a mechanical move.

New experiment narratives belong under `docs/experiments/<topic>/`, with small
portable evidence beside them and raw runs under ignored `outputs/`. Designs
belong in `docs/design/`, user instructions in `docs/public/`, curated comparisons
in `docs/benchmark/`, and implementation checkpoints in `docs/status/`. Link to a
record instead of copying it between directories. Migrate existing experiments
one topic at a time, keeping old published links usable and updating executable
consumers when moving data files.

For documentation-only changes, check relative links and review moved content
for lost context. Report the checks actually run and distinguish them from
native builds or hardware/model validation. For code changes, follow the tests
and packaging requirements above and record any environment limitations.

Pull requests should explain the user-visible behavior, relevant validation,
and remaining limitations. For bugs, include the request with private prompts
and paths redacted, `doctor` output, and the error/progress events.
