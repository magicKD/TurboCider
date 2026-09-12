# Transformer heterogeneous parallelism laboratory

Research-only TurboCider extension derived from `mac_local_ai`. The experiment
implements full causal pre-norm RMSNorm/SwiGLU blocks, with independent weights
and buffers for 1–3 layers. It does not modify production model routing.

- [Full technical report](notes/REPORT.md)
- [Protocol and implementation audit](notes/PROTOCOL.md)
- Machine-readable summary: `notes/summary.json` (generated locally)
- [All measurements](notes/measurements.csv)
- Raw measurements, exact commands, stdout/stderr: `notes/raw` (generated locally)

## Build and reproduce

Run from the TurboCider repository root. Requires full Xcode, Apple Silicon
Metal access and Core ML. Tested with Python 3.11, coremltools 8.3.0 and NumPy.
Private API executables are separate experimental targets and not linked to
TurboCider's production binary.

```sh
make -C experimental/transformer
export PYTHONPATH=/Users/kd/Documents/project/mac_local_ai/.deps/coreml
python3 experimental/transformer/scripts/sweep.py
python3 experimental/transformer/scripts/heads.py
python3 experimental/transformer/scripts/confirm.py
python3 experimental/transformer/scripts/bridge.py
python3 experimental/transformer/scripts/audit_sp.py
python3 experimental/transformer/scripts/triple_audit.py
python3 experimental/transformer/scripts/audit_head_plan.py
python3 experimental/transformer/scripts/verify_reference.py
python3 experimental/transformer/scripts/startup.py
```

Run these suites **serially**, not in parallel. Each script saves raw data and
removes its own generated models after measuring a configuration. Rerunning a
suite overwrites its same-named measurement files; copy `notes/raw` first to
preserve a previous campaign. The historical screening defaults and shape set
are written in `notes/PROTOCOL.md`.

The `steel` fused attention implementation loads
`mac_local_ai/build/steel_qwen_attention.metallib`; the drivers explicitly set
that repository as their working directory. Build it with
`make -C ../mac_local_ai build/steel_qwen_attention.metallib` when necessary.
It supports head dimension 64; dimensions 32 and 128 use MPSGraph SDPA.
The local paths are intentional and visible in raw command records; adjust
these drivers and PYTHONPATH if relocating the repositories.

`src/baseline.mm` is the preserved source snapshot. `adapt_baseline.py` adds
externally loaded CPU weights, independent synthetic stack files and timing
samples, and fixes the ANE=0 concat dereference. `add_heads.py` adds complete
head computation. `add_private.py` and `add_private_gpu.py` produce separate
private implementations. Generated runners are ignored by Git and rebuilt by
Make; all modifications remain reviewable in the adapter scripts and `.inc`
files. Source provenance and third-party attribution are under `notes`.

## Analyze

Use a Python environment with NumPy and Matplotlib:

```sh
python3 experimental/transformer/scripts/analyze.py
python3 experimental/transformer/scripts/architecture.py
python3 experimental/transformer/scripts/report.py
python3 experimental/transformer/scripts/render_report.py
```

`analyze.py` rebuilds CSV/JSON and SVG/PNG figures entirely from raw results.
Confidence intervals use hierarchical moving-block bootstrap (4 adjacent
pairs, 1,500 draws, fixed seed). They describe this campaign, not hardware
variability across machines or operating systems.

## Git tracking policy

Keep source scripts, the baseline and `.inc` files, Markdown documentation,
the published Markdown/HTML report, CSV measurements, and SVG figures.
All JSON and JSONL files in this experiment directory are local artifacts and
are ignored by Git, including raw measurements, summaries, and metadata.
Run the experiment suites above before running the analysis scripts to generate
measurement data and summaries locally. JSON links in the historical report
require the corresponding local files; a fresh checkout does not include them.
Reruns produce new observations and need not match the published timings.

Do not track binaries, generated `src/runner*.mm`, model/weight caches, Python
bytecode, diagnostic logs, or duplicate PNG previews. The root `.gitignore`
already covers Python caches and logs; this directory only adds local rules.
This JSON policy applies only to `experimental/transformer`; repository-wide
configuration and validation JSON files are unaffected. Future scratch runs
can also use the root `outputs/` directory (already ignored).

## Scope

The model uses synthetic, variance-scaled FP16 weights; it is an architecture
and scheduling experiment rather than a pretrained language quality benchmark.
Sequence splitting here means **FFN token rows**, while attention retains all
causal keys. Complete attention-head splitting is measured separately.
Public `CPU_AND_NE` excludes GPU execution inside Core ML; compute plans report
preferred devices but do not constitute an Instruments hardware trace.

`cleanup.py` is the final campaign cleanup entry point. It checks executable
cache birth times against this campaign and preserves preexisting global Core ML
caches. Run after all hardware workers exit; see `notes/raw/final_cleanup.json`.
