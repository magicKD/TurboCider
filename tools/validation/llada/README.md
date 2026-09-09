# LLaDA Python oracle

`python_worker.py` retains the historical JSON-lines reference protocol for
development comparisons. Run it explicitly with its documented CLI arguments
and a development Python environment (`python_worker.py --help`).

It is not packaged or called by the production library. The old
`TURBOCIDER_LLADA_REFERENCE`, `TURBOCIDER_LLADA_WORKER`,
`TURBOCIDER_LLADA_PYTHON`, and `TURBOCIDER_LLADA_SOURCE` environment variables
no longer switch a native session into Python. The standalone reference and
quality benchmark programs remain in `tools/native/benchmark_llada_*.py` and
are not App runtime dependencies.

Native sessions also ignore the retired `TURBOCIDER_LLADA_EAGER_BLOCKS`,
`TURBOCIDER_LLADA_REFERENCE_CONDITIONING`,
`TURBOCIDER_LLADA_REFERENCE_ROUTER_DIR`, and
`TURBOCIDER_LLADA_REFERENCE_QUERYFORMER_DIR` switches. Production inference
always computes its own conditioning, router choices, and QueryFormer output.
These switches no longer inject reference tensors into production inference.
Use the explicit request dump option to capture native intermediates and compare
them with the standalone Python oracle; intermediate replacement is no longer
a supported production diagnostic. GPU-only execution retains the existing
default compiled-block policy.
