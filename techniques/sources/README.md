# 固定版本来源卡：源码事实与本地结果分开

[总目录](../README.md) · [图与布局](../ane/coreml-graphs.md) · [研究工作台](../review/research-campaign-20261015.md)

本页由`scripts/generate_source_index.py`从既有`references/sources.json`生成；章节通过catalog的`source_ids`复用同一登记。没有第二份手工章节清单。核心学习代码仍在`labs/`，来源卡不镜像整个上游仓库。

| 来源 | 固定commit | 阅读/运行证据 |
| --- | --- | --- |
| [apple-aiml-research/ml-ane-transformers](repos/apple-ane-transformers-ffn.md) | `590eb35913622cdef217ebc5329047aab389ac98` | not-run |
| [Anemll/anemll-profile](repos/anemll-profile-safety.md) | `bf8528105a0e61fda2c80d2c9b19615ecab9b7fa` | not-run-unmodified; safety hold |
| [liuliu/ccv](repos/ccv-coreml-rowwise.md) | `66dc16a3a3b688adfa6dc1d19eeeb3071beeb701` | not-run |
| [apple/coremltools](repos/coremltools-compute-plan.md) | `a207b1a3452eafe02f2d870f486287834af2c82d` | installed coremltools 8.3.0; this Python file has the same SHA as the checkout, not a claim of whole-package equivalence |
| [ggml-org/llama.cpp](repos/ggml-metal-layouts.md) | `64e9bceb2c3a856efed96feda784a50947049feb` | two dequantizer excerpts executed via public MLX JIT; whole backend/engine not run |
| [tmc/gputrace](repos/gputrace-timing-provenance.md) | `3d7d32e66605ba994fb0765bf9f287682bd7ccc7` | isolated CLI built; internal/timing and internal/trace tests passed; stats/timing ran on 10 owned captures; no replay/GUI qualification |
| [ml-explore/mlx · affine](repos/mlx-affine-dispatch.md) | `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` | installed MLX 0.32.0 APIs executed; binary hashes recorded, not a reproducible-build equivalence claim |
| [ml-explore/mlx · SDPA](repos/mlx-sdpa-dispatch.md) | `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` | installed MLX 0.32.0 APIs and actual-input replay benchmark executed; not source-built backend or physical dispatch qualification |
| [tgo-app-dev/vpipe](repos/vpipe-runtime-tiling.md) | `c2989a76841888e4d8d208fad4f476c487710962` | not-run |

## 官方API与SDK来源

| 来源 | 固定读取身份 | 本地证据边界 |
| --- | --- | --- |
| [Apple Metal public APIs](official/metal-counters.md) | macOS SDK 26.5 headers; docs read 2026-10-16 | owned capture and stage samples; new same-run clock brackets/windows validated on this configuration, not cross-device or MLX kernel qualification |

源码存在、局部数值通过、消费者性能通过与真实请求通过是独立状态。原创MIL实验验证一个机制，不等于验收整个上游项目。各文件SHA、阅读范围和许可边界在来源卡与登记中保留。

```sh
python3 -B techniques/scripts/query.py 'K tiling' --type source --limit 3
python3 -B techniques/scripts/get_page.py ane/coreml-graphs.md --follow-sources --max-bytes 32000
python3 -B techniques/scripts/generate_source_index.py --check
```

来源卡是原创概括，不分发上游大段正文或代码。各项目及其第三方组件继续遵循原许可，不能用本目录许可覆盖。commit和文件SHA是实际阅读身份，活动网页仅供补充导航。
