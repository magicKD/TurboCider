# 证据地图与更新规则

[总目录](../README.md) · [案例比较](../gpu-ane/experiment-ledger.md) · [机器来源清单](sources.json)

## 本轮核查范围

基于2026-10-15干净的TurboCider `ad8100f` 开始，读取根/native/backend入口、GPU/ANE/streaming/quantized设计、近期状态和机器JSON，再对照关键执行/ownership/调度/量化/IO源码。历史以Git日志和具体diff、项目相关Codex会话检索交叉定位；不是穷尽阅读所有模型文件/所有个人聊天，也不是重跑全部历史实验。

本次新增仅为知识库/辅助脚本/测试及根README导航，没有修改推理默认或模型数学。原有设计、实验记录和失败回执不重写；产生的新验证在 [本轮记录](../review/validation.md)。

## 来源优先级

1. 当前真实代码/API/actual result证明实现与执行合同。
2. 与具体binary/source/input绑定的小型机器记录和原始样本证明测量；链接到ignored本地产物不等于他人clone后拥有它。
3. 同阶段状态文档解释决定/失败/边界。
4. 设计文档只说明计划；若后来记录已否决，不能继续当当前推荐。
5. 历史聊天用于发现问题/出处，必须回到代码/记录核查；聊天里“仍在跑”不能证明今天还有liveprocess。

## 重要入口

| 主题 | 权威实现 | 观测 / 边界 |
| --- | --- | --- |
| 模型/后端当前决策 | [registry](../../native/models/registry.cpp)、[backend说明](../../native/backends/README.md) | [acceleration](../../docs/status/acceleration.md)、[10/15合并](../../docs/status/2026-10-15-main-into-dev.md) |
| FFN/LoRA/owner | [HybridFfn](../../native/backends/ane_ffn.hpp)、[ExecutorABI](../../native/backends/ane_runtime.hpp) | [jointA/B](../../docs/status/local512-qwen-joint-ab-2026-10-09.md)、[downrank](../../docs/status/local512-split-down-ranks-2026-10-08.md) |
| 调度与校准 | [RowScheduler](../../native/backends/ane_scheduler.hpp)、[costmodel](../../native/backends/ane_cost_model.hpp)、[integratedtrial](../../native/backends/ane_channel_calibration.cpp) | [nativeauto](../../docs/status/native-channel-auto-integration-2026-10-05.md)、[prefetch反例](../../docs/status/private-ane-prefetch-2026-10-03.md) |
| CoreML/Public/Private | [coreml](../../native/backends/coreml.mm)、[PublicW8](../../native/backends/ane_public_w8.mm)、[PrivateW8](../../native/backends/private/ane_w8_executor.mm) | [PublicW8实际边界](../../docs/status/public-w8-shared-ffn-2026-10-07.md) |
| 独立nativegraph | [backend](../../native/backends/metal_graph/backend.hpp)、[QwenDiT](../../native/models/qwen21/native_denoiser.cpp) | [SGMMAE2E负结果](../../docs/status/2026-10-08-native-lora-sgmma-e2e.md) |
| streaming/IO | [context](../../native/runtime/streaming/context.hpp)、[readahead](../../native/runtime/streaming/read_ahead_worker.hpp)、[SourceLease](../../native/runtime/streaming/source_lease.hpp) | [streaming目录](../../docs/design/streaming/README.md)、[量化/vpipe对照](../../docs/design/quantized-streaming-vpipe-comparison.md) |
| exact/approx缓存 | [conditioncache](../../native/models/qwen21/conditioning_cache.hpp) | [DBCache](../../docs/status/2026-09-30-qwen21-dit-cache.md)、[prefixhistory](../../docs/status/qwen21-resident-prefix-kv-2026-09-26.md) |
| VAE/媒体 | [SeedVR2VAE](../../native/models/seedvr2/vae.cpp)、[reader](../../native/media/rgb_video_reader.mm) | [VAE全景](../../docs/design/seedvr2-vae-optimization.md)、[blockIO错误](../../docs/design/seedvr2-block-weight-residency.md) |
| 正确测量 | [sampler](../../tools/native/process_tree_sampler.py)、[verifier](../../tools/native/verify_process_tree_samples.py) | [mergeJSON](../../docs/design/validation/main-dev-merge-performance-20261015.json)、[SeedVR2JSON](../../docs/design/validation/seedvr2-comfy-benchmark-current-20261015.json)、[shareJSON](../../docs/design/validation/local512-qwen-prefill-shares-20261010.json) |

## Git历史中本轮检查的接缝

`git show <commit> -- <path>`可定位真实变更，而非从commit标题猜实现：

- `1056615`：bounded GGUF read-ahead / CPU decode，读取worker的实际RAII/error/cancel合同。
- `e140a01`：SeedVR2 norm/SiLU/halo pack与phase-history，保留initial-frame FP32配方及compile-isolated范围。
- `c085b7d`：verified serial block-weight prototype，结合现有文档核对reader错误锁存与真实组件边界。
- `4cda62f`：合并后的匹配性能与cache tradeoff，核对机器JSON原始warm samples及qualification字段。

源码变更、机器回执和产品部署是三件事；本轮没有将这些旧构建重标为当前ad8100f新测量。

## 外部checkout

Splash、带ANE的splash-ffn、vpipe、mac_transformer、flux2-engine、ANEresearch、SolAttn/SmoothQuant在来源清单中以optional记录。Splash与vpipe关键已读文件另存SHA256，防止head相同但worktree内容不同；vpipe本地有未跟踪models目录，不读取或修改权重。清单不是这些项目的完整源码快照。

外部缺失时手册仍可读、CPU工具仍可运行；校验器warning提示未能复核来源，不误报已验证。网上API链接只作补充，未离线镜像、未自动校验HTTP或anchor。

## 更新一条技巧时

补适用场景/代数/layout/成本/失败/命令/资格，链接真实文件和新回执；改`sources.json`相关字面符号与需要的snapshot，而不移动旧benchmark身份。对比scope不相同就并列说明，不覆盖历史数字。运行链接/CPU测试，再按 [coverage](../review/coverage.md)人工复查。

实现扩展新增 [逐技术章节catalog](../labs/catalog.json)、[实验室源代码](../labs/README.md)、[KernelWiki/Sana设计对照](kernelwiki-sana.md)和 [Splash W8细读](splash-w8a8-walkthrough.md)。来源清单现在也检查新增独立代码的关键符号；外部snapshot仍是本轮读到的内容，不自动宣称上游最新。
