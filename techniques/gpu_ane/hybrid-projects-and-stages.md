# 协同项目案例：branch、stage、private backend 与模型生命周期

[专题导航](README.md) · [切分与调度](partition-and-scheduling.md) · [来源](sources-and-coverage.md)

vpipe 和 ccv 分别有 [独立源码报告](vpipe-runtime-weights.md)、[rowwise/IOSurface 报告](ccv-rowwise-iosurface.md)。本页补充其他模型级方案，并明确阅读深度。

## FLUX.2：模型图决定并行机会

本库已固定读取 mac_local_ai、mac_transformer、flux2-engine 与 mflux，工作树 dirty 状态和文件 SHA 在 [既有来源账本](../references/gpu-ane-sources.json)；完整导读见 [项目对照](../references/gpu-ane-projects.md)。这几项是本地 checkout 来源，不虚构公开 GitHub 地址；本轮复核既有记录，不冒称重新审计全部工程。

其差异值得保留：mac_transformer 以 paired intermediate-channel MLP、可选 CPU/ANE 和完整 join 为中心；mac_local_ai 强调 consumer stride、paired segments、resource identity 与持久 binding；flux2-engine 在 FLUX.2 parallel block 中用 GPU attention、ANE MLP 与可选 GPU complement，并保留 bucket 与持久 backend。

联合 QKV/MLP projection、两支 concat 后输出投影、modulation/residual 的具体代数在 [切分页](partition-and-scheduling.md)。`mx.async_eval`+同步 Core ML 只在已启动 GPU 工作且依赖合法时才可能重叠。packed Q4 slices 要按 group 对齐，session 输出 slot 不可被两个请求同时覆写。

## Splash：GPU 模型引擎与历史 ANE 实验分开

[incoai/splash](https://github.com/incoai/splash) 的公开主线定位不能替代特定历史 ANE/W8A8 分支证据。本库 [Splash 机制页](../references/splash.md)、[W8A8 导读](../references/splash-w8a8-walkthrough.md)保留了版本化技术学习路径。

重点是 model-specific Metal、内存计划、prefix/cache、FFN 候选与完整 source identity；W8A8 offload 需要 calibration、GPU stage、ANE call、restore/join、并发和失败回收整条链。不能从 README 的“Metal/快速推理”标题推断默认 GPU+ANE 并行，也不拿 LLM 检索质量替代图像生成质量。

## whisper.cpp：公共 Core ML encoder 是按阶段分工

固定 [`0c7be696…/src/coreml/whisper-encoder.mm`](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm) 全文件已读：构造 Core ML configuration，读本实际是 `MLComputeUnitsAll`，不是强制 ANE；输入借用 host log-mel pointer，输出 `memcpy` 回 caller。

这种设计把稳定 encoder 作为可转换 stage，decoder/其他工作留在原 runtime。encoder→decoder 在单请求上有依赖，不能只凭两种后端宣称同时执行。多段独立音频可另建有界流水，须记录每段 state/时间戳、队列和最终 flush。Core ML 的 CPU 对照提速不证明超过优化 Metal encoder。

## Apple Stable Diffusion / ANEMLL：生命周期比峰值更重要

[`ml-stable-diffusion@ea2805d…`](https://github.com/apple-aiml-research/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2) 的 stage pipeline、attention 表达、模型 chunk/load/unload 是重要学习入口。图更快却每次重载更多模型，完整生成仍可能更慢；旧 UNet 方案也不能不经验证迁到新 DiT。

[`ANEMLL@f4ad26d…`](https://github.com/Anemll/anemll/tree/f4ad26d061dc2e426faa19f81c367343bdeb9f0d) 的 `llama_model.py` 读本包含 FP16 配方、Conv LM head、vocab split 和 state/KV 设计。应分别测 prefill/decode 的 shape、压缩、state 更新与跨 chunk 语义。其模型专用部署方法可借鉴，不推导为所有 DiT FFN 的最优方案。

[`apple/coreai-models@90cc644…`](https://github.com/apple/coreai-models/tree/90cc644367f27338c1ed512b9c4b68eb513d4317) README（D）提供导出配方、Python primitives、Swift runtime；要求 macOS/iOS27、Xcode27。它不是 ANE 专属；本机 macOS26.6.2/SDK26.5 没有运行此路线，不升级产品最低版本。作为新公共框架入口与旧 Core ML 分开评估。

## 新补充：ggml-ane 的可借鉴点和源码风险

固定 [`arozanov/ggml-ane@fecc9c9…`](https://github.com/arozanov/ggml-ane/tree/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0)，已读 `ggml/src/ggml-ane/ggml-ane.mm` 的 MIL/cache/compute/supports_op 和 bridge compile/load 片段。它不是 GGML Metal 自动使用 ANE，而是额外 private backend：仅接部分 2D MUL_MAT，权重转 FP16 常量，以 Conv1×1 包装，图内 transpose，IOSurface I/O；其他 op 要由外层 scheduler 管理。

值得借鉴：backend capability 与 graph_compute 分离、compiled-kernel cache、失败返回。不能直接采信 README 的“无 CPU 参与”：实际源码有 CPU 量化解码/FP16 转换、输入/输出复制和 host 调用。

所读 commit 还暴露必须独立验证的合同缺口（源码观察，非本机 bug 复现）：

- cache key 是 M/K/N+weight pointer；若地址复用或原地修改权重，没有内容 generation 便可能用旧常量。
- `supports_op` 的尺寸检查与 compute 内部的 16384 上界不完全一致，应使宣称能力与执行条件一致。
- compute 失败后可能 `continue`，最终只在 attempted>0 且 computed=0 时失败；部分成功不应允许未经确认的输出被当作全部完成。
- 已读输出分支只显式写 F32；完整 dtype/stride/host buffer 合同要与 scheduler 一起验证。

因此本专题将其保留为研究案例，不默认安装、执行或认为已具备通用可靠 GPU+ANE 调度。它也说明为何必须读源码和负控，不能只摘录上游 TFLOPS。

## 采用优先级（工程判断，不是速度排名）

先做优化 GPU baseline 与公开 stage offload；有足够固定形状复用，再做 public Core ML 封闭 FFN/branch；若私有路径能回答明确研究问题，再隔离测试。依次确认完整 source、数值、交接、cold/warm、驻留、整请求和失败回收；任一步不满足都保留 fallback。
