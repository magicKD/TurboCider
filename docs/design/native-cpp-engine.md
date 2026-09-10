# 原生 C++ 引擎边界与重构验收

本文件描述当前实现。早期设计中的大一统 `runtime.hpp` 已移除；本文取代它的模块划分。原始权重仍可位于项目外，构建、转换与缓存的独立性见 [独立运行设计](standalone-project.md)。

## 实现边界

| 位置 | 职责 | 语言与依赖 |
| --- | --- | --- |
| `native/core` | 请求契约、事件、异常、取消、分词器抽象 | 标准 C++，无 Foundation / MLX |
| `native/runtime` | 模型会话、描述、类型化结果、执行计划、阶段驻留策略、共享 block working-set 规划、进程执行互斥 | 标准 C++；设备租约使用 POSIX；block 规划以 C ABI 同时供 C runtime 使用 |
| `native/models` | 模型注册和约束；FLUX 文本、图像、去噪和解码流程 | C++；具体 FLUX 执行器直接调用 MLX |
| `native/backends/mlx.*` | 权重、张量操作、流配置、编译与定制 Metal kernel | MLX C++ |
| `native/backends/coreml.*` | Core ML 分区加载、预测、GPU/ANE 交接、性能统计 | C++ 接口；Objective-C++ 实现 |
| `native/backends/artifact_cache.mm`, `coreml_resources.mm` | 源模型、编译缓存、导出工具启动、磁盘资源管理 | Apple/Core ML 边缘适配 |
| `native/platform/apple` | JSON 输入输出、配置读取、设备发现、Unicode 分词适配 | Objective-C++；向模型提供 C++ 数据 |
| `native/media` | 输入图像与 PNG 编解码 | C++ 张量接口；Apple 媒体适配 |
| `native/api/c_api.mm` | 现有 C ABI、JSON 与回调协议的兼容边界 | Objective-C++，不实现神经网络 |
| `apps`, `services`, `bindings` | App、CLI、本地服务、SDK | 共用同一个引擎 ABI |

内部 `ModelSession` 返回 `LoadResult` / `RunResult`；`ExecutionPlan` 和 `ModelDescriptor` 都是 C++ 类型。Foundation 序列化只在边缘发生，不在模型中构造字典。C ABI 与现有 JSON 字段保持兼容，Swift 客户端无须迁移。

Tokenizer 保留原有 Unicode 规范化和正则表达式行为，通过 PImpl 隐藏系统对象。Core ML 的 `MLModel` / `MLMultiArray` 同样只存在于 `.mm` 实现。不能以消除所有 Objective-C++ 为目标牺牲兼容性或引入额外动态接口。

## 模型与计算后端

FLUX 的数学计算文件位于 `native/models/flux2`，使用 `.cpp` 真正按 C++20 编译；构建只对 `.mm` 启用 Objective-C++ 和 ARC。矩阵操作、精度、编译开关、block 顺序、同步点保持原实现，不因架构调整改变图像结果。

本次未实现逐算子虚拟后端。连续的 MLX 计算直接调用 C++ API，保留编译与融合能力，避免 Tensor 包装和逐算子分发。`HybridSession` 是明确的后端边界，目前仍使用 MLX Tensor 交接；它不是已经完成的跨所有后端通用 Tensor ABI。

新增后端应先为完整计算块提供实现，并证明端到端收益，再演进交接接口。当前没有新增独立 MPS 执行器，没有重新实现 MLX allocator，也没有把现有 Metal kernel 转写一遍。

## 执行与并发

`runtime/execution.cpp` 管理进程内执行互斥及跨进程 GPU 租约；MLX 流配置位于 MLX 后端。现有 App/服务队列、忙碌拒绝、取消与完成 drain 保持不变。

FLUX 混合 block 仍按以下依赖运行：输入 materialize → GPU attention 异步提交，与 Core ML MLP 重叠 → 合并 → block 完成。PImpl 封装不增加张量拷贝。Core ML 未采用输出 backing 时仍保留显式拷贝并记录字节数。

逐 block eval 也承担缓冲区复用和取消边界的作用，本次不为减少同步而删除。`cpuAndNeuralEngine` 仍不保证实际 ANE 驻留；指标继续明确驻留未知。

`ExecutionPlan` 当前描述请求、阶段依赖与内存估计，并驱动校验；它不是新造的通用图编译器。FLUX 的实际提交仍由类型化模型流水线和后端共同执行。

## 内存策略

`ResidencyPolicy` 独立计算阶段边界策略：

- resident：保留原有速度优先行为；根据机器容量和用户预算决定文本编码时是否保留图像权重。
- component_staged：文本阶段按原策略释放图像组件；图像编码后释放 VAE；去噪后释放 Transformer 和混合分区；解码后释放 VAE。
- 各阶段释放仍由资源所有者执行，MLX 管理实际 Tensor 分配。模型的 conditioning 缓存继续保留，卸载时释放。

本次补齐了输入图像编码后、去噪前的 VAE 释放。混合统计在分区卸载前采集，避免 staged 模式丢失统计。原始模型文件和转换分区不因内存卸载而删除。

预算仍为保守估算，非硬限制；MLX 内存指标不包含 Core ML/系统/文件缓存。H3 与 LTX 的 denoiser 已使用 `native/runtime/block_residency.h` 的共享 working-set 规划：H3 保留固定双槽并按预算选择 pinned prefix，LTX 使用最多三个可复用 look-ahead 槽并在预算足够时自动退化为全驻留。C++ `BlockResidencyPlan` 是同一决策的类型化视图，Session 会在返回结果前校验 C runtime 与框架计划一致；实际块读取、Metal kernel 和取消边界仍由各自 C runtime 所有者管理。请求预算仍不是整个进程 RSS 硬上限，必须继续用真实 RSS、吞吐和质量矩阵验收。

### H3 / LTX 共享流式驻留合同

`native/runtime/block_residency.c` 只负责可审计的容量计算，不持有模型权重，也不引入 MLX、Foundation 或 Metal。H3 的既有 `h3_streaming_policy` 现在只是兼容性薄包装，因此旧 C 测试和 ABI 不变；LTX loader 直接使用同一策略，并把 `pinned_blocks`、`streamed_blocks`、`refill_slots` 与 working-set 估计写入 native result。这样可以借用 vpipe 的“预算选择驻留子集、异步补块、可复用槽”思路，而不把 vpipe 较慢的算子或进程边界带入 TurboCider。

本轮的 CPU-only 回归覆盖 H3 双槽预算边界、LTX 1/2/3 槽容量阶梯、48-block 全驻留退化、溢出和取消；Apple native build 进一步验证了 C/C++/Objective-C++ 链接边界。2026-09-10 主机复测中，H3 matched DiT 仍快于 vpipe，LTX 9 帧 streamed smoke 相对 resident 更快且 latent/RGB byte-exact；这些结果只约束对应 workload，未放宽 ANE、I2V、音频或 97 帧完整矩阵门禁。

## H3 / LTX 接入合同

两者的注册、请求验证、阶段定义、Session 生命周期和结果遥测现已接入 C++ runtime。H3/LTX 数学仍保留已验证的 C/Metal 路径，新的共享组件只统一驻留决策和类型化遥测；模型级 parity、E2E 速度和 ANE 质量门禁仍分别记录，不能由框架接入本身推导通过。

后续执行器必须：

1. 实现 `ModelSession` 的类型化结果、加载、准备、生成和卸载。
2. 保持平台对象不进入模型公共头文件。
3. 描述实际阶段依赖和驻留策略，不能直接套用 FLUX 的组件结构或内存估计。
4. 验证文本及各媒体输入、取消恢复、重复加载、输出时空形状、音画同步和参考实现误差。
5. 对 streamed/prefetch 实现验证内存预算、同步与逐块输出对照；当前 H3/LTX 已完成代码和 CPU policy 验证，真实媒体矩阵仍是模型级开放条件。

`experimental/video` 仍是未进入构建的历史迁移材料，其中旧 `runtime.hpp` 引用不是可用接口。应按上述新合同移植，不能直接加入 SOURCES。

## 验收方法

- `make test`：请求合同、独立性、代码边界；CPU-only C++ 编译并运行驻留/预算/取消测试，不链接 Foundation 或 MLX。
- `make test-model`：真实权重 App 状态层三种图像任务、App/direct PNG 一致、加载卸载、取消恢复、服务队列。
- `tests/native/test_preparation.py`：预热、Core ML 编译命中、混合会话复用、托管缓存清理与卸载。
- `tests/native/benchmark_refactor.py`：保存旧版 dylib，分别在新进程测试 GPU/GPU+ANE；记录首次请求、热中位数、逐阶段时间和 PNG SHA256。
- `tests/integration/EndToEndBenchmark.swift`：相同 Swift 可执行文件分别加载旧/新 dylib，通过实际 `NativeJobStore` 测试事件、PNG 输出、历史持久化完整链路；不包括窗口绘制。

性能比较固定同一台机器、模型、分区、512×512、4 步、提示词、seed=42、精度和驻留模式；每组 1 次首次请求 + 5 次热运行。首次请求是新进程首次调用，不代表清空操作系统文件缓存或 Core ML 系统编译缓存。若热中位数回退超过 3%，必须进一步交替复测和分析，不能以统计噪声为由直接忽略。该阈值是本次验收标准，不是跨机器速度承诺。

实测结果见 [C++ 重构验证](validation/cpp-engine-refactor.json)。

### 负载变化与交替对照

单独整组测试会受到本机后台任务影响。本次额外使用 `benchmark_paired_app.py`：同一 Swift 测试程序分别加载旧/新 dylib，在两个常驻进程间按旧、新、新、旧（ABBA）逐次提交请求，每个版本预备 2 次、测量 8 次，不并发推理。两个进程会同时保有模型资源，因此该结果用于版本对照，不等同于单会话最低延迟，也不用于比较首次加载。

```sh
python3 tests/native/benchmark_paired_app.py \
  --baseline "$BASELINE_DIR/turbocider-paired-benchmark" \
  --candidate "$CANDIDATE_DIR/turbocider-paired-benchmark" \
  --model "$FLUX_MODEL_ROOT" \
  --manifest "$ANE_MANIFEST" \
  --output outputs/cpp-refactor/paired
```

两个可执行文件必须来自同一次 Swift benchmark 构建，分别放在旧/新打包 dylib 旁边；原始模型和分区必须相同。使用新输出目录以免历史文件影响测试。

首次 Core ML 加载另行记录。新客户端上下文的首次加载可能触发系统设备准备，即使 `.mlmodelc` 文件已经存在；本次观察到新旧版本均出现约 9.29 秒分区加载。后续加载约 0.2 秒。该证据支持缓存准备的解释，未声称直接观测到系统内部编译，也没有删除系统缓存来制造冷启动。

## 本机验收结果

M4 Pro 48 GiB、MLX 0.32.0、512×512、4 步、固定提示词与 seed=42。最终 ABBA 对照每个版本各 8 次有效热运行，经过真实 App `NativeJobStore`（不含窗口绘制）：

| 模式 | 重构前中位数 | 重构后中位数 | 变化 |
| --- | ---: | ---: | ---: |
| GPU | 5.0050 s | 4.9820 s | -0.46% |
| GPU + ANE | 3.0712 s | 3.0481 s | -0.75% |

变化幅度很小，结论是未发现性能退化，不宣称显著提速。记录包含初始独立运行、后台负载变化期间的整组复测及最终交替对照，未删除慢样本。所有新旧对照输出 PNG 哈希一致；19 个 C ABI 导出保持不变。

17 项仓库/请求/边界测试、App 行为测试和真实权重的 6 组回归通过。分阶段模式中，两种图像输入操作分别在 GPU 与 GPU+ANE 上和 resident 输出一致，返回时 MLX 活跃内存为 851,968 或 917,504 字节；这不代表整个进程占用不到 1 MiB。Transformer、文本编码器、VAE 编解码的数学实现除格式与注释外保持一致。

已重新构建并验证本地签名 App。H3/LTX 已有实际 native Session 与 GPU streaming 执行路径；仍未宣称所有输入、音频、LoRA 或 GPU+ANE 组合均通过正式质量/性能验收。
