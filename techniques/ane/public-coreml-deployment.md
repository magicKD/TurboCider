# 公共部署：ANEMLL、语音阶段与 Core AI 的不同边界

[ANE 目录](README.md) · [生命周期](lifecycle.md) · [profiling](profiling.md)

## 1. ANEMLL：把 LLM 改写成可部署的 Core ML 部件

固定读本：[anemll](https://github.com/Anemll/anemll/tree/f4ad26d061dc2e426faa19f81c367343bdeb9f0d)，重点是 [Llama 模型](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/models/llama_model.py)与 [转换器](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/ane_converter/llama_converter.py)。选定范围见 [来源审计](source-audit.md)，未运行完整模型。

其技术组合不是“PyTorch 模型直接加一个 ANE flag”：

- **投影 Conv 化**：Q/K/V、gate/up/down 和 lm-head 使用 1×1 Conv。MLP 将 `[B,S,C]` 转为 BC1S，保留 SwiGLU；可拆 gate/up 的输出通道，down 的另一种拆法则是输出通道拆分后 concat，不能与 intermediate 分担后 sum 混淆。
- **prefill/decode 分开**：多个 token 的 prefill 与单位置 decode 有不同 forward 和 cache 更新路径。前者需要吞吐，后者容易被权重流量、dispatch 与状态管理支配，不共用一个性能结论。
- **stateful KV**：转换器创建 `ct.StateType`，典型形状为 `[2*layers, kv_heads, state_length, head_dim]`，也支持分组状态。图内更新指定位置，避免每次把完整 KV 作为普通 feature 送进送出。
- **模型分块**：按 layer group/组件转换，减少单模型编译/加载压力；代价是更多 prediction、状态映射与跨块传输。
- **压缩可分组件**：grouped-channel LUT，以及 embedding/lm-head 独立 bit/组宽。低比特索引、KV 的 FP16、普通浮点 activation 是不同精度维度。

状态“由 runtime 管理”不等于永驻 ANE 片上 SRAM。会话必须隔离 state；reset、取消、prompt 改变、长度越界、chunk 模型切换和模型重载都需要一致性协议。`make_state`/`MLState` 也不是训练 optimizer state 的同义词。

### 如何迁移到图像/视频生成

最容易借鉴的是固定 encoder 子图、有限 bucket 的 FFN、分阶段加载与压缩策略；DiT 通常没有自回归 KV，不能为了套用 ANEMLL 而引入不成立的 cache。先复用图/生命周期知识，不整套引入 LLM runtime。

建议消融：相同权重的 dense Linear / Conv、单块 / 多块、显式 KV / stateful KV、prefill / decode；分别报告 TTFT、token 间隔、吞吐、load、进程峰值及实际设备计划。模型能输出文本不等于 perplexity/任务质量通过。

## 2. whisper.cpp：稳定 encoder 阶段比任意算子 offload 更容易管理

固定 [Core ML bridge](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm)将 log-mel 数据包装为 MLMultiArray，调用转换后的 encoder，再把输出复制给现有运行时；[转换脚本](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/models/convert-whisper-to-coreml.py)负责准备部署资产。

关键优点是边界稳定、encoder 工作量足够大，decoder 可继续原后端。关键代价是资产准备、第一次加载、host 数据包装与输出 copy。所读 bridge 默认 **`MLComputeUnitsAll`**，并非强制 ANE；还使用 `error:nil`，不能仅凭 backend 标签就给每次预测成功或 ANE 放置资格。

encoder→decoder 是依赖链，不等于同一音频请求中 GPU/ANE 同时执行。多请求/音频 chunk 流水另需队列与顺序证明。

[WhisperKit/Argmax](https://github.com/argmaxinc/WhisperKit/tree/9f99b5cb6a907730543823ad271b82532782a899)是另一个 Core ML 音频部署生态入口，本轮只核对 README/项目定位，未审核其所有算子、decoder 配置或性能。不要把 whisper.cpp 的 bridge 分工套给它，或合并两者的 benchmark。

## 3. Core AI：新公共入口，不是本机已验证的 ANE 后端

固定 [apple/coreai-models README](https://github.com/apple/coreai-models/blob/90cc644367f27338c1ed512b9c4b68eb513d4317/README.md)区分 Python 导出/primitives、Swift runtime 和模型 recipes，输出 `.aimodel` 及必要附属资源。所读运行要求为 **macOS/iOS27、Xcode27**；本机 OS26.6.2/SDK26.5 不满足，因此记录为文档研究，不安装或冒充运行结果。

值得继续对照的是导出图兼容性、custom operations、部署格式、调试方式和权重压缩；不能从框架名字推出所有算子运行在 ANE。原项目附带的 agent skills 只作为研究对象，不自动成为本任务的执行指令或安装授权。

## 4. 一个容易遗漏的历史公共接口

MLCompute 也曾提供 `.ane` **推理**设备。Apple SDK26.5 的 `MLCTypes.h` 在 `MLCDeviceTypeANE` 注释中明确：仅 inference graph；不支持 training graph，也不能与 training graph 共享层；`.any` 不自动选择 ANE。框架已弃用，不能把类名中有 TrainingGraph 误写成“官方曾支持 ANE 训练”。官方入口：[MLCDeviceType.ane](https://developer.apple.com/documentation/mlcompute/mlcdevicetype/ane)。

因此本报告使用“Core ML 是当前主要公共部署路线”，而不照搬一些私有项目“历来只有一个公共入口”的绝对说法。训练研究另见 [训练总览](training.md)。

## 本地学习入口与复查

- [model_stages.py](../labs/model_stages.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [ane/compute_plan.py](../labs/ane/compute_plan.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py stage-llm`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
