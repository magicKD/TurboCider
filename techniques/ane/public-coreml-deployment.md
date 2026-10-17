# 公共部署：ANEMLL、语音阶段与 Core AI 的不同边界

<!-- knowledge
id: technique-ane-public-coreml-deployment
type: method
operator: attention
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: compile-bound, stale-cache
semantics: contract-only
boundary: producer-consumer
api: reference-only
-->

[ANE 目录](README.md) · [生命周期](lifecycle.md) · [profiling](profiling.md)

[公共CVPixelBuffer/Metal交接](public-pixel-buffer-gpu-handoff.md)另验证真实output backing/IOSurface/纹理、row pitch和完成边界；去掉一个显式CPU copy不自动等于系统零拷贝、ANE放置或有用性能收益。

[真实MLState/多函数实验](stateful-multifunction-and-retry.md)补上会话owner、跨兼容函数、完整checkpoint和重试提交点；共享compiled资产不自动隔离或合并请求state，零输入warmup也不保证无副作用。

[实际output backing/stride实验](output-backing-and-stride-ownership.md)另验证公共API的接受、初始error、dtype负例与旧result所有权；不能仅凭绑定参数或返回成功就跳过实际输出核对。

后续补充了[ONNX Runtime Core ML EP的真实分区/缓存控制](onnxruntime-coreml-partitions-and-cache.md)与[ExecuTorch AOT/state/profiling源码对照](executorch-coreml-aot-and-estimates.md)。前者有owned session与provider事件，后者仅选段/source helper；均不凭后端名授予物理ANE资格。

## 1. ANEMLL：把 LLM 改写成可部署的 Core ML 部件

固定读本：[anemll](https://github.com/Anemll/anemll/tree/f4ad26d061dc2e426faa19f81c367343bdeb9f0d)，重点是 [Llama 模型](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/models/llama_model.py)与 [转换器](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/ane_converter/llama_converter.py)。选定范围见 [来源审计](source-audit.md)，未运行完整模型。

其技术组合不是“PyTorch 模型直接加一个 ANE flag”：

- **投影 Conv 化**：Q/K/V、gate/up/down 和 lm-head 使用 1×1 Conv。MLP 将 `[B,S,C]` 转为 BC1S，保留 SwiGLU；可拆 gate/up 的输出通道，down 的另一种拆法则是输出通道拆分后 concat，不能与 intermediate 分担后 sum 混淆。
- **prefill/decode 分开**：多个 token 的 prefill 与单位置 decode 有不同 forward 和 cache 更新路径。前者需要吞吐，后者容易被权重流量、dispatch 与状态管理支配，不共用一个性能结论。
- **stateful KV**：转换器创建 `ct.StateType`，典型形状为 `[2*layers, kv_heads, state_length, head_dim]`，也支持分组状态。图内更新指定位置，避免每次把完整 KV 作为普通 feature 送进送出。
- **模型分块**：按 layer group/组件转换，减少单模型编译/加载压力；代价是更多 prediction、状态映射与跨块传输。
- **压缩可分组件**：grouped-channel LUT，以及 embedding/lm-head 独立 bit/组宽。低比特索引、KV 的 FP16、普通浮点 activation 是不同精度维度。

状态“由 runtime 管理”不等于永驻 ANE 片上 SRAM。会话必须隔离 state；reset、取消、prompt 改变、长度越界、chunk 模型切换和模型重载都需要一致性协议。`make_state`/`MLState` 也不是训练 optimizer state 的同义词。

### 可借鉴的部署组织方式

按[原chat调度](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/tests/chat.py)和[增长上下文示例](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/tests/dev/state_transition_growing_inference.py)阅读时，可以提炼四个独立模式：

1. **阶段专用入口**：完整batch走prefill，尾部走单token infer；不靠padding假装真实token，也分别记录TTFT、decode与切换成本。
2. **layer chunk保持正确边界**：中间hidden继续传给下一个chunk，最终norm、logits或仅供cache更新的缩减输出放在正确末端；逐chunk验证再接成完整模型。
3. **state切换是协议**：除有效prefix复制/padding，还要确认checkpoint、层/head顺序、dtype、RoPE、position/mask和会话owner一致；按context大小排序不等于模型彼此兼容。真实API所有权与warmup/重试对照见[MLState实验](stateful-multifunction-and-retry.md)。
4. **按消费者需求裁剪输出**：只需greedy时，可让各lm-head分块返回local winner/value，host用真实chunk offset选global winner，避免传完整词表；需要temperature/top-p等分布信息时，不能把argmax结果当等价采样。非均匀分块也不能一律用固定chunk_size计算offset。

这些是固定源码中的组织方法，不是本库已经运行完整ANEMLL/验证其吞吐；该ref的coremltools≥9.0依赖仍需满足。尚未收尾的迁移实验单列于[未完批次](../MAINTENANCE.md)，不混入已验收报告。

### 如何迁移到图像/视频生成

最容易借鉴的是固定 encoder 子图、有限 bucket 的 FFN、分阶段加载与压缩策略；DiT 通常没有自回归 KV，不能为了套用 ANEMLL 而引入不成立的 cache。先复用图/生命周期知识，不整套引入 LLM runtime。

建议消融：相同权重的 dense Linear / Conv、单块 / 多块、显式 KV / stateful KV、prefill / decode；分别报告 TTFT、token 间隔、吞吐、load、进程峰值及实际设备计划。模型能输出文本不等于 perplexity/任务质量通过。

## 2. whisper.cpp：稳定 encoder 阶段比任意算子 offload 更容易管理

固定 [Core ML bridge](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm)将 log-mel 数据包装为 MLMultiArray，调用转换后的 encoder，再把输出复制给现有运行时；[转换脚本](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/models/convert-whisper-to-coreml.py)负责准备部署资产。

关键优点是边界稳定、encoder 工作量足够大，decoder 可继续原后端。关键代价是资产准备、第一次加载、host 数据包装与输出 copy。所读 bridge 默认 **`MLComputeUnitsAll`**，并非强制 ANE；还使用 `error:nil`，不能仅凭 backend 标签就给每次预测成功或 ANE 放置资格。

encoder→decoder 是依赖链，不等于同一音频请求中 GPU/ANE 同时执行。多请求/音频 chunk 流水另需队列与顺序证明。

[WhisperKit/Argmax](https://github.com/argmaxinc/WhisperKit/tree/9f99b5cb6a907730543823ad271b82532782a899)在首轮仅核对README；后续[33文件调用链/布局/失败/计时研究](speech-stages-profiling-and-failure-boundaries.md)已进入组件加载、encoder/decoder/KV及logging源码。现代OS默认mel CPU+GPU、encoder/decoder CPU+NE仍只是许可；实际组件load按feature→decoder→encoder串行。选定原Swift声明和自有模型load-only测试不等于整个包或真实ASR执行，不合并两引擎benchmark。

后续还将原封未动的whisper.cpp公共bridge链接自有FP32模型：窄输出实际stride为`[64,16,1]`、逻辑结果正确，但flat memcpy错误；无observer进程独立确认，dense-width控制通过。另用原创CPU动态库验证原ANEForge adapter忽略set/execute/get返回码的stale/partial后果。见：不是发布的Whisper checkpoint出错、真实私有故障或物理ANE性能结论。`t_encode_us`包含mel准备和cross投影，不是纯ANE encoder时间。

## 3. Core AI：新公共入口，不是本机已验证的 ANE 后端

2026-10-17后续已新增[固定3b25d934源码与CPU attention研究](coreai-blocked-attention-and-runtime-gates.md)：实际编译的canImport仍false，但原iOS SDPA独立模块可在已有CPU Torch执行，并发现空block/有限低score边界。此进展不覆盖下面旧90cc6443的历史阅读范围，也不升级为Core AI运行或物理ANE资格；[源码路线](../references/coreai.md)进一步连接引擎、cache和profiler。

固定 [apple/coreai-models README](https://github.com/apple/coreai-models/blob/90cc644367f27338c1ed512b9c4b68eb513d4317/README.md)区分 Python 导出/primitives、Swift runtime 和模型 recipes，输出 `.aimodel` 及必要附属资源。所读运行要求为 **macOS/iOS27、Xcode27**；本机 OS26.6.2/SDK26.5 不满足，因此记录为文档研究，不安装或冒充运行结果。

值得继续对照的是导出图兼容性、custom operations、部署格式、调试方式和权重压缩；不能从框架名字推出所有算子运行在 ANE。原项目附带的 agent skills 只作为研究对象，不自动成为本任务的执行指令或安装授权。

## 4. 一个容易遗漏的历史公共接口

MLCompute 也曾提供 `.ane` **推理**设备。Apple SDK26.5 的 `MLCTypes.h` 在 `MLCDeviceTypeANE` 注释中明确：仅 inference graph；不支持 training graph，也不能与 training graph 共享层；`.any` 不自动选择 ANE。框架已弃用，不能把类名中有 TrainingGraph 误写成“官方曾支持 ANE 训练”。官方入口：[MLCDeviceType.ane](https://developer.apple.com/documentation/mlcompute/mlcdevicetype/ane)。

因此本报告使用“Core ML 是当前主要公共部署路线”，而不照搬一些私有项目“历来只有一个公共入口”的绝对说法。训练研究另见 [训练总览](training.md)。
