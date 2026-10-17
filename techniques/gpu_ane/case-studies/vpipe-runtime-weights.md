# vpipe：runtime-weight 图、row split 与隐藏 staging

<!-- knowledge
id: technique-gpu-ane-case-studies-vpipe-runtime-weights
type: case-study
stage: ffn
operator: swiglu, gemm
backend: cpu-reference, mlx, coreml
device: cpu, gpu, ane
symptom: straggler, precision-loss
semantics: numerically-equivalent
representation: fp16, fp32
boundary: gpu-ane, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [切分代数](../partition-and-scheduling.md) · [来源](../../MAINTENANCE.md)

续研已进一步读到实际控制器和模型调用点，见 [双层迟滞、精度和失败消费](vpipe-controller-and-failure.md)；以下第一轮的接口/方案说明不替代后续失败路径检查。

第五轮补充[headroom穿过bias/LoRA的完整函数](vpipe-bias-lora-and-headroom.md)，并追到MiniMax-H3 VAE的bias-enabled SwiGLU与gated residual调用；是固定源码/CPU条件见证，不是上游实机故障复现。

源码读本：[`tgo-app-dev/vpipe@c2989a76841888e4d8d208fad4f476c487710962`](https://github.com/tgo-app-dev/vpipe/tree/c2989a76841888e4d8d208fad4f476c487710962)，不是“最新版本”声明。重点阅读 `generative-models/shared/ane-ffn.h` 1–220、`ane-ffn.cc` 1–460；继续定位 begin/finish/plan、`ane-module.*`、`apple-silicon/coreml/ane-worker.*` 与 `ane-emitter.h`。本轮没有运行上游程序。

## 技术方案

`AneFeedForward` 按独立 token 行分 SwiGLU/GELU FFN。ANE 计算尾部整 chunk，GPU 计算头部和余数。这里两设备均需要完整 FFN 权重，区别于按 intermediate F 拆分权重并相加 partial。

每个模型只有一份固定形状的 runtime-weight module，gate/up/down 是 IOSurface-bound inputs，逐层写入 slots；避免为每层常量权重都建立和常驻一个编译程序。交换代价是每层 staging 和 Core ML 可能的内部副本。常量图与 runtime-weight 图是空间/准备成本/执行速度的 tradeoff，不能预设后者更快。

源码中的默认 `chunk=2048`、graph `tile=1024` 和不同常量图最佳 tile，是作者工作负载的经验参数，不是 ANE 规格。host chunk 与 graph 内 K/N tile 是不同参数：前者增加 prediction 次数，后者改变单图 lowering/内存行为。

## 调用时序

```text
GPU：attention 已提交 ────────────── drain ─ GPU head-row FFN ─── join/consumer
CPU/worker：stage 本层权重 ─ join_stage ─ begin → ANE tail chunks → finish
```

这是预期依赖与重叠窗口；是否物理并行仍需设备 trace。关键不是仅把 `prediction()` 放线程，而是 attention 必须已经 commit，stage 可用且不读未完成 buffer，begin 之前输入已经 ready。

如果上一些 block 的 GPU command 仍延迟提交，split 点的 drain 会把之前积压工作也算进本层。源码 `needs_barrier()` 专门处理 unsplit→measured split/probe 的这种测量污染。只看某一段 wall time 会给自适应 balancer 喂错数据。

## 权重准备为什么值得单独优化

`AneFfnSource` 保留 source `[out,in]`，允许 `stride/offset` 表示 `[gate;up]` 或交错 `[g0,u0,…]`；multi-part source 可写到同一 slot 的不同 row。避免为了满足假想 ANE 布局做逐元素转置。

源码分三条解码路径：BF16→FP16 用 65536 项 LUT；affine Q4 每 group 建 16 项 FP16 LUT 再解 packed words；Q8 按实际 scale/code/bias 计算。带 LoRA 或 headroom scale 时转入 FP32 行累加再收窄。其代码注释的 staging 毫秒值只解释作者为何改变实现，本专题不复用成当前机器性能。

LoRA 的 `A/B` layout 不一定跟 base gate/up interleave 相同，故 source adapter 有独立 `b_stride/b_offset`；banded adapter 又有 part/group 语义。直接复制 base 的 row 索引可能输出 finite 但错误。

FP16 不是 BF16 的同宽 reinterpret。SwiGLU 乘积有溢出风险；GELU 的缩放也不能机械跨过非线性。正确方案需要数值见证和恢复比例，不能只依据“FFN 内部封闭”便宣称无损。

## 性能与内存控制

源码账本把 weight slots、Core ML staging/activation 与一个 chunk 的 host I/O 分开；系数是特定模型测量拟合，不是 API 保证。runtime module 数固定不意味着总内存只有一份权重；需检查 slots、framework copy、GPU 原权重、scratch 同时存活。

`kSplit/kProbe/kGpu` 比较分担和整 GPU。只优化 rows/ms 会漏掉 mid-block commit、stage job 等固定成本，因此更快 GPU 或更短序列可能让整个 split 失利。pinned share 则会强制 split，是实验控制，不应视为自动盈利。

析构先 `worker().join()` 再释放 slots/host buffer，提前退出也要 join；每层来源 buffer 必须活到 staging 完成。多请求共享一个输出/weight slot 时必须串行或双 bank+generation 管理，不能只因为输入不可变就并发复用输出。

## 公共与非公共边界

执行入口采用 Core ML，但 emitter 自行生成未文档化 `.mlmodelc` 格式。读 [`ane-emitter.h`](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.h) 的兼容性警告和 self-test：公共预测 API 不会让自建内部格式自动成为稳定公共合同。应保留 emitter 验证失败→GPU fallback。

## 建议实验

相同 checkpoint/LoRA/rows 下做 full GPU、serial row split、concurrent row split；再比较常量/runtime weights、chunk/tile、原转置与 source-layout staging。分别记录准备、load/首调、热调用、完整 FFN、整请求和峰值。

否决：只隐藏了线程等待而未提交 GPU、量化 source 不同、GPU 强基线改变后不再盈利、尾部 padding/host prediction 太多、stage 争带宽、常量/cache 身份过期。可参考本库 [已有混合 FFN 证据](../../MAINTENANCE.md)，但它是另一个实现和切分方案，不是 vpipe 的实机复现。
