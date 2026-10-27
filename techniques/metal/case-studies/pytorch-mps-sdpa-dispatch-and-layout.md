# PyTorch MPS SDPA：真实分流、view布局与mask类型的graph cache失败

<!-- knowledge
id: technique-metal-case-studies-pytorch-mps-sdpa-dispatch-and-layout
type: case-study
stage: prefill, decode, dit
operator: attention, layout, scheduling
backend: metal, mpsgraph, cpu-reference
device: cpu, gpu
symptom: wrong-layout, stale-cache, precision-loss, launch-bound, measurement
semantics: numerically-equivalent, contract-only
representation: fp16, bf16, fp32
boundary: cpu-gpu, producer-consumer
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [真实Qwen与profiler](pytorch-mps-profiler-and-real-attention.md) · [固定来源](../sources.md#pytorch-mps-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十五轮。实际调用已有Torch **2.14.1**的原始`F.scaled_dot_product_attention`，不是原生MPS Matrix、手写替代attention或Torch CPU模拟。四个独立合成进程的**4324个完整输出**通过预置FP64门槛；另将一个真实native abort缩减为两种mask dtype交替时的graph cache身份冲突。通过的子集不掩盖原生失败，也不构成所有MPS模型资格。

## 1. math入口并不意味着普通GPU矩阵乘法

读本/安装包声明的Git身份为`5c4886908584029761b579af026dcfb627c84070`。重点串起[前端attention.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/attention.cpp)、[MPS Attention.mm](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/Attention.mm)和[Decode/Prefill/MPP shader](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels)。inference/no-grad时，`SDPBackend::math`可以调用专用`_scaled_dot_product_attention_math_for_mps`，其内部再选以下路线。

| 源码条件，非性能保证 | 选定实现 | 数据流与边界 |
| --- | --- | --- |
|Q长度≤8且≤KV长度，Dq=Dv∈{64,96,128,256}，mask兼容|vector或两阶段vector|GQA通过`head/gqa_factor`读取原KV；KV长度达到1024时走两阶段partial/reducer，不能只算第一pass|
|Q长度>8、FP16/BF16、Dq=Dv∈{64,96,128,256}、MPP可用且mask兼容|MPP prefill|`matmul2d` cooperative tensors；64×32 query/key tile，4个SIMD group|
|Q长度>8、dtype/实例化D/同Dq-Dv/mask兼容|普通prefill|例如D80，或FP32；按D选择16/32行query tile和8/16/32行KV tile|
|以上均不满足，或禁prefill且不是短Q vector|`sdpa_general_mps`|MPSGraph图、GQA显式repeat KV、QK/softmax/PV以及完整attention张量输出|

这是固定源码的选择器模型；训练、nested/特殊rank、dropout或其他版本须重新读。`PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1`是源码中的诊断开关，**不会禁用短Q vector**，不是“强制所有SDPA都变成MPSGraph”。本轮dropout与非整除head比例各四次明确拒绝，之后四个零输入消费者正常完成；不是device fault或SIGABRT后的同进程恢复。

### MPP存在于GPU路径，不是M5或ANE的同义词

`has_mpp()`在该读本检查macOS26.2+；shader用Metal4条件编译。`MppAttention.h`明确区分Apple10与较早GPU的cooperative-tensor lane/slot映射：a10的连续quad，与非a10分支的split-pair不能互换。`a9`只是该实现的variant标签，不应据此推断某个SKU性能。

MPP核以16×32×16的simdgroup `matmul2d_descriptor`做QK/PV；online softmax在tile之间维护max/sum和未归一化O，尾块安全加载、GQA索引与全空行处理都在实现中。普通prefill的scale作用于浮点score fragment，而非先把Q及scale舍入为输入dtype；`fast::exp2`与编译器fast-math环境变量也不是同一层语义。源码浮点对象类型不等于物理ISA/累加器证明。

本机为M4 Max、macOS26.6.2/25G83，SDK26.5、clang21。它不是M5实验；没有NAX/ANE、硬件矩阵指令或带宽counter资格。下一页说明为何capture中的library名称清单也不能直接证明MPP分支实际执行。

## 2. 不把contiguous、origin和末维stride混在一起

固定`Attention.mm`对custom kernel的硬要求是`stride(-1)==1`；不要求整个Q/K/V都contiguous。普通batch/head/sequence stride分别传入kernel，`OperationUtils.h::mtl_setBuffer`绑定实际`storage_offset()*element_size()`。这与上一轮[Candle非零origin丢失](candle-packed-views-and-copy.md)是不同实现，不能互相套用。

每个合成point有三份**逻辑值完全相同**的resident bank，均实际GPU读回核对：

- head-major owning contiguous，offset0、末维stride1。
- token-major owner中先留两行，再narrow/transposed成BHTD view；非零offset、末维stride1。
- owner末维留间隔，取`[...,1::2]`；offset1、末维stride2。此时原实现需要准备contiguous输入，成本保留在forward窗内。

计时还包含SDPA结果转成连续`[B,L,H*Dv]`，不是只报理想kernel。bank最初构造/上传、验证读回、hash/FP64在窗外；因此“token-view没必要整张重排”不等于模型producer改布局免费。real bank的构造见下一页。

两个seed×default/禁prefill四进程，三个dtype、12类geometry/mask、三布局；每case每臂first、warmup和8个hot。合计4320输出，加4个错误后恢复输出为4324；hot为3456个，全部原始时间和完整输出SHA保留。Q1/K1023与1024、Q8与9、D33/64/80、不同Dv、GQA4:2和非整尾均有控制。

独立参考从**实际输入dtype舍入后的全部值**计算FP64 QK、稳定softmax、PV和相同flatten，不调用Torch CPU SDPA。每个输出全量读回/hash；确定性重复内容去重存储，仍逐次检查完整bytes，不把抽样行验证写成全输出。预置`atol,rtol,relative-L2上限`：FP32=`2e-5,2e-4,.001`，FP16=`.002,.003,.01`，BF16=`.02,.01,.01`，不因后续结果更改。

## 3. 三种“能算完但语义错了”的负控

非方`is_causal=True`采用**左上对齐**：Q长度1/KV长度5时只允许key0，不等于位置4的cached decode。cache-aware chunk应按真实位置构造显式mask，不能用方阵测试替代。Q9/K35的upper-left、显式lower-right各自按自己的数学验证；把upper-left结果错当lower-right，12个dtype/进程控制全部拒绝。

bool的False和加性`-inf`表示真正不参与softmax。720个正式输出的全空首行均为零。全行加有限`-100`只是共同平移logit，不会消除所有keys；其输出可能完全有限、且对于该加性bias数学是正确的，但不符合“空行应零”的意图。12个把有限sentinel当空mask的错误参考均拒绝。这里**24个负控拒绝不是24个引擎数值失败**。

## 4. mask dtype不在graph key：两个顺序均触发原生断言

最初扩展cohort在禁prefill后，同一Q/K/V shape先用FP16 mask、后用FP32 mask，native MPSGraph断言“expected f16 but received f32”，进程exit`-6`；没有Python异常可让普通`try/except`恢复。原日志/部分文件保留，不给缺少完整journal的数据回填数值资格。

缩减使用未禁prefill的正常入口：Q=`[1,2,9,33]`，K/V=`[1,1,17,33]`，FP16/GQA。D33自然走graph；mask同shape、同值，只换dtype。六个有界独立进程记录每次调用前journal：

| mask顺序 | 完整输出 | 第二次/终态 |
| --- | --- | --- |
|仅FP16、仅FP32|各1，均通过FP64|正常退出|
|FP16→FP16、FP32→FP32|各2，均通过FP64|正常退出|
|FP16→FP32、FP32→FP16|各第1次通过FP64|第二次均类型断言/exit−6|

总计8个已完成诊断输出，与4324个正式合成输出分开。两次abort不是重试性能样本，也不授予错误后继续复用GPU buffer的资格。

源码解释与最小现象相符：`sdpa_general_mps`的key包含Q/K/V、causal、mask是否存在、scale，却不包含mask dtype；创建graph时用当时mask dtype建立placeholder，后来直接命中旧graph。反向顺序将expected/received类型对调，单独或同dtype重复正常，支持这是**cache身份遗漏**，而不是某个mask dtype单独不支持。未改上游、未清全局cache、未声称所有/最新版PyTorch都有该问题或已修复。

技术迁移点不是“一律copy输入”，而是把影响typed graph ABI的mask dtype等信息放进cache身份，并在入图前验证。这是设计建议，不是本轮提交的生产修复。

## 5. 来源、重放和重开条件

labs/pytorch_mps_contracts.py提供无依赖mask/selector/cache-key合同，case `pytorch-mps-mask-contracts`；测试只审计CPU合同/归档，不导入Torch或启动GPU。可选客户端、失败缩减需显式已有Torch环境、固定source和全新ignored目录，使用监督器。

正式数据只计`outputs/metal-pytorch-final3-20261017-*`；先前ImportError、两次busy拒绝、扩展pilot和原native abort都在证据中分账。正式合成矩阵不跨进程交替graph mask dtype；这条失败由独立缩减覆盖，不能因矩阵全绿而删除。所有source/库/wheel、OS/SDK/SoC、mask/dtype/shape/stride、graph cache或消费者变化均重开；完整模型、LoRA、动态batch、训练、OOM/取消/并发、峰值内存和物理profiling仍待研究。
