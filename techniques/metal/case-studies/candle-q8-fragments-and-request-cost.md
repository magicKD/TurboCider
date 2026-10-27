# Candle Q8真实请求：half权重片段、条件数值参考与完整成本边界

<!-- knowledge
id: technique-metal-case-studies-candle-q8-fragments-and-request-cost
type: case-study
stage: prefill, decode, ffn, pipeline
operator: gemm, gemv, attention, quantization, sampling, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, launch-bound, compile-bound, memory-bound, measurement
semantics: numerically-equivalent, approximate, contract-only
representation: q8, fp16, fp32, gguf
boundary: weight-runtime, producer-consumer, cpu-gpu, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [完整预训练模型/tokenizer](candle-pretrained-qwen-and-tokenizer.md) · [来源](../sources.md#candle-pretrained-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十六轮精度/成本部分。真实Qwen2.5完整图说明：**8-bit权重存储、FP32 API输出、生成相同词、完整logits通过**是四种不同结论。packed warm请求虽然约38ms，resident dense约65ms，但packed没有通过原定完整logits门槛，所以不发布它的合格加速比。没有修改默认或放宽误差线。

## 1. Q8文件不会自动得到全FP32中间计算

固定[quantized Metal实现](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/quantized/metal.rs)按输入倒数第二维是否为1分流，重要读本是[quantized.metal](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-metal-kernels/src/metal_src/quantized.metal)：

| 源码路线 | 相关类型/操作 | 本轮含义 |
| --- | --- | --- |
|M1 Q8 GEMV|int8 lanes乘float activation，float block sum乘half scale，float reduction/output|不先将每个decoded weight写入half矩阵片段|
|M>1 Q8 GEMM|实例化`T=half`、half4x4、simdgroup_half8x8；activation/accumulator对象为float矩阵|decoded Q8权重会经过half片段，即使函数后缀是`_f32`|
|公开`dequantize`/resident dense|CPU block解码后新建FP32 Metal storage，再走普通Tensor matmul|是准备路径，不是热forward里免费融合的GPU materializer|

这是源码类型与数据流事实，不是物理ISA/累加器指令证据，更不能由Q8存储宣称INT8 tensor-MAC或M5 NAX。该模型类本身还显式做QK、softmax、PV与KV cat/repeat；本轮不是另一个fused Flash Attention模型实现。

## 2. 次级参考解释误差，但不改写原目标

原参考完整解码Q8到FP32，独立FP64-stage计算整个模型。packed正常输出relative-L2最高约0.00187，低于.005的整体L2门槛，但152份输出仍有逐元素误差超出`.005+.005*abs(ref)`。**只看整体L2或top1会漏掉失败。** resident FP32 dense的208份正常输出全部通过，最大绝对误差约1.12e-4，支持原参考并非误读了权重或tokenizer。

根据源码建立第二个、明确条件化的CPU模型：仅把`blk.*`里的量化矩阵权重按decoded-FP32→FP16→FP32舍入；norm/bias、embedding和最后M1 LM head不变，activation仍按浮点阶段边界处理。它不是重新量化GGUF、不是从GPU输出拟合参数、也不是把整个模型改成FP16。

只比较**fresh且输入行数>1**的实际调用：原prefill、fresh全prefix、chunk的首段和错误位置前的正常primer。此时各层投影确实全走M>1路线。7个独立prefix、64份完整GPU输出全部通过该次级参考，最大绝对误差约1.16e-4；同64份输出对原目标均失败。

这个反事实与真实resident dense消融一起，有力支持half权重片段是选定误差的来源。没有逐层buffer/counter或ISA采样，不能把它扩大成所有误差的唯一物理原因。特别是cached decode混用了先前M>1 prefill和当前M1投影，不能将一个“全prefix统一half权重”的CPU模型冒充这种混合历史；本轮不为这些调用补造次级通过结论。

因此cached/chunked32组均过线、cached/fresh仅24组过线是可理解的精度路径差异，但仍保留8个cache/fresh预算失败。它不是上一轮[packed非零origin丢失](candle-packed-views-and-copy.md)的重命名，也不是自动获得原encoded-weight gate资格。

## 3. 完整请求计时实际包含什么

正式客户端保持模型/tokenizer已加载，Python开始计时后依次完成：原chat-template render/encode→JSON/stdio IPC→Rust clear-cache、token上传、原始model forward+sync、全词表D2H、有限值检查和CPU greedy argmax→返回选择ID→官方tokenizer decode成文本。

Rust在请求期间保留每步已有的完整CPU logits Vec，供计时结束后flush；没有把结果文件写入、SHA计算或日志JSON序列化塞进模型计时。所有256个实际生成步骤的完整向量都保存，核对sampled ID等于该向量argmax，不能用另跑一次的验证结果替代正式消费者。

元数据构造、全词表有限值检查、显式sync和IPC仍属于这个原创诊断客户端的成本；它不是无observer的上游example最佳实现。timed Rust区间与Python完整encode-to-text区间分别保存，后者不能仅用各GPU kernel时间相加推算。

同一进程有两个prompt各两轮，replicate17/29反转顺序。只把round1作为warm汇总，round0、首次model-ready请求及所有时间原样保留。每配置的warm样本为两进程×两prompt=4；flags固定于各进程创建模型前，不伪装同trial ABBA或独立随机seed。

| 路线 / compute-request limit | warm encode-to-text中位数ms | 原参考合格中位数 |
| --- | --- | --- |
|packed /50|38.414|`null`，数值门槛拒绝|
|packed /1|85.585|`null`，数值门槛拒绝|
|resident dense /50|65.357|65.357|
|resident dense /1|100.211|100.211|

dense两配置输出通过，跨进程中位数之比约1.53×；该小矩阵支持“提交/等待策略可能改变完整请求成本”，不证明50全局最优或全部收益来自某一物理launch成本。54组按mode/prefix/chunk/position区分的输出，在limit与replicate变化后完整bytes相同。

packed较低诊断wall不能自动成为相对dense的合格性能结论；64个条件参考通过也不能覆盖另外的主目标。若实际应用接受此精度路径，需要**事先建立任务质量预算**，再测原模型质量/采样分布和完整请求，而不是看完数据后只保留top1门禁。

## 4. load、first-use、logical bytes和峰值分别记账

所有进程先执行额外的291-tensor公开CPU解码/hash，用于独立decoder确认，约1.27–1.28s；这不是原引擎必要startup成本。之后原模型load+sync约packed180–183ms、dense456–465ms。模型文件此前已完整读取/hash，不能称为cold-storage loading。

最早packed pilot的第一请求约4.66s，后续约40ms；正式进程第一请求约53–136ms。没有清driver/PSO/system cache，也没有compile事件分解；first-use准备可以解释这种方向，但不能将4.66s全归给某个未测编译环节或把后续进程当真正冷启动。

文件大小约676MB，完整权重FP32解码约2.52GB，embedding/head还有重复存储。原loader将embedding公开dequantize为FP32，dense flag又会准备resident矩阵；这些是应计入采用决策的逻辑表示成本。实际allocator缓存、临时矩阵、KV、进程RSS与GPU峰值未独立采集，不用字节之和或steady延迟伪造峰值/带宽/能耗。

## 5. 资格、重放和下一步

全部正式8进程/448个完整输出，计数与[预训练报告](candle-pretrained-qwen-and-tokenizer.md)一致；32份错误position负控本身也保持相同top1，说明只看文本尤其不可靠。两prompt/8-token cap不是困惑度、事实性、长文本、原BF16量化损失、LoRA、serving吞吐或TurboCider媒体质量的验收。

CPU合同：labs/gguf_pretrained_contracts.py，case `pretrained-request-cost-contracts`；归档器保留主/次参考与null资格；测试不启动设备。设备selector `candle:pretrained-request`对应显式请求驱动，使用监督器/新目录、原始模型SHA和固定build身份。

来源/二进制/Rust/lock、GGUF/dtype、M/stride/layout、KV/位置、tokenizer或生成策略、硬件/OS/SDK、request边界变化均重开。后续需要更多真实模型/质量预算、严格同预算的新权重或精度候选、真实内存/并发/取消生命周期及可信物理profiling；本页不授予全局后端默认。
