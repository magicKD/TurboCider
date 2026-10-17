# Candle完整预训练Qwen：官方GGUF、tokenizer与151936维logits验证

<!-- knowledge
id: technique-metal-case-studies-candle-pretrained-qwen-and-tokenizer
type: case-study
stage: prefill, decode, ffn, pipeline
operator: gemm, gemv, attention, norm, rope, swiglu, quantization, sampling
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, stale-cache, wrong-layout, measurement, compile-bound
semantics: numerically-equivalent, approximate, contract-only
representation: q8, fp16, fp32, gguf
boundary: weight-runtime, cpu-gpu, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [Q8片段精度与请求成本](candle-q8-fragments-and-request-cost.md) · [来源](../sources.md#candle-pretrained-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十六轮。不再只有合成Qwen图或真实attention激活：实际运行**官方预训练Qwen2.5-0.5B-Instruct的完整24层模型**，使用固定Candle原始GGUF loader/`quantized_qwen2::ModelWeights`、官方tokenizer与原chat template；CPU采样、增量KV、完整词表输出和文本decode均进入有界请求客户端。

正式8进程/32个最多生成8 tokens的请求，共448份完整logits。416份正常调用中264通过、152超出不放宽的完整logits门槛；另外32个刻意错position的负控全部拒绝。**所有448份输出的argmax仍与正确数学参考相同**，因此“生成相同词”并不能代替完整验证。精度原因和不获性能资格的packed数据见下一页。

## 1. 固定真实资产，区分文件、模型与运行时dtype

模型是[官方Q8_0 GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q8_0.gguf)，revision=`9217f5db79a29953eb74d5343926648285ec7e67`，675710816 bytes，完整SHA256=`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`。本轮新增下载约676MB，不是从已有FP16 fixture或随机矩阵倒推“真实权重”。

[原始tokenizer/config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)固定revision=`7ae557604adf67be50417f59c2c2f167def9a775`。API返回的Git blob/LFS身份与下载内容逐一核对，模型/tokenizer资产在CPU归档前后再次校验。两仓库的Apache-2.0许可文件/README哈希保留，raw权重不随Wiki分发。

| 资产/表示 | 本轮实读事实 | 不能直接推导 |
| --- | --- | --- |
|GGUF v3|291 tensors：170个Q8_0、121个FP32|文件名Q8不表示每个参数都是8bit|
|模型几何|H896/F4864，24层，14 Q heads/2 KV heads，D64，context32768，RoPE base1e6|不覆盖其他Qwen、长context、sliding-window或LoRA|
|原HF config|声明BF16、tie-word-embeddings=true、sliding-window=false|没有下载原BF16 checkpoint或验收Q8相对BF16的任务质量|
|文件中的embedding/head|`token_embd.weight`和`output.weight`都存在，独立解码后完整SHA相同|语义tie不保证文件/运行时只保存一份|
|存储参数计数|包含重复head共630167424个元素；全部FP32展开为2520669696 bytes|不是模型独立参数量、进程RSS或实际GPU峰值|

Q8_0的32个值占34 bytes：2-byte little-endian FP16 scale与32个signed int8。它不是无scale的逐byte FP8，也不是GGUF整体平均恰好1 byte/parameter。此处独立reader只接受有界v2/v3、FP32/FP16/whole-row Q8_0，不宣称兼容全部GGUF、K-quants或恶意文件安全审计。

## 2. tokenizer的ID空间必须和GPU head对齐

官方tokenizer有151665个可识别ID，GGUF embedding/head为151936 rows，差271个padding rows。逐个检查全部tokenizer token→ID映射与GGUF词表一致，不能仅比较vocab size。全部实际生成ID也在原tokenizer集合内；完整logits验证仍包括所有151936维，没有裁掉padding维来美化误差。

使用已安装tokenizers0.23.3和Jinja2，直接加载官方`tokenizer.json`、render官方`tokenizer_config.json`中的完整chat template，`tools=None, add_generation_prompt=True`。不手写special-token编号或拼一个“看起来类似”的prompt。新客户端没有导入Torch/MLX模型、安装全局包或执行远程模型代码。

两个固定prompt是继续数列、解释叶子绿色；每个原模板prompt为39 tokens。客户端明确采用**greedy、max-new-tokens=8、尊重原EOS ID集合**，不应用官方generation_config推荐的sampling/temperature/repetition-penalty配方。本fixture均触及8-token cap，没有声称自然EOS或完整句子质量通过。

这是“原始Candle模型类＋原创Python/Rust请求客户端＋官方tokenizer”，不是未修改的上游example CLI。模型所有embedding、24层RMSNorm/QKV+bias、GQA、half-split RoPE、causal attention、SwiGLU、residual、final norm/head都由原始类执行，没有attention/FFN功能替身。

## 3. last chosen token不等于已经写进KV

一个8-step请求的状态是：

| forward | 本次消费 | 正确index_pos | forward后实际prefix长度 |
| --- | --- | --- | --- |
|prefill|39个prompt IDs|0|39|
|第一次decode|刚选出的第1个ID|39|40|
|第7次decode|刚选出的第7个ID|45|46|
|返回请求结果|已选出8个ID，第8个尚未forward|—|仍为46，不是47|

因此fresh/cache比较针对`prompt + choices[:-1]`，不是把最终选择但尚未消费的token也塞进参考。每请求再执行clear/fresh完整prefix、5-token prefill→剩余多token chunk→最后1 token，覆盖非零past的causal mask和GQA缓存。

32组cached/chunked比较全部通过，但cached/fresh只有24组通过同一门槛；8个失败均在packed模式，不能因文本相同就抹掉。packed的M>1和M1投影精度不同，详见[下一页](candle-q8-fragments-and-request-cost.md)。源码在index_pos=0时也会重建KV；本轮显式clear，不把正常新请求通过解释成“只有clear才能恢复”。

负控先用正确N−1个tokens建cache，再把最后一个token的RoPE位置错报为N而不是N−1。原API能完成该有界调用，但32份输出全部不符合正确prefix参考；**它们的top1也没有改变**。这是调用方位置合同的误用负控，不是宣称Candle必须自动拒绝所有错误position，更不是device fault恢复资格。

## 4. 全输出参考不调用Candle CPU量化forward

原创NumPy reader直接从同一GGUF byte ranges解码。全部291个完整FP32参数张量的SHA，与原始Candle公开CPU `QTensor::dequantize`逐一相同，8进程均核对；不是只抽一个block或仅信任模型文件名。

随后CPU独立实现完整24层图：FP64 dense reductions，在声明的norm、projection+bias、RoPE、QK/scale、softmax、PV、SwiGLU与residual边界舍入FP32，最终完整151936维head。没有用Candle CPU quantized matmul作gold；后者可能引入自己的activation量化。17个独特完整prefix逐一fresh重算，重复样本仍逐次读回全部GPU logits并验SHA。

预置门槛保持`abs(error) <= .005 + .005*abs(ref)`，同时relative-L2≤.005；不是根据真实权重结果放宽。两种原始运行模式、request-limit1/50、两个独立进程replicate17/29，共8进程；replicate只反转请求/配置顺序，**不是随机权重seed或两个新模型**。

| 输出类别 | 数量 | 原encoded-weight参考通过/拒绝 |
| --- | --- | --- |
|resident FP32 dense正常调用|208|208 /0|
|packed Q8正常调用|208|56 /152|
|故意错误position|32|0 /32|
|全部完整输出|448|264 /184，其中32为故意负控|

数值门槛是本研究的精度预算，不等于困惑度/聊天质量标准。packed失败也不是新的非零origin逻辑bug：选定fresh调用的误差可由源码中half权重片段解释；该次级参考不替换主参考，亦不授予默认采用资格。

## 5. 重放、来源和下一步

Candle固定`12bb5d1f6350f4ea9827d9086f213feec1f0af6a`。新原生客户端和驱动复用已有私有Rust1.99.1/Cargo/cache与原依赖lock，在全新target中`--frozen`离线构建。未改参考源码、历史客户端、全局工具链或生产默认。

labs/gguf_pretrained_contracts.py的`gguf-pretrained-contracts`为标准库格式/状态合同；完整CPU归档器另需显式NumPy。普通单测不启动Rust、GPU或tokenizer。设备selector `candle:pretrained-qwen`仅作发现入口。

正式raw只计`outputs/metal-candle-pretrained-final-20261017-*`；两个先行pilot及早期CPU归档仍保留，不重复计数。模型、原始完整logits、compiled binary/target留ignored目录，未清用户模型或cache。

设备资产布局为root下`gguf-api.json`、`tokenizer-api.json`，`gguf/model.gguf`及其README/LICENSE，以及`tokenizer/`内的config、generation_config、tokenizer、tokenizer_config JSON与README/LICENSE。[source capsule](../sources.md#candle-pretrained-sources-20261017)给出每份原文件的固定URL/大小/hash；`assets`只核对明确准备的文件，不联网安装或隐式下载模型。

source/lock/compiler/SDK/OS/SoC、GGUF codec/模型/词表/template、dtype、KV位置/布局、生成策略或真实消费者变化均重开。下一步仍包括原BF16/量化任务质量、更多prompt/规模/长context、LoRA、实际内存/并发/取消/服务生命周期，以及可信GPU timeline/counter；完整小模型执行不关闭整个GPU/Metal研究目标。
