# llama.cpp完整Qwen：输出需求裁剪最后层FFN，而不只是少复制logits

<!-- knowledge
id: technique-metal-case-studies-llama-pretrained-output-demand
type: case-study
stage: prefill, decode, ffn, pipeline
operator: gemm, gemv, attention, norm, rope, swiglu, sampling, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, memory-bound, launch-bound, measurement
semantics: numerically-equivalent, approximate, contract-only
representation: q8, gguf, fp16, fp32
boundary: weight-runtime, producer-consumer, cpu-gpu, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [Flash dtype与observer](llama-metal-flash-and-eval-observer.md) · [来源](../sources.md#llama-pretrained-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十七轮。新建私有构建，实际执行未修改的llama.cpp `llama_model_load_from_file/llama_init_from_model/llama_decode`，读取[前轮Candle](candle-pretrained-qwen-and-tokenizer.md)的**同一份官方Qwen2.5 Q8_0 GGUF**，没有再次下载或替换checkpoint。

本轮最有用的优化事实：**last-only输出需求会传播到最后一层FFN和LM head**，不只是最后D2H时丢掉多余行。实际图元数据证明FFN rows从39变为1；但形状变化也改变数值路径，不能承诺逐位等价或无条件性能资格。

## 1. 什么可以裁，什么还必须算

固定`ggml-org/llama.cpp@64e9bceb2c3a856efed96feda784a50947049feb`，已读[Qwen2模型graph](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/src/models/qwen2.cpp)。每层attention完成后，只有`il == n_layer-1`且存在output IDs时，才对attention输出和residual做`ggml_get_rows`，再执行最后FFN、norm和head。

对于下一token生成，39-token prefill只需要最后位置的logits；对于perplexity/逐token评分，可能真的需要全部39行。两者不能用同一“输出量”标签混在一起。

| 实际consumer需求 | 较早层FFN | 最后层attention | 最后层FFN up | LM head |
| --- | --- | --- | --- | --- |
|last-only|39 rows|仍构建39 queries|`[4864,1,1,1]`|`[151936,1,1,1]`|
|all-outputs|39 rows|39 queries|`[4864,39,1,1]`|`[151936,39,1,1]`|

表中维度是GGML原始`ne`顺序，features在前，不能与NumPy `[rows,features]`直接混读。callback实录的第一层`ffn_up-0`两模式都为`[4864,39,1,1]`；最后层norm/up/gate与head才发生选择。

为什么不能把同一规则随意提前？较早层的所有hidden rows还要供后续层建立K/V并参与attention。最后层K/V已在选择前写入，旧位置的最后FFN输出不再被下一层消费。这里是**consumer-driven graph pruning**，不是删prompt、裁context、近似attention或改变token数；但它也不是理论上最少Q计算的完整证明。

迁移到扩散/视频时须先确认consumer：DiT若需要全部image-token输出，就没有“只取最后token”的同样许可，不能机械照搬。

## 2. API的output index也不是想当然的row index

使用原始`llama_batch_init`，显式给每个token填位置、seq0和logits flag。`llama_get_logits`给出所有被选中的紧凑row；`llama_get_logits_ith`的正索引却对应**batch输入token编号**，负索引按输出倒序。

例如flags=`[false,false,true]`，紧凑输出只有row0，但正索引0不是它；正索引2或负索引−1才指向被选输出。不能把single-token decode的索引经验直接搬到prefill。当前客户端只使用合法紧凑getter；此索引负例另由labs合同解释，没有执行非法指针解引用。

all模式下，实际请求的每一行都保存并检查，**不只取末行argmax算通过**。每行的独立gold是对应`prefix[:position+1]`的完整24层CPU图和151936维head；长矩阵不是只抽一行见证。

## 3. 是完整原始引擎，仍不是无条件跨引擎排名

配置为Release、Metal+CPU、embedded MSL，BLAS/native tuning/DL backends/common/tools/server等关闭；使用全新源码副本、build和可选API客户端，不修改reference或上游代码。实际加载的llama/GGML dylib集合与逐库SHA全部核对，避免旧binary/新header ABI混用。

模型仍是24层、H896/F4864、14 Q heads/2 KV heads。原GGUF tokenizer实际tokenize/detokenize；原官方chat template由已有Jinja渲染，两个实际prompt的39个ID和生成文本与HF tokenizer逐值核对。greedy/max8、无LoRA，均在8-token cap返回，不宣称自然EOS/任务质量。

请求设置context128，实际API/日志为256；训练context32768不能混为运行分配长度。日志的`model type=1B`也不是新的checkpoint：本固定Qwen2分类器对24层按embedding长度做粗分类，896落到1B分支。模型SHA/API几何与官方配置才是判断依据。

GGML使用预分配KV view和`SET_ROWS`更新，Candle选定类则用cat续接KV；GGML还裁剪最后层输出。即使同一GGUF，图、缓存、请求实现和数值路径都不相同，不能仅将wall相除便宣布某个项目整体更快。

## 4. 完整数值检查保留了大量预算拒绝

正式部分有8个Metal进程：last/all×Flash on/off×两个独立order replicate。另有2个CPU控制、4个独立metadata-observer诊断进程，均读取同一文件。每个常规进程4请求，observer进程1请求。

| 类别 | 完整API调用输出 | 实际请求的完整词表rows |
| --- | --- | --- |
|正式Metal|448|3168|
|CPU控制|112|112|
|observer诊断|56|396|
|合计|616|3676|

继承且校验前轮独立GGUF decoder/完整CPU oracle身份，全部291个解码权重SHA一致，68个独特prefix完整重算。原门槛不变：逐元素`.005+.005*abs(ref)`且relative-L2≤.005。

3676行中只有63行通过，3613行超预算；CPU控制112行也未过线，不能把原始CPU量化forward自动当gold。CPU q8权重的vec-dot trait会使用自己的Q8 activation表示。Metal fresh-prefill末行的整体L2约0.0019–0.0022仍可能同时有数千个近零logits超逐元素线；“均值误差小”不等于该预算通过。

所有44个请求生成相同两组8-token结果，但请求的更早位置也有36行argmax与独立参考不同。last-only对all末行的54组对照有46组过线、8组失败；不伪造bitwise输出需求等价。上述是**固定数学预算的拒绝**，不是未经任务评估便宣称引擎聊天质量低或硬件计算错误。

## 5. 性能只保留诊断值

host wall从原chat render开始，包含IPC、原GGUF tokenizer、token上传、decode/sync/getter、保存待flush向量/有限值检查/greedy、原detokenizer返回文本。文件与SHA flush在窗外；all模式本身请求/保留更多输出，因此复制和验证量也更多。

各配置round1共4个样本，两个进程各两个prompt，原始round0/first-use也保留：

| 配置 | warm host中位数ms | 数值合格中位数 |
| --- | --- | --- |
|Metal last-only /Flash off|33.208|`null`|
|Metal all /Flash off|39.056|`null`|
|Metal last-only /Flash on|33.086|`null`|
|Metal all /Flash on|37.903|`null`|
|CPU last-only /Flash off|78.939|`null`|

只对完成并数值合格的同契约consumer才讨论合格倍率。本轮没有发布“llama.cpp快于Candle多少”或全局输出开关最优；CPU构建也不是所有CPU优化均启用的冠军基线。没有清system/driver cache、锁频或独立GPU时间/带宽/能耗。

## 6. 可执行入口和重开条件

CPU case `llama-output-contracts`见labs/llama_runtime_contracts.py；离线测试不启动外部引擎。C++客户端/驱动需要显式私有固定source/build与前轮只读资产，device selector为`llama:pretrained`。

正式raw前缀为`outputs/metal-llama-final-20261017-*`；三个pilot保留且排除正式计数。源码/ABI/library/compiler/SDK/OS/SoC、GGUF/词表/template、dtype/FA/outputs/batch/positions、callback/队列或consumer变化均重开。后续仍需任务质量预算、更多模型/prompt/长context、LoRA/服务/真实内存和可信物理profiling；整体研究继续。
