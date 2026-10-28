# Candle原始Qwen2图：GGUF、KV状态、提交阈值与空capture反证

<!-- knowledge
id: technique-metal-case-studies-candle-qwen2-queues-and-capture
type: case-study
stage: prefill, decode, ffn, pipeline
operator: gemm, gemv, attention, norm, rope, swiglu, quantization, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: launch-bound, measurement, lifetime, stale-cache, graph-interruption
semantics: numerically-equivalent, approximate, contract-only
representation: q4, fp16, fp32, packed
boundary: weight-runtime, producer-consumer, cpu-gpu
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [packed view/copy](candle-packed-views-and-copy.md) · [来源](../sources.md#candle-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十四轮模型/提交部分。通过原始GGUF writer/loader实例化**未修改的Candle `quantized_qwen2::ModelWeights`**，实际执行embedding、两层GQA/RoPE/attention、MLP、norm、LM head和KV cache。权重是小型合成fixture，非预训练Qwen；224个完整64维logits输出全部通过独立数学参考，不能扩成文本质量或完整聊天服务资格。

## 1. 不用替身attention凑“完整模型”

固定Candle `12bb5d1f6350f4ea9827d9086f213feec1f0af6a`，新任务私有Rust1.99.1/Cargo依赖lock；原始参考checkout不改。核心入口为[quantized_qwen2.rs](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-transformers/src/models/quantized_qwen2.rs)、[GGUF reader/writer](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/quantized/gguf_file.rs)、[quantized nn](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-transformers/src/quantized_nn.rs)和[RoPE](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-nn/src/rotary_emb.rs)。

fixture为H64/F96、2层、4 query heads/2 KV heads、head-dim16、vocab64/context64；27个参数中矩阵使用原Q4_0 codec，norm/bias为FP32。用原writer生成GGUF v2，再由原loader加载Metal0；不读取私有字段、抽出改写模型body或手写一个GPU attention替代原实现。

模型本身保留CPU初始化工作：原quantizer、GGUF IO、若干元数据/frequency构造、公开dequantize等不是GPU热forward。packed forward与`CANDLE_DEQUANTIZE_ALL=1`的resident FP32 dense模式分别在新进程中、首次构造前固定；不在已缓存thread-local flag后翻环境变量。两模式相同seed的GGUF/encoded参数相同。

## 2. 验证的是完整logits，不是只看token恰好相同

每进程4轮，依次prefill4、decode1、带已有prefix的chunk3、decode1；清KV后prefill9、decode1，再清KV对相同完整10-token prefix作一次fresh forward。原mask cache、KV cat、RoPE位置和clear API都实际运行。

两seed×packed/dense×提交阈值1/50，共8进程、224个完整logits输出。32个cached-vs-fresh对照通过同一数值线，184个把旧输出错用于新prefix的stale负控均拒绝。缓存与fresh可能因GEMV/GEMM路径不同而不逐位相同，不能伪造bit-exact cache资格。

独立CPU参考从实际block bytes解码，完整计算embedding、FP32边界的RMSNorm、QKV/bias、half-split RoPE、GQA repeat、causal mask/softmax、attention output、SwiGLU、residual与LM head；dense乘法使用FP64，再按声明的FP32边界舍入。**没有用Candle的CPU量化matmul当gold**，后者还有自己的activation量化/向量实现。

预置门槛为`0.005+0.005×|ref|`，所有logits而非抽样检查。packed最大误差约7.02e-4/3.30e-4（seed17/29），dense约4.69e-7/2.68e-7；它们都针对encoded权重，不是原float权重或真实任务质量的承诺。

## 3. 提交粒度的收益不能写成矩阵指令峰值

原[commands.rs](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-metal-kernels/src/metal/commands.rs)默认`CANDLE_METAL_COMPUTE_PER_BUFFER=50`。计数增加于compute/blit encoder请求，复用现有encoder，达到条件时commit/swap；**这个逻辑计数不等于physical encoder或每CB恰好50个kernel**。

Untracked hazard模式依赖原输出fence map、跨encoder wait/update和完成后的清理。`flush_and_wait`等待最后一个FIFO CB再检查先前错误；CPU读回另用`flush_and_wait_current`保存特定CB完成关系。Arc/Mutex确保CPU owner/编码边界，不自动证明GPU可见性。本文只验证单调用线程下的这批依赖，没有复现所有并发readback race、取消、OOM或设备fault恢复。

下面是warm轮host `forward+device.synchronize`中位数µs，单格seed17/29；token上传、GGUF加载、reset调用和随后D2H在窗外：

| 模式 / request-limit | prefill4 | 第一次decode1（position4） |
| --- | --- | --- |
|packed /50|491 /492|372 /366|
|dense /50|451 /461|345 /350|
|packed /1|1087 /1150|1106 /1082|
|dense /1|1060 /1048|1090 /1125|

在这个小型完整图，频繁提交配置明显慢，packed也未胜resident dense。它与源提交/等待开销相符，不是GPU busy/能耗归因或“50全局最优”证明；两模式分进程反序采集，仍没有全机持续独占、长时热状态和更大真实权重验证。Q4减少逻辑权重bytes不保证小模型latency更低。

## 4. API返回成功，capture却可能没有实际命令

调用原`MetalDevice::capture`、显式SDK stop，MTL_CAPTURE_ENABLED仅在自有子进程设置。最初所有9个before/captured/after数值输出都正常，capture API也成功，但capture stream只有8 bytes；只读parser的stats失败，timing反而exit0、synthetic且0行。**文件夹存在和exit0不能证明捕获成功或得到设备时间**。

源码在创建/flush时预先持有下一个current CB。加入一次有界GPU primer并同步，使目标batch使用捕获窗口内新建的CB后，stream增长到约18KiB，parser识别1 CB/3个目标dispatch；这与持久precreated-CB边界一致，不作为所有Metal程序通用规则。primer是额外诊断工作，不能声称该API窗口只有三次GPU操作或完整捕获了所有先前工作。

原view错误在before/captured/after仍逐位相同：owning/force通过，view仍读错prefix，9个完整输出中6通过/3失败。capture不会自动验算kernel，更不会修复错误。

只复用固定`tmc/gputrace@3d7d32e...`的只读stats/timing入口，不运行replay、buffer修改、GUI或Xcode自动化。parser的unique-kernels=4含非目标标识/函数标签，buffer-usage=0也不是实际零allocation；不要与3个受控dispatch混为一谈。timing仍为synthetic、approximate，严格adapter保留`physical_gpu_duration_seconds=null`。没有新Instruments、PMU、shader物理耗时或overlap资格。

## 5. 冻结范围和后续

本轮总计13进程/1449个完整消费者输出：linear1216中240失败，模型224全部通过，capture9中3失败；合计1206通过/243失败，全部独立CPU验算。512个linear hot样本、模型warm样本与capture/准备成本分开，失败从不进入合格加速比。

CPU学习：labs/candle_contracts.py，case `candle-model-cache-contracts`；离线测试、归档器不启动GPU。设备使用驱动/Rust客户端，显式supervisor/fresh目录、固定build-root与lock。

正式raw前缀`outputs/metal-candle-final2-20261017-*`；早期clone失败、空capture、smoke/第一批版本全部另留，最终只计冻结一批。未下载预训练权重、改全局Rust默认、系统策略或产品backend；约33GiB可用，无需清用户cache。

接下来需要真实checkpoint/更大shape、tokenizer/采样/文本请求、LoRA、并发、峰值内存和物理trace/counter。任何source/lock/库/Rust/SDK/OS/SoC、dtype/布局、queue/owner、cache协议或真实消费者变化均复查。此处原始模型图的合成证据不授予真实模型质量、M5/NAX/ANE或跨引擎排名；整体调研继续。
