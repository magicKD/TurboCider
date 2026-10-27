# llama.cpp Metal观察边界：FP32 KV转FP16、query-only callback仍同步

<!-- knowledge
id: technique-metal-case-studies-llama-metal-flash-and-eval-observer
type: case-study
stage: prefill, decode, ffn, pipeline
operator: attention, gemm, quantization, scheduling, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, measurement, graph-interruption, stale-cache, lifetime
semantics: numerically-equivalent, approximate, contract-only
representation: q8, fp16, fp32, gguf
boundary: producer-consumer, cpu-gpu, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [完整模型与输出需求](llama-pretrained-output-demand.md) · [来源](../sources.md#llama-pretrained-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十七轮profiling/状态部分。原始模型图和实际callback共同证明两件事：**API设置FP32 KV不代表Flash Attention全程FP32；callback只问metadata并返回false，也不代表没有同步扰动。** 图中op数、allocated-buffer标签和API完成wall仍不是物理dispatch/counter。

## 1. F32 cache与FA输入是两个边界

客户端将`type_k/type_v=GGML_TYPE_F32`，日志确认256 cells/24层的K/V各3MiB、共6MiB的F32 KV buffer。这不是原始默认F16 cache，也不是因模型权重Q8便采用Q8 KV。

但[llama-graph.cpp::build_attn_mha](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/src/llama-graph.cpp)在Flash分支中，把F32 K/V显式cast为F16后才建立`ggml_flash_attn_ext`，再设置F32 accumulation。原始Metal graph的实际metadata显示：

| 配置，39-token prefill | 关键IR op | FA输入types |
| --- | --- | --- |
|Flash off|24个SOFT_MAX、217个MUL_MAT|无FA node|
|Flash on|24个FLASH_ATTN_EXT、169个MUL_MAT、48个CPY|每层均`[f32,f16,f16,f16]`，对应Q/K/V/mask|

这些是原始scheduler查询到的**图节点**；48个CPY与源码两次cast/层一致，不声称实际GPU必有48个独立kernel。选定节点buffer为MTL0，证明分配/调度对象属性，不代替物理placement或ISA采样。

因此F32 accumulator设置不会恢复已经舍入的K/V。Flash on/off的54组相同prefix末行对照在本轮严格门槛下全部拒绝，不能以同样生成8个词来宣称数值等价；它也不能单独证明全部误差只由这两次cast造成，融合算法/归约/矩阵路径仍会变化。

### 不要把另一个量化KV PR套在这里

[PR #27390](https://github.com/ggml-org/llama.cpp/pull/27390)引入**量化KV先展开为F16 scratch再跑FA**的路线，含pipeline/scratch与V-view-of-K复用。2026-10-17取得的API为merged，merge时间2026-08-20；API/diff/base/head/merge SHA均另存。

固定源码中的该helper又有Q长度<32禁用条件，支持Q4/Q5/Q8等KV类型。原PR diff的选读版本没有这个后来加入的Q阈值，不能把活动源码所有guards都说成原patch已有。更重要的是，本轮上层graph送给FA的是F16 K/V，**不是这个量化KV prepass的运行复现**。权重Q8、cache F32、FA operands F16和quantized-KV scratch是不同层。

## 2. query-only eval callback也会改变程序

public `llama_context_params.cb_eval`接入GGML scheduler。源码[ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-backend.cpp)的区别是：

- 没有callback：整个split调用`ggml_backend_graph_compute_async`。
- 有callback：逐node询问是否需要数据，构造graph view执行，随后**无条件backend synchronize**；若需要某node的数据，还会继续切分执行范围。

本轮callback始终在ask阶段返回false，仅记录少量FFN/head/FA的ne、dtype、source types、buffer标签及IR op计数，没有读tensor数据；但每split仍多出源码中的等待。不能用“没有取值”替代observer开销验证。

4个独立诊断进程覆盖last/all与Flash on/off，各一个真实请求加缓存控制，共56个完整调用payload。与对应无observer正式进程比较，56组全部逐位相同。这个结论只证明本fixture**保值**，不证明callback无开销、无同步、跨backend overlap不变或所有模型graph惯性。

正式benchmark进程根本不设置callback，不以query返回false伪装关闭observer。诊断wall不混入正式round1中位数，IR op数也不当kernel launch数。

## 3. 位置拒绝、恢复与自有客户端清理错误分开

本固定普通causal seq0的[batch验证器](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/src/llama-batch.cpp)要求新batch从已有max position+1连续开始；M-RoPE等分支规则不同，不能推广成所有模型/序列都一样。

44次控制先建45-token cache，max position44，再故意提交position46而不是45：原始`llama_decode`全部返回`-1`，metadata max position保持44。随后**不clear context**，仅修正为45，正常调用全部返回0，max position变45，完整logits均已保存和独立检查。

这比只看`return0`强，但仍须保留数值范围：与最早cached output按当前严格门槛比较，仅8/44恢复对照通过；fresh/chunk对照也并未因此变成通过。两次准备past所走的batch/矩阵路径不同，缺少同历史无-refusal的完全配对不能把差异全归给拒绝、也不能宣称任意fault/OOM恢复。最稳妥结论是**该输入验证分支的拒绝/position保持与合法继续已实测；完整数值预算另列**。

最早pilot的自有C++客户端把预期`-1`直接throw，未在异常路径free context/model，退出时又触发Metal资源集合断言。修正只在本库API客户端：增加异常清理、把特意的拒绝记录成控制、及时写journal；上游源码未改。旧失败、两个中间成功pilot的源码/driver/binary完整SHA及raw日志都保留，不称为“修复了llama.cpp Metal设备bug”，也不据其中部分文件回填正式通过。

## 4. 同模型对照仍需精度与输出契约

本轮CPU量化forward也不是gold；独立oracle解码同一GGUF后运行完整FP64-stage数学图。全部616个API调用、3676个请求的151936维rows均检查，63通过/3613预算拒绝，包含更早prompt位置的完整分布，而不是只验sampled token。

两个prompt的实际生成IDs和文本在各recipe中相同，但36个更早位置的top1已与独立数学参考不同。所有recipe的合格性能字段保持null；~33–39ms的Metal诊断wall和~79ms CPU wall不构成通过数值门槛的跨引擎倍率。参考[前轮Candle](candle-q8-fragments-and-request-cost.md)时，还要区分其最后层FFN计算量、KV cat、tokenizer/IPC实现和resident dense模式，不能只因GGUF SHA相同就隐藏这些差异。

日志里的mapped buffer/compute buffer MiB是运行时报告分配口径，不是实际RSS/峰值或DRAM流量。embedded Metal library在此构建中嵌入MSL source，不保证消除runtime library/PSO首用；pilot后的warm driver cache也不是冷启动。没有新Instruments、物理GPU timeline/PMU/能耗/ISA证据，M4结果不扩为M5/NAX或ANE。

## 5. 学习入口与重开

labs/llama_runtime_contracts.py的`llama-observer-contracts`解释callback切分/等待，`llama-output-contracts`解释选择器/getter；测试只读CPU与归档，不启动设备。可选API驱动显式`--observe`产生独立诊断cohort；设备selector `llama:graph-observer`不是自动qualification。

source/API/library、callback/graph fusion/queue、cache/FA dtype、batch/output/sequence/position、tokenizer/consumer或硬件/SDK/OS改变均重开。后续优先质量校准/更强真实消费者与物理profiling，不将更多相同小case或parser字段替代这些缺项。
