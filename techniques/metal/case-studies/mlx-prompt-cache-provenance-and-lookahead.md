# 原始MLX-LM KV来源：prefill舍入会延续到decode，返回与预采样不是同一进度

<!-- knowledge
id: technique-metal-case-studies-mlx-prompt-cache-provenance-and-lookahead
type: case-study
stage: prefill, decode, pipeline
operator: attention, gemm, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, lifetime, stale-cache, measurement
semantics: approximate, contract-only
representation: bf16, fp32
boundary: weight-runtime, producer-consumer, request, cpu-gpu
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [分块/数值路线](mlx-prefill-chunking-and-numerical-routes.md) · [此前batch交接](mlx-lm-continuous-batching-and-cache-handoff.md) · [来源](../sources.md#mlx-prefill-schedule-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十二轮。本页跟踪未修改原始单请求`generate_step`、原KVCache、真实24层Qwen。不是BatchGenerator、paged缓存、HTTP服务或新缓存格式部署。固定teacher-forced续文让不同prefill策略在后续decode仍接收相同token，从而观察**producer算术历史如何留在KV里**。

## 1. 两次yield，三份logits，但只缓存了两个新token

[`generate_step`](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py)先计算当前token，再在yield之前计算下一token/logprobs。`max_tokens=2`限制返回数量，不保证内部只产生两份sample；本例完整观测如下。T0/T1/T2是固定真实续文的前三个token，不是模型greedy选择。

| 7-token、step4的事件 | KV有效长度 | 得到什么 |
| --- | --- | --- |
| prefill 4，再2 | 6 | 原prompt前6个token进入cache |
| `_step`消费最后prompt token | 7 | raw0，采样T0 |
| lookahead `_step`消费T0 | 8 | raw1，采样T1 |
| 第一次yield | 8 | 返回T0、API logprobs0 |
| lookahead `_step`消费T1 | 9 | raw2，采样T2 |
| 第二次yield，随后达到上限退出 | 9 | 返回T1、API logprobs1；T2未被消费 |

192个实际观察case都得到3份raw、2份返回logprobs，所有24层offset在两次yield分别为L+1/L+2，最终是L+2。第三份raw是lookahead而不是第三个已返回token；第三次采样的T2也不在cache。采样计数、返回计数、提交token、KV长度和后端dispatch数不应共用一个“generation steps”。

本轮并未执行中断/serialize/reload/resume；不能把这一状态表当作跨进程恢复验证。此前batch导出也已包含返回token，但不能据此假设两套API内部预取状态完全相同。恢复策略应分别核对其实际返回历史、已消费token和pending样本，不能重复喂回已缓存token。

## 2. KV的有效区与分配容量不同

读[`KVCache`](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/cache.py)：step=256增长，`.state`是底层keys、values及offset的三元组，前两项可包含未使用容量；`keys_and_values()`才返回截至offset的有效视图。物理capacity256不等于本次7/9/32/34个真实位置。

初版自有collector错误地把`.state`前两项当有效KV，遇到shape断言后停下。失败receipt、执行时client完整hash与副本保留；v2改读有效区，再重新冻结plan/从头跑正式数据，不将这次采集器错误描述成GPU不支持或引擎bug。首个失败和后来的21向量smoke都不计正式1344向量。

本轮仅对doc0的两个prompt长度，六step、两dtype、两重复，在以下三个时点保存**全部24层K/V有效元素**：prompt刚完成、observed两yield完成、unobserved两yield完成。共144个bank/6912个完整tensor；每段有shape、字节offset/长度和SHA，CPU独立核对无重叠、无遗漏tail及完整文件身份。源stream已同步，读回是显式诊断成本，不称zero-copy或测得了物理cache峰值。

其他三文章确有native完整48个tensor的hash及观察开关对照，但未保存其raw banks；不把这些hash-only观察升级为独立数学/字节回放。所选banks也只和同dtype原生默认策略比较，没有另写24层独立KV oracle。

## 3. decode query已相同，KV仍保留prefill历史

所有策略在prompt末尾和后续两次lookahead都是query=1、GQA7，源码准入同为vector候选；不同的是此前写入各层的K/V值和上层隐藏状态。切换到相同decode API不会重算历史KV，也不能自动抹去prefill舍入。

doc0、BF16、step4相对默认2048：

| 时点 | 最早不同的层，零基 | 48个有效K/V中不同的scalar总数 |
| --- | --- | --- |
| L7 prompt完成 | 1 | 34785 |
| L7再消费2个token | 1 | 45083 |
| L32 prompt完成 | 1 | 144008 |
| L32再消费2个token | 1 | 152547 |

observed/unobserved最终bank逐位一致，且各自相对默认的完整差异表也一致；两重复同scope可核查。分块页的完整词表KL表显示后续两步差异仍存在。不同元素数量随cache增长不能直接当“单位位置误差变大”；长度和比较分母同时变了。

FP32也不逐位等价：例如L7/step4在prompt时34548个scalar与默认不同，但所有raw输出通过原模型预算，所选KV的差距很小。这说明“cache字节不同”不是“模型算错”；是否可复用仍取决于消费者容差和它声称保持的语义。

## 4. 第一个不同的层能排除什么

[`Qwen attention`](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen2.py)先norm/投影/RoPE，更新KV，然后才调用本层SDPA。按这条原图依赖，第0层K/V差异不能仅用其**后面的第0层attention输出或最后词表head的数学差异**解释；这不是排除任意实现/观察效应或证明某种内存破坏的物理取证。

本例step4的doc0与默认在第0层KV完全一致，差异从第1层出现，和前轮实际输入SDPA精度证据相容。但step1/5的同scope比较在第0层就已不同（两dtype都观察到），所以分块还影响进入attention之前的shape相关投影/norm/RoPE算术路径。这里给出依赖边界上的排除，不冒充精确到某条shader/编译优化的root cause。

**相同token、位置、权重和存储dtype仍不保证cache bitwise相同。** 前轮GQA fallback解释不能被扩成“所有分块差异只由该guard引起”；本轮也没有证明某条路径违反它自己的typed舍入策略。

## 5. 缓存来源合同与采用前检查

若缓存复用层声称重现某一数值recipe，producer身份除了tokenizer/prefix/模型与位置外，还应记录dtype/KV格式、来源版本、实际chunk序列、已消费/返回/pending状态和允许的数值预算。step size只记录一个上限仍不充分：同一个8会因剩余prompt不同产生6、7或8等实际query。batch/多请求组合也可能改变物理shape，见[ragged控制](mlx-lm-ragged-cache-and-padding.md)。

原`save_prompt_cache`支持调用方metadata，并不等于本轮已实现自动producer-recipe失效/重放；本轮没有调用该序列化API。若需要另一个recipe的历史状态，要显式验证或重算，不能假设仅cast旧KV、换未来step、清allocator cache即可恢复其完整数学目标；本轮没有验证这些修复候选。

另外，原API的768个logprobs不机械套raw门槛；BF16原log-mass偏离0最大约0.06149，FP32约1.23e-6。curator保留原值和显式重新归一化的KL/NLL诊断，不因名字叫logprobs就丢掉质量差异或混作原raw参考通过。

CPU状态/调度合同、、tests不在普通运行中导入MLX或启动设备；真正原API仅由显式客户端启动。公开capsule为全部metadata/指标/hash，不含原始K/V向量、checkpoint或compiled产物。

复查条件：producer source/shape/chunk/precision、RoPE/mask、KV接口/布局/有效长度、token历史、generator/sampler/stream及consumer预算变化。更长上下文、代表性任务、真实greedy、跨进程恢复、serving和正式性能/内存/物理pipeline仍未验收，不据本轮设缓存或分块默认。
