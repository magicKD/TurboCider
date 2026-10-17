# MLX-LM连续批处理：提前安排下一步，返回token已入缓存

<!-- knowledge
id: technique-metal-case-studies-mlx-lm-continuous-batching-and-cache-handoff
type: case-study
stage: prefill, decode, pipeline
operator: attention, sampling, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, stale-cache, lifetime, launch-bound, precision-loss
semantics: approximate, contract-only
representation: bf16, fp32
boundary: producer-consumer, cpu-gpu, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [ragged KV与精度](mlx-lm-ragged-cache-and-padding.md) · [固定来源/PR](../sources.md#mlx-lm-batch-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十轮。本页研究并实际运行原始`BatchGenerator`的进程内连续批处理，不是HTTP server、paged serving或synthetic model。最重要的接口差异：**返回token时，缓存已消费它**。不能沿用[前轮llama客户端](llama-pretrained-output-demand.md)“最后选择token尚未入cache”的约定。

## 1. PromptProcessingBatch与GenerationBatch分工

[固定generate.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py)将队列、prompt batch和generation batch分开。

`insert_segments`会把最后prompt token独立成最后segment。prompt路径将本轮输入**右填充**，调用cache.prepare，按step分块执行，只eval各层cache state，最后finalize把右填充旋到左侧。generation路径则用最后真实prompt token启动预测；不是把所有请求直接补成同一真实长度。

原model函数虽构建head表达式，prompt路径只消费cache state。按lazy图依赖推理，无消费者的head不必执行；若观察器额外保存所有prompt logits，就改变了需求。这里是源码/dataflow解释，**不是实际逐kernel计数证明**。本轮只保存原API本来返回的完整logprobs，没有给model.call装hook强求未消费输出。

调度器`_next`先服务已有decode，在completion容量尚有空间时才补prefill；prompt完成的请求移到generation batch，其余继续分块。这缓解长prefill阻塞短decode，但本轮未做相同质量预算的生产吞吐比较。

## 2. 为什么导出cache包含刚返回token

`GenerationBatch`构造时先调用一次`_step`，准备第一个预测。随后每次`next`：

1. 把当前待返回token作为本轮model输入，推进KV。
2. 计算、采样并`mx.async_eval`下一token与下一完整logprobs。
3. 等当前token/当前logprobs可读，将当前token加入`self.tokens`并返回它。

因此对prompt长度7，第一次公开返回`y0`时：返回的logprobs属于7-token prefix，但可提取cache和all_tokens长度已经是8，即`prompt+y0`。下一预测通常已提交，不过本probe的文件/metadata读回可能让它完成；不能据此声称实现了多少物理overlap或测过任意in-flight取消。

## 3. 真正移除/插入/恢复，而不是只测forward

四个完整模型进程均运行相同调度脚本：completion最多3、prefill最多2、chunk16，起始真实prompt长度7/31。短请求第一次返回后，调用原`remove(...,return_prompt_caches=True)`；再插入255-token请求和使用导出cache的短请求continuation。两重复的独立single控制逆序，**调度脚本顺序未改变**，不是随机化服务到达实验。

输入是保留WikiText原tokenizer的真实正文prefix，未加chat模板。stop_tokens未配置，按预置length cap结束，不宣称自然EOS、完成对话或任务质量。

实际四次handoff均为24层offset=8、all_tokens长度8，且精确包含刚返回token。新continuation只需添加新输入token；完成后的合法cache offsets与all_tokens长度一致。返回cache来自原`extract`，本轮只证明这些值及继续推理正确，不证明任意并发别名/CoW/释放时序或device-fault恢复。

同时执行两种有界负控：

- `max_tokens`列表长度错误：原API抛ValueError，UID计数/待处理队列长度不变，随后正常插入和生成继续。不是任意异常恢复证明。
- 故意把已返回token再喂一次，并把公开历史少报一个，使history看起来仍像合法prefix。API没有自动证明cache内容与声明历史一致；最终cache offsets=12，all_tokens只有11。

第二种是**调用方契约误用**。FP32完整分布相对声明的正确history有大偏移，但相对真正重复token的实际prefix，最大KL约9.8e-11；模型算的是实际输入，不应归咎GPU故障。完整148行中16行是故意负控，独立记录，不混入正常资格。

事件记录本身也需要快照。初版自有probe把可变`prefix`直接放进insert事件，生成append随后把20条历史insert字段都改成了最终长度；逐行输出prefix、词表文件和cache导出另有独立copy，未受此字段别名影响。保留初版driver/curator/plan/raw/capsules后，仅将自有事件改为`list(prefix)`，用全新source/plan重跑四进程及post-hoc控制。最终20条insert snapshot正确，148份同scope完整payload与早期逐位一致；这不证明观察器计时无扰动。初版13项检查未覆盖此字段，新增第14项专门否决该别名错误，未改上游或旧数学门槛。

## 4. 原API logprobs不等于原始logits，exp总和也未必恰为1

正式60个logprobs向量全部保存151936项，其中52正常/8故意错误history。源码用`logits−logsumexp(logits)`，但BF16减法/表示会舍入。完整向量提升到FP64后计算`log(sum(exp(logprobs)))`，BF16约−0.05840到+0.05158，FP32约−4.36e-7到+1.23e-6。

证据同时保留原API chosen-token NLL、原概率总和的log和**明确重新归一化后的**分布指标，没有悄悄用后者抹掉原API值。采样器/score consumer可能有自己的归一化，不能凭导出误差就宣称采样错误；本轮greedy的选择只取argmax。raw-logit门槛只用于88个raw行，不对60个normalized-domain向量伪造“原门槛通过”。

## 5. BatchStats是host工作计数，不是GPU profiler

[generate_utils.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate_utils.py)的计数和snapshot可解释调度，但不能当per-kernel时间：

| 实际计数 | 本轮每进程 | 正确理解 |
| --- | --- | --- |
| `prompt_tokens` | 291 | 5个接受请求共296个新prompt tokens，各留最后一个给generation初始化；不是丢了5个 |
| `generation_tokens` | 15 | 包括13个正常返回和2个故意负控返回 |
| `generation_steps` | 12 | 一次可返回多个请求token，不是12个CB/kernel |
| `decode_time` | 保存原值 | 对`generation_batch.next`的host区间，不含所有caller/首次generation准备，也非独立GPU忙碌时间 |
| `peak_memory` | 保存原值 | start/end的历史process高水位取max，不是本window或KV专属peak |

原prompt代码会`mx.clear_cache`，这是allocator缓存释放请求，不是“清掉模型KV”或“清空所有编译PSO”的证明。官方[get_peak_memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html)也明确峰值需区分reset/历史范围。本probe在stats前已做整模加载、逐参数readback及manual控制，FP32还保留原BF16载体；约4.53/1.79GB的读数不能当优化引擎最小驻留或生产峰值。

原utils导入把子进程NOFILE从原值改为2048/4096，BatchGenerator将子进程MLX wired limit从0改为设备建议值55662788608，close调用原恢复逻辑回0。没有运行sysctl或改宿主/其他进程配置；源API限制的申请与真正物理驻留峰值仍是不同结论。所有这些发生在显式owned child，不进入普通CPU tests。

## 6. 两个PR状态与实际能力分别看

[PR1141](https://github.com/ml-explore/mlx-lm/pull/1141)已于**2026-04-15**合并：empty-side placeholder原先借用了nonempty侧B，混合2个有cache和3个空请求后keys行数可能4而offset长度5。修复按各自`c.offset.shape[0]`分配。**不是修K/V head宽度**；当前源码还把原patch的empty reshape改成dtype明确的zeros，不能把整份旧diff当今日文件。

额外原始cache调用覆盖BatchKV/BatchRotatingKV两类、empty3/full2的双向extend、right-pad/finalize/extract，40个完整K/V tensor逐元素等于独立CPU列表。另8个正式宽度控制验证K dim4/V dim7；这是独立的cache接口控制，Qwen模型自身K/V都是64。本轮未执行pre-fix代码、完整上游tests或rotating eviction，不能称复现了历史崩溃。

[PR941](https://github.com/ml-explore/mlx-lm/pull/941)于**2026-08-21**关闭且**未合并**。提议是merge时展开成float、extract时重新量化，主要节省长期LRU存储，不是证明batch attention一直做低bit算术。所测固定源码里，QuantizedKVCache没有merge、BatchGenerator没有kv_bits参数，原`_merge_caches`实际拒绝该对象。不能从单生成器支持quantized KV推断连续batch也自动支持，更不能拿已关闭PR当有效修复。

## 7. 学习与复查

CPU合同/tests校验已保存的完整数据、UID/history长度、失败边界、PR状态与统计范围；不启动MLX/模型。实际客户端是原始batch probe、混合空cache控制及post-hoc形状控制。均仅写新ignored目录、使用原模型/源码，未发PR/issue/comment/commit。

复查条件：scheduler/source/版本、token-return时点、cache class/offset/history/采样器/stop sequence、成员变更、异步完成和任何stats/memory/clock定义改变。本轮不是生产服务、任意fault/abort/OOM恢复、LoRA/长context资格或跨引擎速度排名；BF16数值敏感点与物理profiling继续研究。
