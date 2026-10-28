# 原始MLX-LM完整Qwen：ragged KV的逻辑位置、物理填充与精度路径

<!-- knowledge
id: technique-metal-case-studies-mlx-lm-ragged-cache-and-padding
type: case-study
stage: prefill, decode, ffn, pipeline
operator: attention, gemm, norm, rope, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, wrong-layout, measurement, memory-bound
semantics: approximate, contract-only
representation: bf16, fp32
boundary: weight-runtime, producer-consumer, cpu-gpu, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [连续批处理/交接](mlx-lm-continuous-batching-and-cache-handoff.md) · [来源](../sources.md#mlx-lm-batch-sources-20261017) · 后验padding控制 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十轮。真正执行`mlx-lm@93e6d4a6447b3080861985947f0820e76fcb957d`的原始Qwen2完整模型、KVCache/BatchKVCache和BatchGenerator，复用上一轮同一官方BF16 checkpoint；不是另写一个model stub。环境实测为MLX0.32.3、Transformers5.19.0、M4 Max，不继承此前MLX0.32.0实验的版本资格。

四个正式进程（BF16/显式FP32权重提升控制×两重复）保存148个完整151936维输出：88个raw logits、60个原API logprobs。24个独特prefix全部独立重算24层CPU参考，actual loaded的290个参数逐一SHA匹配独立BF16位展开。FP32正常raw行40/40过原门槛；BF16正常raw行0/40过线，拒绝不抹去、不用同词替代。后验另20行，不混入正式148。

## 1. 原始路径怎样把模型放到GPU

[Qwen2源码](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/qwen2.py)使用原nn.Linear、fast RMSNorm/RoPE/SDPA及[shapeless-compiled SwiGLU](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/activations.py)。本轮加载完整24层、H896/F4864、14Q/2KV heads、D64和tied embedding/head；验证default device为GPU，不声称每个内部op的物理放置/ISA都已取得counter证明。

BF16路径保留原文件精确权重值，FP32控制只提升表示，不重新量化/校准。两路都是完整原模型，不是把另一个引擎CPU forward当gold。私有源码副本未改tracked模型/runtime；只添加源码checkout缺失的`_version.py`本地packaging标签，明确不是上游release。未安装新package、下载新模型或改生产默认。导入原utils会在**子进程**降低NOFILE，BatchGenerator另外申请/恢复该子进程MLX wired limit，见交接页。

runtime身份记录package版本和`mlx.core`扩展文件SHA，未逐一枚举全部dyld image或Metal pipeline产物。因此这些身份不能证明所有底层kernel/ISA与另一次安装相同。原WikiText短token窗口沿前轮数据卡与Wikipedia贡献者CC-BY-SA3.0/GFDL署名，完整语料/权重始终在ignored目录，不改授权。

版本登记沿用既有`runtime_build_id`维度：对commit、MLX/Transformers版本、core扩展SHA及model SHA作排序紧凑JSON哈希，记录为`tc-observed-context-v1-...`，派生方式见验证capsule。它是观察依赖指纹，不冒充完整native build，也没有为单一package放宽检索schema。

## 2. 一个batch有三个不同的“长度”

[原始cache.py](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/models/cache.py)的`BatchKVCache`维护：

- `offset[i]`：每请求的逻辑RoPE位置。
- `_idx`：所有行共用的物理有效终点。
- `keys.shape[2]`：已分配容量，不是每请求真实context。

例如真实长度`[7,31,255]`被left-pad到255，padding=`[248,224,0]`；初始offset取负padding，RoPE先用该offset，append后offset为真实长度。mask排除物理padding列；即使decode只有一个query，也不能照单请求经验把batch mask都删掉。

两组padding值0与23的12份完整输出逐位相同，支持这些已测case正确屏蔽了padding内容；不是证明任意padding/shape都无影响。故意把padding metadata错填全0后，短请求把填充当真context，FP32四份负控全部拒绝原目标，但全部通过独立的**实际零填充prefix**参考。解释输入契约错误，不改写其原目标失败或归因硬件。

## 3. 实际filter/extend/extract跨过了256边界

原step=256只是增长块。实际三请求控制的每层状态一致：

| 操作后 | 逻辑offsets | `_idx` | capacity | left padding |
| --- | --- | --- | --- | --- |
| 255长度prefill后再append1 | [8,32,256] | 256 | 256 | [248,224,0] |
| filter仅保留前两请求 | [8,32] | 32 | 32 | [24,0] |
| extend独立primed257请求 | [8,32,257] | 257 | 257 | [249,225,0] |
| 三请求再append1 | [9,33,258] | 258 | **513** | [249,225,0] |

`filter`移走所有剩余行共同的最小left pad，所以物理index/容量都缩小，offset不变。incoming原KVCache容量512，`merge`构造矩形batch时只取257有效行；`extend`按最大index右对齐。下一append溢出后，代码在非256整倍数的prev位置先裁到257，再加256，故513。**不能把step直接当容量对齐保证。**

`extract`得到连续的单请求KV，再继续decode与独立完整prefix比较。最长请求extract后容量258，下一append容量514，和保留原512容量的单缓存不是同样分配历史。源代码调用`mx.contiguous`，不能无条件称zero-copy，也不能凭逻辑`nbytes`判断所有底层alias/物理峰值。它是矩形连续cache，不是vLLM式物理paged block table。

## 4. metadata正确仍不授权BF16数值等价

原预算沿用逐元素`.005+.005*abs(ref)`及relative-L2≤.005，只对raw logits使用。实际40个正常FP32行全部通过，40个BF16行全部拒绝。相同dtype的single与ragged末行对照，FP32六份通过，BF16六份拒绝；后者最大abs差约2.28125。两重复的74个完整payload分组逐位相同，重复性不代替数学通过。

这还不能归因padding mask或某个kernel。后验控制用**同一7-token真实prefix、B=1、同原始模型**，只改变cache/物理padding，logical最后位置始终6：

| dtype /控制 | 与同dtype无填充max abs | 对独立数学参考KL | 原raw门槛 |
| --- | --- | --- | --- |
| FP32，全10种cache/padding | ≤0.000112 | ≤1.11e-9 | 10/10通过 |
| BF16，单KV/no-cache/BatchKV pad0、pad1 | 0 | 约0.126069 | 全拒绝 |
| BF16，BatchKV pad7（物理14 tokens） | 2.273438 | 约0.001401 | 拒绝 |
| BF16，BatchKV pad128/248/256 | 2.273438 | 约0.001979 | 拒绝 |

所以本例并非“批处理一定更坏”：部分填充后更接近FP32-stage数学参考，但仍没过原预算。no-cache、scalar KV和pad0完全一致，不能仅凭怀疑cache类型就宣布找到了原因。物理长度可改变matmul/attention等算术路线；**本轮没有primitive级分解或ISA trace证明具体是哪一个，也不声称上游GPU kernel bug或最新版本状态**。20个post-hoc向量单独归档，未更改输入规则/阈值或隐藏拒绝。

## 5. 全输出参考与学习边界

第二十一轮后续已补[同实际QKV、GQA完整准入与typed-score逐阶段归因](mlx-sdpa-gqa-fallback-and-score-precision.md)，以及[wide GEMV/head控制](mlx-small-m-wide-gemv-and-head-attribution.md)。后续证据支持本例GPU fallback图的低精度score边界解释，不是vector shader bug证明；本页第二十轮历史capsule/门槛与“不作具体归因”的原记录不回写。

只用独立BF16 reader及固定完整FP64-reduction/FP32-stage oracle，不是MLX CPU模型。所有请求的末行完整词表保存，不检查未请求的prefill早期head行，也不把BF16 hardware arithmetic与该参考混称同一数学实现。

完整BF16模型现在确实已运行，但这不是代表性任务质量/长context测试；255–259只覆盖本次分配边界。正常API logprobs另有52行，不能拿它们直接过raw-logit预算，详见交接页。本轮重读/保存source/PR身份与负结果，未给任一路径发布性能资格。

CPU合同/tests检查所有保存的状态、padding、完整tensor/词表见证与失效边界，普通运行不导入MLX/模型。大实验需显式supervisor、已有固定资产、独立新目录，GPU串行；不改变旧sealed客户端或参考仓库。

复查条件：source/wheel/native SHA、MLX版本、OS/SoC、dtype、padding/mask/RoPE、model weights、allocation history、batch成员及consumer变化。下一步优先分解BF16小物理长度的算术敏感点、更多真实请求/长context及可信device profiling；不把padding改善的一个case晋升成通用技巧或默认。
