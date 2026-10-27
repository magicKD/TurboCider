# MLX SDPA：GQA准入、GPU fallback图与softmax之前的精度边界

<!-- knowledge
id: technique-metal-case-studies-mlx-sdpa-gqa-fallback-and-score-precision
type: case-study
stage: prefill, decode
operator: attention, gemm, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, measurement
semantics: approximate, contract-only
representation: bf16, fp16, fp32
boundary: producer-consumer, cpu-gpu, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [此前完整模型/padding现象](mlx-lm-ragged-cache-and-padding.md) · [small-M/head排查](mlx-small-m-wide-gemv-and-head-attribution.md) · [固定来源](../sources.md#mlx-bf16-attribution-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十一轮。相同7个真实token，仅把物理长度从7/8改成9，此前完整BF16 Qwen的KL从约0.126降到0.00135。本轮通过原模型阶段捕获、相同实际QKV重放、原生force-fused拒绝、独立逐阶段舍入和attention精度岛，定位到一个足以重现主要差异的机制：**GQA guard令7/8走未融合GPU图；QK score先存回低精度，再做precise softmax。不是已证实的vector shader算错。**

范围固定于MLX0.32.3/tag `v0.32.3`的源码`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`、原MLX-LM `93e6d4a6447b3080861985947f0820e76fcb957d`、M4 Max、macOS26.6.2/build25G83。原24层模型、290个实际参数的完整哈希全部匹配同一官方BF16文件。FP16仅是本轮primitive控制，不是完整FP16模型结果；不声称最新版本/其他SoC也如此。

## 1. 先读完整准入，而非只看query≤8

[Metal `has_fused_kernel`/`use_fallback`](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/scaled_dot_product_attention.cpp)先检查GPU stream、训练/LSE、head维度、mask及非方causal约束；vector候选另外要求`query_length * GQA_factor <= 32`。本例Dq=Dv=64、inference、无sink，Q14/KV2所以GQA=7：

| 物理query长度 | query×GQA | 此选定合同下的路径 | 原生force_fused |
| --- | --- | --- | --- |
| 1或4，KV长度7 | 7或28 | vector fused可用 | 接受；与默认完整输出相同 |
| 5或7，KV长度7 | 35或49 | 未融合图 | 拒绝，明确报query×GQA上限 |
| pad1后8，KV长度8 | 56 | 未融合图 | 同样拒绝 |
| pad2后9，KV长度9 | 63 | full fused可用 | 接受；与默认完整输出相同 |

这不是“9×7突破硬件极限”：9走的是另一族实现，32是此软件vector路径的准入规则。不同D、sink、训练、mask、NAX环境变量等仍需读各自完整guard；CPU路由例仅复现本研究D64 inference子域，不是通用selector或ISA证明。

[fallback构图](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/fast.cpp)仍在原GPU stream上使用MLX算子，不是CPU fallback，也不是只有fused才用GPU。将QKV升FP32不会改变这里的query×GQA guard，不能说“FP32岛强制走fusion”。

## 2. precise softmax不能修复已经舍入的score

此fallback依次做：输入dtype的scale×Q → GQA reshape/broadcast → QK matmul → mask → `softmax(... precise=True)` → PV matmul → 恢复head形状。低精度QK的**输出tensor**仍低精度；softmax以FP32计算，并不把其输入或返回tensor永久升级为FP32。类型规则见[`ops.cpp`](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/ops.cpp)与[Metal softmax](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/softmax.cpp)。不能由输出dtype反推出所有内部乘加的物理指令。

最小CPU反例：scores=`[1024.25,1024.75]`，真实softmax第一项约0.37754；两项先按BF16 RNE落盘都变1024，之后即使用FP64 softmax也只能得0.5。先减最大值再舍入则是`[-0.5,0]`，保留了差异。失效的是“公共平移与低精度量化可交换”假设，不是真实softmax的平移不变性。FP32 reduction/exp不等于score仍保留原始差异。

本例scale=1/8可精确表示，不能把这个机制简单等同scale常量不精确；QK大公共偏置下的低精度score间距本身就可能损失相对差异。GQA共享KV也没有要求提前真实复制7倍KV，原图使用额外维度广播。

## 3. 三层参考与完整证据

不能用“某一参考通过”覆盖另一个目标失败。本轮保留三种独立域：

| 参考域 | 固定什么 | 能回答什么 |
| --- | --- | --- |
| 原完整模型目标 | 原始BF16权重值、24层独立FP64 reduction/FP32 stage数学 | 是否满足原来的全词表预算 |
| 实际输入、仅末端低dtype舍入 | 原GPU真实QKV/hidden，独立FP64算子、FP32中转，再按输出dtype RNE | 当前阶段与该选定数学参考差多少 |
| 实际输入、显式typed边界 | scale/QK输出/softmax输出/PV逐阶段舍入 | 是否符合低dtype图的精度策略 |

证据字段中的single-round指只在算子末端加入低dtype边界；实际CPU实现先转FP32再做BF16 RNE，不是从任意FP64值直接BF16舍入且绝无双舍入的承诺。FP16 primitive另按NumPy FP16转换；原模型参考和阈值都未因这层条件参考而更换。

：四个进程BF16/FP32×17/29，两顺序、6个padding、17阶段，共408个完整tensor，204个重复身份组全部逐位相同。只保存有效7个query的全部元素/完整151936维head，不抽样、不要求未请求的早期head行。BF16第一层native SDPA在物理7/8对单次舍入参考relative-L2约0.14236，9/14/16/255约5.62e-5；相同输入FP32 SDPA控制全通过，contiguous不能消除差异。投影/RoPE条件参考通过，不把它们当未舍入原模型的中间gold。

：42个完整primitive输出中，不含投影或LM head，BF16和FP16在7/8分别约0.14236/0.017734 relative-L2，9以后通过；FP32全部通过。bool/causal mask的无padding控制一致；每种情况下第一个query只有一个合法key，输出都精确等于对应V。支持注意力精度路径解释，但没有覆盖全空mask/任意stride/长context。

：两进程、三dtype×6形状，共36case/342完整输入及中间输出数组：

- 18个原生拒绝与选定完整GQA guard一致，默认输出与显式GPU图18/18逐字节一致，含FP32控制。
- 全部24个BF16/FP16 case的score、probability及显式PV输出，与独立逐阶段舍入参考**完全一致**。这也包括fused可用时仍显式构建的图，不能把其错误预算归为shader算错。
- 12个低dtype fallback case对单次舍入参考全拒绝、对逐阶段参考全精确；只在低dtype QK之后提升FP32仍12/12拒绝，在QK之前提升则12/12通过。全36个early-FP32控制均通过。
- 其余12个低dtype fused默认输出通过单次舍入参考，却不匹配低dtype显式图；6个FP32 fused默认同force输出逐位相同。所有dtype共18个fused接受控制均相同，不等于观测到了每条ISA。

因此本例没有“默认路径违反自己逐阶段数学”的证据；有的是同API随shape切换存储/舍入边界，导致相对更高精度目标的差距明显变化。重复性与原目标通过仍是两件事。

## 4. 原完整模型中的因果干预，仍不是质量修复

仅在owned child临时包裹Qwen模块实际使用的attention函数，try/finally恢复。weights和KV保留BF16，原290参数逐一验证；baseline/首层attention FP32/全部24层attention FP32/restored，4种padding×两重复，32个完整词表向量。

| 同7-token、pad0 | 原数学参考KL | 原raw门槛 |
| --- | --- | --- |
| 原baseline | 0.12606905 | 拒绝 |
| 只第一层attention提升 | 0.00415915 | 拒绝 |
| 全24层attention提升 | 0.00257063 | 拒绝 |
| 恢复原函数 | 0.12606905，字节也恢复 | 拒绝 |

pad1的全层岛KL约0.00204968；pad2/9时三策略输出与baseline相同。32/32仍拒绝原严格`.005+.005*abs(ref)`且relative-L2≤.005预算。单次舍入算子参考通过、KL下降或top1相同都不替代该门槛，也不是代表性任务质量。岛包含cast/临时内存与不同调度，未做性能采用比较；不是修改生产默认/上游源码。

## 5. 上游PR与身份边界

[#3119](https://github.com/ml-explore/mlx/pull/3119)于2026-02-10合并，修的是非NAX full fused的scale应用：移除低dtype Q预缩放，将scale应用在FP32 score累加结果。它展示相同类型的精度边界风险，却不是这次GQA fallback图的修复证明；不能因PR标题有precision就当作本例根因。vector scale原已FP32，不能只凭query短就套该shader。

本轮不仅记录Python版本，还枚举实际dyld加载的core/libmlx/libjaccl镜像SHA与安装的metallib身份。core=`963a14ee…dad5ef`、libmlx=`ebc8a5f5…110a1e`；完整值在证据。metallib清单不证明所有pipeline都从它加载；源码tag/本地文件SHA也不等于已可复现wheel build或物理dispatch/ISA取证。

发布器仅去除12个镜像的绝对个人路径，保留name/SHA/全部指标；`publication.raw_capsule_sha256`指向未修改原件，island原`main_evidence_sha256`仍绑定raw，portable交叉引用另在`publication.published_main_evidence_sha256`。不为了脱敏悄悄重写旧执行身份。raw tensor、模型和原始checkpoint不分发。

## 6. 学习与复查

第二十四轮另有[实际公开API编码/PSO名字关联](mlx-dispatch-pso-and-function-association.md)与[7/9/32+decode路由](mlx-attention-routes-and-wide-gemv-observed.md)，补上选定wheel的实际函数族记录。它不改变本页历史同QKV舍入参考，也不构成ISA或某条shader故障证明。

第二十二轮已直接执行[原generate_step的prefill分块](mlx-prefill-chunking-and-numerical-routes.md)及[KV延续/lookahead控制](mlx-prompt-cache-provenance-and-lookahead.md)：7-token小块更近参考，32-token却非越小越近，且部分第0层KV已有shape相关差异。该后续不回写本页第二十一轮capsule、阈值或当时未执行的边界。

CPU合同和标准库tests不导入MLX/模型；保存的capsule核查不是离线重跑所有原始数学。重算完整数组需显式提供ignored原件/已有NumPy，仍不执行GPU；native客户端必须显式supervisor、新目录和串行GPU准入。

复查条件：source/wheel/native SHA、SoC/OS、dtype、query/KV/head/GQA、mask、training/sinks、scale、位置或消费者改变。下一步可测试分块query是否改变准入及完整质量/成本；**本轮没有执行4+3整模型prefill，也没有把padding或分块推荐成性能技巧**。代表性任务、长context、实际device trace、M5/NAX及跨引擎对比仍需独立研究。
