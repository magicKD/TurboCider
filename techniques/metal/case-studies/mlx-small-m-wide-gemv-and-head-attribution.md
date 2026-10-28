# MLX small-M wide GEMV：多向量复用权重，以及怎样排除词表head误归因

<!-- knowledge
id: technique-metal-case-studies-mlx-small-m-wide-gemv-and-head-attribution
type: case-study
stage: prefill, decode, ffn
operator: gemm, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: memory-bound, precision-loss, measurement
semantics: approximate, contract-only
representation: bf16, fp16, fp32
boundary: weight-runtime, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [GEMM/量化数据流](../gemm-and-quantization.md) · [GQA/score精度](mlx-sdpa-gqa-fallback-and-score-precision.md) · [来源](../sources.md#mlx-bf16-attribution-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十一轮。`Y[M,N]=X[M,K]W[N,K]ᵀ`里，M不是简单的“batch size”：它是进入某次matmul的物理向量数。单请求短prefill、批处理、padding、只请求最后一行都可能改变它。MLX的wide GEMV在少量行时把同一权重块用于多个向量，避免宽GEMM tile大量空算；它不是量化、不是ANE，也不是M5专属矩阵指令。

本轮既读固定源码，又对真实Qwen最后hidden/head做替换控制。结论分开：**多向量权重复用是可信源码机制；“它造成7-token整模型KL≈0.126”不受本轮控制支持。** 没有测wide开关的严格性能A/B，不发布收益倍率。

## 1. 为什么小M值得单独做一族kernel

逐行GEMV容易反复流过整份W；普通GEMM则可能为很少M补出很多无用行。wide的折中是每个threadgroup流入一小块W，在寄存器里同时维护几个输入向量的累加和，再沿K做SIMD shuffle reduction。权重载入一次服务多个向量，提高局部复用；向量数再增大则会挤占寄存器、降低occupancy，不能无限加宽。

读[固定`gemv.h`的GemvWide](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.h)：相邻lane读取相邻vec4权重，8级unroll保持载入在途，局部权重再供多个X；尾向量/行的读被夹到合法地址，写另有guard。k_lanes决定K reduction参与lane数，剩余lane可承载不同输出行，不等同“线程越多越快”。这段是dense FP16/BF16，不能套作packed Q4原生消费。

[PR #3888](https://github.com/ml-explore/mlx/pull/3888)在2026-07-22合并。作者的其他shape/设备/热cache表是上游结果，不是本机Qwen收益；本页不复制该表，也不按减少访存的公式声称实际DRAM下降。`AccT`源码类型不替代binary ISA/counter证明。

## 2. host配置规则：寄存器、并行网格与vocab宽度

[`matmul.cpp`的gemv_wide_config及调用点](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)要求arch generation≥15，代码注释对应M3及以后；旧代load issue瓶颈保留原路线。还要求M>1、N>1、K整除4、输出FP16/BF16及offset/leading/batch stride对齐。具体offset按字节8对齐，leading/batch stride按元素4对齐，不要用“contiguous”一个词抹掉所有条件。

每个register tile最多5个向量；`passes=ceil(M/5)`，超过3就拒绝wide，因此该配置域是M2..15。实际`vecs_per_tg=ceil(M/passes)`。若只有1个pass或N≤64，k_lanes=32，否则16；N≥65536则grid_x=1，让向量chunk在同threadgroup里轮转，否则grid_x=passes。

| M、N、K、dtype/条件 | passes / vecs_per_tg | k_lanes / grid_x | 说明 |
| --- | --- | --- | --- |
| 7、151936、896、BF16 | 2 / 4 | 16 / 1 | 词表head，后一个chunk尾部guard |
| 9、151936、896、BF16 | 2 / 5 | 16 / 1 | M变更可改变register tile |
| 7、896、896、BF16 | 2 / 4 | 16 / 2 | 不够vocab宽，分成网格列 |
| 1、151936、896、BF16 | 不适用 | 不适用 | 无多向量复用，沿原单向量路线 |
| M≥16、FP32、K尾部或不对齐 | 配置拒绝 | 不适用 | 返回调用方选择其他实现 |

grid_x=1不等于整个GPU只有一个threadgroup：N方向依然有大量输出行组。轮转chunk期望利用cache局部性，不保证整矩阵驻留L2或只从DRAM读一次。M≥16只是此wide拒绝，之后要读外层matmul完整选择，不能将helper泛化成整个后端只剩单一路线。

CPU配方显式检查上述子域；它不检查真实array的全部layout、调用方transpose、addmm/bias、indexed gather和实际PSO。源码预测不能冒充trace观测，也不能仅凭shape认定安装wheel执行了哪个kernel。

## 3. 实际hidden固定后，head会改变什么

主实验运行原始24层Qwen同一7-token prefix；padding=`[0,1,2,7,9,248]`形成物理长度`[7,8,9,14,16,255]`，BF16/FP32各两进程。最后norm后的真实7×896 hidden完整保存，然后比较：

1. 原模型按最后输出需求返回的logits。
2. 原始完整物理行head，取最后有效行。
3. **先取最后hidden，再做M=1 head**。
4. 相同已舍入hidden/原权重先提升FP32，再head并转回原输出dtype。

四种都保存完整151936输出，参考使用同一个实际hidden最后行和原BF16权重的FP64点积、FP32中转及最终BF16 RNE（FP32路线不再做BF16转换）。文中“单次输出舍入”指不在QK/softmax间插低dtype边界，不承诺从FP64直接舍入为BF16且无双舍入。这个条件参考不重新修复上游层的误差，不等于原完整模型参考。

| head条件参考 | BF16，12份 | FP32，12份 |
| --- | --- | --- |
| full physical rows | 8通过、4严格点值拒绝；最大relative-L2≈4.57e-5 | 全通过 |
| 先取末行、M=1 | 全通过；最大relative-L2≈1.90e-5 | 全通过 |
| FP32 head后再转回输出dtype | 0通过；合计22个点值越线，最大relative-L2≈4.85e-5 | 全通过 |

FP32计算再转BF16也可能在tie邻近落到另一BF16格点，不能凭名称“更高精度”宣告每点通过。小L2不覆盖逐元素预算；12个FP32-head/BF16-output严格拒绝如实保留。

24个dtype/shape重复case的原模型输出与materialized full-row head**全部字节相同**，但full-row与M1的完整输出全都并非逐位相同。说明观察/重排head可改变极小算术结果；不能直接说last-row优化bitwise无损。BF16无padding时M1 head的原模型KL仍约0.12606905，与baseline几乎一样；head提升也没消除此现象。所有BF16 head策略仍拒绝原完整模型门槛。

## 4. 怎样避免把相关route当作原因

padding改变M，同时改变attention query长度和内部shape；只观察某个GEMV阈值对应KL跳变，不足以归因GEMV。这里有更强反证链：

- 固定实际hidden，只改head的M/计算dtype，没有消除主差距。
- 固定实际QKV、完全移除投影和head，仍在物理7/8重现大差异。
- 读完SDPA的query×GQA guard，原生force_fused实际拒绝5/7/8。
- fallback输出与显式typed GPU图、独立逐阶段舍入精确一致；QK前FP32控制才恢复单次舍入参考。

详见[attention机制页](mlx-sdpa-gqa-fallback-and-score-precision.md)。这足以否定“必须由词表head造成”或“query7必走vector shader”两种解释；不是证明wide的所有shape、任意舍入或其他模型永无数值问题。

## 5. 可迁移优化与下一次正确实验

第二十四轮已取得[实际wide/regular/split-K编码与PSO资源属性](mlx-attention-routes-and-wide-gemv-observed.md)，包括词表宽grid候选。其token长度不同、无matched wide开关性能A/B，因此本页的因果/性能边界仍保留；不能以新dispatch数量覆盖旧原目标拒绝。

学习点是用少量寄存器累加摊薄权重流量，按输出宽度决定向量chunk是否留在同组，且把alignment/tail/architecture作为准入合同。代价是更多累加状态、反复流过W及可能降低占用。若消费者只需要最后token，提前选行会少做工作，但会同时丢掉多行W复用，收益必须对实际原图计量，不能仅以FLOPs比例估计。

真正比较需固定完整输入/输出需求、cache冷热/权重驻留、source/native SHA、数值域，显式可控wide消融，加入M1/2/5/6/15/16和N65k边界、非零offset/不对齐负控，再计cast/copy/完整消费者和请求成本。**本轮未做该性能实验**，未修改上游或给生产策略提默认。

复查条件：M/N/K、输出需求、dtype/偏置、stride/offset/alignment、source/wheel/native SHA、SoC/OS、权重驻留/cache和消费者改变。其源码可用范围与真实dispatch/ISA、组件数值、完整质量、性能是独立资格；M5/NAX/ANE及最新上游状态不继承本轮结果。
