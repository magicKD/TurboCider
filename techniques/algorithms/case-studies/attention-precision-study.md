# QK量化×稀疏消融：真实输入、条件误差与不可相加的预算

<!-- knowledge
id: technique-algorithms-case-studies-attention-precision-study
type: case-study
stage: prefill, dit
operator: attention, quantization, layout
backend: cpu-reference, metal, mlx
device: cpu, gpu
symptom: precision-loss, measurement, memory-bound
semantics: approximate, numerically-equivalent
representation: bf16, fp32, q8, q4
boundary: producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [Sage/Sparge](../methods/quantized-attention.md) · [入口合同](sparge-call-contracts.md) · [前轮路由复用](sparse-route-reuse.md) · [实验](../../examples/README.md)

2026-10-16第五轮。目标是区分量化、连接选择和重新路由的误差，而非用fake-quant宣称INT8/INT4硬件加速。来源验证的两seed×两phase，共4224个目标行；没有GPU工作、新模型加载或媒体质量资格。

## 四条路径和一个反事实对照

| 路径 | Q/K数值 | plan | 用途 |
| --- | --- | --- | --- |
| A | 原BF16值转FP64 | dense | 数值分母，另与原BF16 reference核对 |
| B | 重建的量化值 | dense | 单独的QK量化误差 |
| C | 原值 | 原值构建的固定稀疏plan | 单独的连接误差 |
| D | 重建的量化值 | 与C完全相同 | 组合量化与稀疏，不混入重路由 |
| E | 重建的量化值 | 用重建Q/K重新选 | 反事实移植策略；不是固定源Sparge的fused selector |

qwen_precision_probe.py为原创NumPy分析，不执行Sage/Sparge源kernel。量化在FP32参考中分组计算scale/codes；重建后QK、softmax及原始V/PV都在FP64。**没有模拟FP8 V、P量化、低精度累加、tensor-core packing、log2融合或输出cast**，所以不验证CUDA逐位等价，也不测延迟。

输入仍是：layer0、step0 prefill/step1 steady、seed42/43。每head抽33个视觉query，全部32 heads与1048有效keys。text-query不入样本，不代表已替换完整text-causal attention；不是新增中晚层/长视频数据。

## plan和量化配方的精确定义

Q按128-token、K按64-token计算FP64均值；每个pooled Q选top4 K blocks，**随后**加exact block0 sink，实际为4或5块。sink保护24个text和40个visual keys。尾块按24个有效key计；本次token density约0.206–0.305，不是固定4/17，更不是实测FLOPs。

此selector省略similarity fallback，保留数学上清楚的对照；不使用前轮单行q路由、Sol tau或旧mask。因此不能把两篇不同selector的曲线相减，当成量化单项影响。

| 配方 | Q分组 / K分组 | K共享中心化 | 舍入 / scale保护 |
| --- | --- | --- | --- |
| Q8A raw | 128 / 64 | 否 | half-away参考 / amax÷127+1e−7 |
| Q8A center（主配方） | 128 / 64 | FP32共享均值 | 同上 |
| Q8A group32 | 32 / 64 | 是 | 同上；不是原CUDA warp布局复现 |
| Q8R center | 128 / 64 | 是 | RNE参考 / max(amax,1e−7)÷127 |
| Q4R center压力对照 | 128 / 64 | 是 | RNE参考 / max(amax,1e−7)÷7 |

Q8A/R借鉴了所读[Triton/CUDA差异](../methods/quantized-attention.md)，但FP32除法/reciprocal、输入cast、mean dtype和低精度内核舍入仍可能不同。Q4R是独立对称整数压力对照，**不是SageAttention3/NVFP4**。所有codes在probe中均存INT8；JSON分别记录实际code bytes与“若packed时”的理论payload bits，没有交付packed Q4压缩或INT4算术。

## 结果：数值误差和连接误差要分开看

以下为四个点各自统计量的范围，不是置信区间；均为行向量相对L2，不能换成图片损失百分比。

| 配方 | dense B误差 median / p95 | 固定plan组合D误差 p95 / max |
| --- | --- | --- |
| Q8A raw | 0.00052–0.00174 / 0.0361–0.0404 | 0.8303–0.9063 / 1.4184–1.6094 |
| Q8A center | 0.00053–0.00145 / 0.0370–0.0395 | 0.8301–0.9030 / 1.4159–1.6175 |
| Q8A group32 | 0.00055–0.00149 / 0.0346–0.0381 | 0.8284–0.9041 / 1.4175–1.6080 |
| Q8R center | 0.00053–0.00145 / 0.0370–0.0395 | 0.8301–0.9032 / 1.4159–1.6175 |
| Q4R center压力对照 | 0.0144–0.0312 / 0.6610–0.7499 | 1.0129–1.1210 / 2.2078–3.9957 |

单独稀疏C的p95为0.8309–0.9035，说明本次大部分长尾来自所选连接而不只是Q8量化。不能由此推论所有模型的Q8都合格；V/PV还没有降精度，当前参考省略了多种实际误差来源。

K smoothing和细分Q group并没有逐行/逐point保证更准。例如seed42/prefill的dense Q8A raw p95=0.03810，center=0.03955；center的最大行误差0.16654，group32反而0.19055。共享K平移在实数attention下等价，量化后的网格与误差传播不是同一个命题。

## 中位Jaccard=1仍有路由变化

主Q8A center的E与C在seed42 prefill/steady、seed43 prefill/steady分别有91/108/135/86个样本行的plan不同，尽管Jaccard中位数都为1。它们来自每query block共享的plan，不是独立随机样本。Q4R的Jaccard中位数为0.6，不能假定低比特移植只改变数值而不改变连接。

固定源Sparge fused helper先从未量化的pool/sim结果构造路由，同时生成量化codes；把selector改为消费重建Q/K是E这个**新策略**。需重新计selector与模型误差，不从原项目继承资格。

## 独立误差预算不能直接相加

定义`eB=||B−A||/||A||`、`eC=||C−A||/||A||`、`eD=||D−A||/||A||`。本次主配方有7/4224行满足`eD > eB+eC+1e−12`，最大超出约0.01892。1e−12只用于排除比较舍入，不是媒体质量门槛。

这不违反三角不等式：有效的对应界是`eD ≤ eC + ||D−C||/||A||`，而不是把dense量化误差eB当作量化在稀疏条件下的误差。JSON保留了条件项及逐行norm，整理器独立检查这条正确的三角关系；E换了plan后还需另测，不能由mask重叠率给输出误差背书。

参考输出norm本次约0.528–15.393；保留绝对delta norm和相对误差，避免只看归一化数字。它们仍不是RGB、时序或下游模型放大后的指标。

## K中心化与similarity gate是另一层

所读pool/sim逻辑对单位化行求平均Gram；精确算术下等于`||mean(unit_rows)||²`。共享K平移不改变未量化attention，但会改变单位化方向和这个gate。原创标量反例可使gate翻转；这不是本次真实输入已出现该现象的声明。

本次K profile在−0.1/0.1/0.6三个门槛，中心化前后flip均为0；score范围约0.0038–0.126。按该FP64 profile与所读fallback逻辑，门槛0.6会把所有K块标作保守full，未实测CUDA。默认topk入口−0.1与这个设置的实际省算条件相差很大，不能只比较同名`sparse`开关。

## 证据与迁移边界

保留两份完整raw report、来源/输入身份、五配方摘要和主配方所有head/row标量、plans及存储范围。curate_precision.py仅用标准库重算统计、valid-token density、Jaccard、相对误差/norm、条件三角关系与7个超界行；另有原BF16、理想K平移和独立math.fsum见证。

NumPy参考只遍历有效token，不测试原Triton masked-load、padding、实际MMA或CUDA尾块。它没有模型权重、原始tensor或媒体，也不宣称补齐了源项目所有语义；[入口合同](sparge-call-contracts.md)的桩输出更不能进入本数值矩阵。

Apple下一步应先实现强BF16/FP16基线，再独立验证量化变换与同一plan，最后测完整quantize/summary/LUT/layout/attention消费者。Q4若仍先解成浮点再调用SDPA，不能报为INT4硬件加速；减少code bits也不保证host-ready更快。完整质量预算、低精度V/PV及实际Metal kernel仍需继续研究。

默认tests不会读activation、执行上游body或GPU；通过表示已发布证据的合同自洽，不表示近似全部匹配dense或生成画质通过。

## 第五轮交付复核

算法专项66项通过，其中17项为第五轮新增；覆盖opt-in/新输出保护、AST源digest与提取包络、参数/CDF负例、Q4未packed范围、全部主配方row norm/plan/统计、7个独立预算反例以及篡改证据拒绝。标量cosine参考另用缩放避免大/小有限值归一化溢出，不把这一参考实现称作源kernel修复。
