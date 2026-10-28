# 方法综述：从冗余来源理解加速取舍

<!-- knowledge
id: technique-algorithms-methods-readme
type: navigation
-->

[学习入口](../README.md) · [实现技巧](../implementation-patterns.md) · [分析案例](../case-studies/README.md) · [固定来源](../sources.md)

这里回答“为什么可能更快、到底省哪段计算、需要哪些前提”。不要按论文名字或作者倍率挑开关；先定位实际瓶颈，再按下列类别读。页面中的本机实验只限定其已测配方，完整复核材料另在[档案](../../MAINTENANCE.md)。

## 先建立分析坐标

从[模型与稀疏性地图](model-sparsity.md)区分dense/joint/cross/linear attention、CFG、模态和采样日程；再读[近似验收原则](approximations.md)。同名cache可以保存条件KV、middle residual、完整feature或velocity，它们的失效条件与误差传播不同。

## 1. 时间冗余：复用、预测与跳过网络

| 方法 / 阅读入口 | 值得学习的机制 | 不应照搬的部分 |
| --- | --- | --- |
| [精确cache](caches.md) | 从依赖关系识别真正不变的encoder/条件KV；先做无损复用 | 把跨步生成KV也当成常量 |
| [TeaCache](teacache.md) · [EasyCache](easycache.md) | 用便宜signal决定是否跳过昂贵body；比较离线拟合与在线增益 | 多项式、threshold跨模型照搬；proxy冒充误差界 |
| [FBCache/DBCache](block-caches.md) · [DiCache](dicache.md) | front/probe → decision → middle reuse → tail；检查继续执行的全部effect | 仅恢复tensor而丢掉ControlNet、全局层号或branch状态 |
| [DeepCache](deepcache.md) | 保留当前浅层路径，复用U-Net深特征；沿拓扑定义跳算 | 把U-Net branch编号直接映射到DiT或不同MLX port |
| [PAB](pab.md) · [T-GATE](tgate-phases-and-branches.md) | 将attention类型、采样阶段与CFG分支作为不同冗余来源 | 只看outer NFE；不计投影/cache读写或CFG-only强分母 |
| [ToCa/DuCa](token-feature-caching.md) | token级fresh/stale更新、age与真实gather/scatter消费者 | 用完整attention-map做昂贵selector却忽略其成本 |
| [TaylorSeer](taylorseer.md) · [Spectrum](spectrum-forecasting.md) · [FoCa/DMD](forecast-calibrators.md) | 缓存旧值之外，用实际坐标与历史拟合外推；定义fallback | index当真实时间、重复读推进history、区间外漂移与fit失败 |
| [VDE](vde-input-adaptive-velocity.md) | 分开输出幅值/方向与输入norm关系；比较replay/linear/VDE | 全batch归约耦合、近零与FP16溢出；同调用数当同质量 |
| [SeaCache/MagCache概览](spectral-magnitude-cache.md) · [MagCache消费者](magcache-calibration-and-consumer.md) | 区分频谱与幅值先验、校准曲线、在线累计预算 | 范数接近当方向接近；幅值预算当生成质量保证 |
| [AdaCache/MoReg](adacache-motion-and-period.md) | 自适应周期与运动调制，区分metric范围与实际跳算范围 | CFG pooling掩盖分支、非finite或最后period档位默认安全 |
| [FasterCache](fastercache.md) | attention时间复用与CFG频谱重构分开建模，再测组合 | 根据局部变量名认定c/uc身份，或把两个节省比例相乘 |
| [Draw Things](draw-things-caching.md) | Apple原生图上的anchor、head共享与显式readback | 控制器接缝成功即完整H3/FLUX/Wan速度或物理共享证明 |

迁移实践先读[技巧1—3、7、10、12—14](../implementation-patterns.md)；完整请求取舍见[Qwen](../case-studies/native-cache-request-study.md)与[DeepCache](../case-studies/deepcache-sd15-mps-quality.md)，不要从preset名称判断质量。

## 2. Attention冗余：少连接、块摘要与学习修正

| 方法 / 阅读入口 | 值得学习的机制 | 首先要验证 |
| --- | --- | --- |
| [局部窗口、STA/VSA](structured-sparsity.md) · [SVG系列](svg.md) | 空间/时间局部性、tile路由、head差异与语义聚类 | 未选QK/PV是否真的不算；重排、selector与模态保护 |
| [Sol-Attn](sol-attention.md) · [PISA](piecewise-attention.md) | 关键块精算，远端用低阶摘要参与正确归一化 | sink、tail、mass、online softmax分母；all-exact对强dense |
| [VMoBA](vmoba.md) | disjoint分支、block选择与LSE合并 | 不平均已归一化输出；空remote/dense fallback与变长布局 |
| [Sparge/Sage](quantized-attention.md) | 低精度QK/PV与稀疏误差分开，随后做组合 | CUDA低比特路径不能当Metal支持；路由输入精度也变了 |
| [SLA/SageSLA](sparse-linear-attention.md) · [SpargeAttention2/SLA2](spargeattention2.md) | sparse主路与可学习linear修正、训练后分布集中 | 全局/补集函数scope、checkpoint语义、bias、native backward/QAT可得性 |
| [DiTFastAttn](attention-redundancy-composition.md) | 空间、时间与条件冗余的依赖关系和刷新成本 | 每种补偿缓存的物理量不同，组合不能机械叠加 |

先看[真实Qwen稀疏性](../case-studies/real-qwen-sparsity.md)，再看[Sol真实组件反例](../case-studies/sol-mps-component-study.md)和[精度×稀疏消融](../case-studies/attention-precision-study.md)。有mask、稀疏率高、retained mass高，都不是实际净收益或逐行误差保证。

## 3. Token与空间冗余：减少消费者的输入规模

- [ToMeSD/token reduction](token-reduction.md)：学习match → merge → 短计算 → restore的全链；保护原坐标、CFG与RoPE。恢复shape不恢复被合并的信息，原默认只合并self-attention。
- [HiDiffusion](hidiffusion-spatial-compute.md)：学习RAU-Net深层空间降算与MSW-MSA窗口，区别少空间工作与少NFE；检查odd shape、skip、patch时钟和恢复。

案例入口：[ToMeSD与Qwen FFN-local](../case-studies/token-merge-consumer-and-real-ffn.md) · [HiDiffusion真实Tiny U-Net](../case-studies/hidiffusion-unet-consumer.md)。前者不是原默认ToMeSD媒体复现，后者不是预训练画质。

## 4. 数值冗余：压缩、低秩与部署内核

- [量化/混合精度](quantization.md)：先区分容量、带宽、驻留和实际低精度计算；fake-quant与packed runtime分开。
- [SVDQuant/低秩](low-rank-quantization.md)：学习outlier分解、校准目标、共享因子和融合成本；W4A16分离分支不是CUDA W4A4融合。

先读[真实Qwen低秩/holdout](../case-studies/real-qwen-lowrank.md)，再读[完整FFN反收益](../case-studies/real-ffn-gpu-study.md)。如果目标是放得下模型，应报告容量收益；如果目标是更快，必须测相同驻留与完整消费者，不能用bit数替代时间。

## 5. 配方与阶段：另外建账的优化

- [少步采样、蒸馏、线性attention](sampling-and-distillation.md)：区分同权重近似、solver改变和student权重；更少NFE并非同一生成函数。
- [Tiny decoder](tiny-autoencoders.md)：区分预览与最终质量，单独看颜色、细节、ABI和视频闪烁。
- [Sana组合源码导读](sana-sol-engine.md)：学习effect依赖、路由与后端分离；设计空间不等于所有组合已测。
- [研究前沿与可得性](research-frontier.md)：只作论文/作者实现的边界记录，不混入当前可执行支持清单。

## 把阅读转成可复用分析

每种方法至少回答五个问题：省掉哪个实际consumer？新增什么controller/数据搬运？哪些状态必须失效？用什么正反控证明实现保持合同？在相同质量预算下完整请求是否更快？具体模板见[评测合同](../evaluation.md)，不要只摘录作者倍率。
