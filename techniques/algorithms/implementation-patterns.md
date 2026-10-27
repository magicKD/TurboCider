# 从仓库抽出的实现技巧：把省算边界、状态事务与后端成本写清楚

<!-- knowledge
id: technique-algorithms-implementation-patterns
type: implementation
stage: dit
operator: attention, sampling, scheduling
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, measurement
semantics: contract-only
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](README.md) · [源码账本](sources.md) · [状态审计](case-studies/forecast-state-audit.md) · [实验](../examples/README.md)

整理截至2026-10-17第26轮。本页把源码阅读、CPU反例、Apple组件和完整请求结果提炼为可复用设计；不是把作者preset转换成通用建议。每条按“机制 → 适用边界 → 来源 → 最小验证”阅读：可迁移的是省算边界、依赖和验证方法，阈值、倍率与质量资格不能一起搬走。

快速选读：接新cache看1—3、10、12；写attention/token优化看4、8—9；做Apple迁移看5、11、13—14；做组合或性能归因看6、15—16。固定commit与文件SHA在各条的来源账本，直接源码入口也汇总在[阅读路线](README.md)。

## 先学一套可重复的分析流程

不必重读所有实验。面对一个新加速方案，先写一张分析卡，按下面五步推进；每步都有一份可以检查、也可能推翻假设的产出。

| 分析步骤 | 应产出的东西 | 可参考的案例 |
| --- | --- | --- |
| 找出实际冗余 | full/hit两条消费者路径，标明哪些body不再执行、哪些仍执行 | [DeepCache祖先/leaf边界](case-studies/deepcache-unet-boundaries.md) |
| 估算净收益而非只算省量 | `净节省 = 省掉的真实计算 − 新增控制、搬运、恢复与同步成本`；列出可能吞掉收益的项目 | [VDE host依赖](case-studies/vde-mps-controller-cost.md) · [完整FFN反收益](case-studies/real-ffn-gpu-study.md) |
| 证明实现合同与失效条件 | no-op等值、一次真实hit、fallback和异常/新请求正反控；写清clock/branch/ownership | [MLX生成事务](case-studies/mlx-teacache-lifecycle.md) · [状态审计](case-studies/forecast-state-audit.md) |
| 用反事实区分解释 | 同输入比较原版、强对照和候选；组合加A/B/A+B；观察器另作校准 | [NNC强读回分母](case-studies/nnc-gate-controller-cost.md) · [FasterCache组合](case-studies/fastercache-consumer-study.md) |
| 同时交付收益与代价 | 实际计时边界、配对统计、独立质量预算、全部拒绝与重开条件 | [DeepCache完整媒体取舍](case-studies/deepcache-sd15-mps-quality.md) · [评测模板](evaluation.md) |

一条假设只能解释自己观察到的边界。例如“每步少做一次host检查”需要同轨迹等值对照；“更少调用”不能单独证明更快；“相对full误差更小”也不能直接证明更符合prompt。这些区分比某次设备上的倍率更适合迁移学习。

## 1. 用被省略对象定义技术，不用名字

| 被省对象 | 当前步仍需计算 | 缓存/预测对象 | 实现风险 |
| --- | --- | --- | --- |
| block middle | front/back与tail | middle residual | prefix/target shape、gate与residual anchor不同 |
| attention branch | timestep gate、FFN/residual | attention output | gate前后边界、CFG branch、mask |
| MLP的旧tokens | 选fresh行、当前norm/modulation | full-size MLP output slots | 真正gather少算而非mask，scatter与token age |
| 远端KV blocks | 当前local/exact attention | 当前summary或历史full-local差 | summary补偿与WA-RS不是同一归一化 |
| 整体feature演化 | prelude、tail、controller | polynomial/modes/history | clock、fit成功与fallback、history ownership |

每项至少写出before/after张量、shape、dtype、原位置、branch identity及实际调用计数。参数名为`teacache`不保证存的总是residual；`sparse`开关也不保证kernel少算。

借鉴方式：先标出“完整路径 / 命中路径”的实际consumer，再为其选择控制器。DeepCache返回旧深层feature、DBCache加middle residual、VDE估计velocity，三者不是可互换的cache payload。来源：[DeepCache/S70](sources.md#s70)、[Cache-DiT/S37](sources.md#s37)、[VDE/S67](sources.md#s67)。最小验证：no-op对stock等值；hit时确认被省body没执行，当前必须更新的浅层/tail仍执行。

## 2. state不能只是几个无版本tensor

推荐拆为：request identity；trajectory/CFG/stage；signal anchor；最近full anchor及其clock；历史owned tensors；拟合generation；本step decision/prediction memo；最后成功commit的step；consecutive reuse budget。

其中signal anchor可能按相邻step推进，也可能只随full更新，取决于算法。rolling预测可以递归推进，但必须定义一次logical step最多推进几次。缓存查询、状态推进、执行full、提交成功结果分成独立动作，比一个会偷偷更新历史的getter容易验证。

取消、异常、半个CFG、非finite或shape变化不提交新history；失败后明确reseed/full。不能从某个项目允许重复index的test推导所有sampler都应允许，也不能强行以index+1代表所有ODE多阶段eval。

来源：[Spectrum/S32](sources.md#s32)、[MLX TeaCache/S42](sources.md#s42)，解读见[状态审计](case-studies/forecast-state-audit.md)与[生成事务](case-studies/mlx-teacache-lifecycle.md)。最小验证：相同对象连续请求对fresh；caller原位修改旧输出；末步前/末步后异常；漏一条CFG完成事件。缓存状态不应被未完成请求污染。

## 3. 正确性oracle应覆盖最弱的fallback

恒定/线性/指数/振荡、非均匀full间隔、相同step重复读、变shape、caller原地写、首份history、fit不足、dtype-cast后overflow，都是便宜且高价值的控制。本轮反例说明“主算法公式正确”不足以保护历史更新与fallback。

DMD fit抛异常而reuse、VMoBA无remote而dense fallback、TeaCache零命中，均应有actual counters。若只测输出是否finite，算法未engage仍会pass；若只测速度，弱分母或冷暖差也会伪装成收益。

来源：[FoCa/DMD/S33](sources.md#s33)、[VMoBA/S35](sources.md#s35)；[CPU状态反例](case-studies/forecast-state-audit.md)可作为低成本测试模板。最小验证同时要求输出、full/hit/fallback计数和最后成功anchor；注入故障后必须能回到已声明的full/reseed路径，不仅是“不崩溃”。

## 4. 用分子/分母合并attention，不平均输出

VMoBA的disjoint分支用LSE权重，Sol/PISA的summary进入online softmax，DiTFastAttn缓存的是已归一化full-window输出差。三者都可称“补偿”，但可组合的物理量不同。先写明是否未归一化、是否包括token mass、exact/approx是否互斥，之后再谈fusion。

多个模态的indicator也一样：拼接平均、每模态ratio求和、取max会产生不同gate。需要最差模态保护的音视频任务，不应让视频token数压低音频变化的权重。

来源：[VMoBA/S35](sources.md#s35)、[Sol/S01](sources.md#s01)、[DiTFastAttn/S34](sources.md#s34)。最小验证：两个分支logit尺度极不相同、空remote、重复sink、tail block；对全dense参考核对归一化和有效token集合。练习中的LSE正控与错误平均反例见[算法实验](../examples/README.md)。

## 5. Apple 低成本优先级是成本模型，不是经验神谕

优先使用已有优化GEMM/SDPA，只绕过昂贵调用；进一步再减少索引/summary/launch/完整cache读写。为了一个便宜selector而物化N²attention-map，或者为了P×P小求解器回读K×F大history，往往破坏原来高性能路径。

CPU可以处理小型clock/plan/Gram求解，GPU持有大feature，ANE可尝试固定full/predict/bucket图；必须比较真实consumer-ready路径。动态token bucket、cast、pad、LoRA和参数共享都属于边界成本，不能从kernel microbench中隐去。

本研究中的反证：[真实FFN](case-studies/real-ffn-gpu-study.md)的packed/低秩全部慢于强dense；[NNC gate](case-studies/nnc-gate-controller-cost.md)对合并读回强分母没有稳定通用优势。固定实现见[MLX/S44](sources.md#s44)、[NNC/S56—57](sources.md#s56)。最小验证：分别计setup、完整refresh/hit周期、consumer-ready和完整请求；逻辑nbytes只记容量，不冒充峰值。这里的ANE固定图只是研究方向，不是已取得的设备结果。

## 6. 组合按依赖重新验收

Sparse/PTQ改变signal或历史feature分布，forecast又改变下一步稀疏路由；token更新影响staleness和GEMM shape；full/predict切图还可能破坏buffer donation。需要A、B、A+B三组实际trace和holdout，不乘倍率、不把独立误差预算相加。

对应原创CPU练习：`algorithm-design-round2` 展示fresh行实际调用、LSE合并与错误平均、WA-RS在远端变化时失败、多模态平均掩盖小流、Chebyshev区间外增长；范围见[实验室](../examples/README.md)。

来源：[Sana effect组合/S01](sources.md#s01)、[DiTFastAttn/S34](sources.md#s34)、[FasterCache/S46](sources.md#s46)。最小验证：baseline、A、B、A+B共享输入/日程并保存实际执行trace；[FasterCache消费者](case-studies/fastercache-consumer-study.md)的attention-only/CFG-only/组合计数56/74/57说明组合不是独立概率相乘。

## 7. probe恢复必须沿用完整依赖，不只是复用tensor

[DiCache](methods/dicache.md)的微型真实forward说明：能从probe继续且每层只跑一次，仍不保证full语义；ControlNet residual、全局层编号与任何被probe跳过的effect都要恢复。浅层相似度决定是否省算，多历史预测决定省略段的值，两个合同必须各自检查；clip不是NaN防护。

来源：[DiCache/S38](sources.md#s38)。借鉴方式：让probe输出显式continuation状态，而不是只返回tensor/局部index。最小验证：强制每步full时对未patch原路逐层等值；再加入非零ControlNet residual、partial probe、NaN signal和不同CFG。不要用全零辅助分支掩盖缺失effect。

## 8. 把路由缓存做成不可变版本化plan

[通用policy接缝](case-studies/sparse-route-reuse.md)在hit仍计算score，而且mask与indices可以在caller修改后不同步。先明确被缓存的是mask、indices、labels还是summaries，之后将shape之外的route预算、token坐标、sink、CFG和策略版本加入身份。writer完整生成后提交一代，读者固定到该代，不从`.to(same device/dtype)`推导独立副本。

新case `route-reuse-contracts`证明：token均值不变也能使重要连接翻转；重复padding indices若不被consumer屏蔽，会改变softmax分母。这是便宜的拒绝witness，不代表任何论文proxy总会失败。真实两步分析还要求保留每head/row的p95与最大值，不能只以mask Jaccard或中位误差挑策略。

来源：[Sana路由/S01](sources.md#s01)，解读见[路由plan复用](case-studies/sparse-route-reuse.md)。最小验证：hit后修改caller持有的mask、budget/CFG/token顺序变更、tail padding重复项；记录selector是否仍运行。复用mask却仍做完整score时，不能宣称省掉selector成本。

## 9. 参数shape兼容不是算法兼容

[SLA两个固定版本](methods/sparse-linear-attention.md)的P/b能strict-load同一state，却将linear归一化分别放在全局KV和未选补集上。checkpoint身份需绑定函数scope、feature map、epsilon、bias与路由；只固定tensor keys和D×D shape不够。all-selected控制还必须关闭global修正和bias，零P也不能消除先产生的NaN。

初始化的梯度检查要区分路径：P=0可让linear feature路径为0，同时P/b和稀疏QKV路径仍有梯度；CPU autograd替身的成功不验证原custom backward或QAT。原创标量case另证明较高retained mass只收紧一个界，不保证每行实际输出误差更低。

来源：[SLA全局修正/S40](sources.md#s40)与[补集修正/S41](sources.md#s41)。最小验证：same-state双函数对照、all-selected、空选集、零P/非零bias以及各路径gradient。加载成功只验收结构合同，不验收算法、训练或媒体。

## 10. generation提交要验证消费者完成，不只验证控制器计数

[MLX TeaCache接缝](case-studies/mlx-teacache-lifecycle.md)把after-loop staging与自然返回commit分开；完整loop后另一个callback失败仍不提交。CFG一条decision可驱动两次branch调用，但decision长度不证明两条tail都完成，最后漏negative的构造包络就是反例。将request identity、logical step、branch、image shape和完成状态绑定，再提交public统计与可复用状态；不把自然返回或finite当作正确消费者证明。

这个反例故意违反正常orchestrator的两次调用前提，不是声称上游常规请求正在漏调用。`release_arrays()`同样只表明引用被丢弃，不表明allocator、物理backing或异步消费者已经完成；source/probe结论必须停在实际观察边界。

来源：[MLX TeaCache/S42](sources.md#s42)。借鉴方式：`begin → stage → consumer完成 → commit`；任一异常走abort，staging不能对外成为新一代cache。最小验证：loop完整但导出callback失败、最后一条branch未完成、取消后新输入重试；同时核对public统计和实际tail输出。

## 11. 校准目标、分解目标与部署精度必须分别命名

[真实Qwen低秩研究](case-studies/real-qwen-lowrank.md)把SVD(W)、SVD(W−Q(W))、Ws对角加权和多轮输出目标分开。相同rank并不代表相同数学或校准工作；FP64参考也不等于原BF16执行语义。固定一份fit数据，freeze bank后再看后续timestep/独立prompt，不以heldout重新挑rank后继续叫独立确认。

多个量化矩阵串成非线性网络时，独立消融与组合都要测；GU/Down的误差还有cross term，不直接相加独立relative L2预算。来源byte/hash审计、native-vs-same-formula数值gate、对原函数近似误差和最终RGB/视频仍是不同证据轴。

来源：[DeepCompressor/S43](sources.md#s43)、[Nunchaku/S17](sources.md#s17)、[MLX部署/S44](sources.md#s44)。最小验证：冻结fit/holdout数据后比较rank0与不同rank、单投影与完整FFN、eager与compiled强dense；同时报告误差和总时间。W4A16+分离低秩不能沿用原CUDA W4A4融合的性能声明。

## 12. 分支语义身份与batch位置分开保存

`conditional/unconditional`、prefix/target、image/text/audio是语义；batch0/1、concat顺序和chunk位置只是当前表示。FasterCache所读CogVideoX实际保留uc并重构c，不能根据变量名猜；ToMeSD的CFG两路共享src/dst网格，也不保证逐batch匹配图相同。

借鉴方式：为缓存键显式记录request、logical step、call/substep、semantic branch、modality及原token坐标；进入具体adapter时再解析batch位置。来源：[FasterCache/S46](sources.md#s46)、[ToMeSD/S48](sources.md#s48)、[T-GATE/S62—65](sources.md#s62)；解读见[CFG消费者](case-studies/fastercache-consumer-study.md)、[阶段/分支](methods/tgate-phases-and-branches.md)。

最小验证：反序batch、unbatched CFG、no-CFG、phase等号边界、branch缺失；检查guidance对误差的放大系数。不要拿当前payload的一个slot给另一条语义分支的cache背书。

## 13. 把可证明的控制条件搬到setup，而非每步回读

VDE原predictor检查设备时间差后用`.item()`驱动Python；DeepCache原root回读t，再在时间列表查首次相等位置。固定唯一日程且调用合同已知时，可以在setup验证两锚点/唯一时间和phase，再在循环使用显式时钟，避免重复host依赖。这是减少同一近似轨迹的实现成本，不是增加跳步或改善质量。

来源：[VDE原实现/S67](sources.md#s67)、[MPS标量路径/S69](sources.md#s69)、[DeepCache/S70](sources.md#s70)。本库受限变体及byte等值结果见[VDE成本](case-studies/vde-mps-controller-cost.md)与[DeepCache ordinal](case-studies/deepcache-sd15-mps-quality.md)；两者倍率不同，不推导通用scalar成本。

最小验证：原版与变体逐步clock/refresh、端点bytes和请求重置；再用重复时间`[9,6,6,3]`、partial-start、额外denoiser调用、动态callback作拒绝控制。DeepCache原value-first时钟在该列表为`[0,1,1,3]`，不能无条件替换为ordinal。证明前提失效时恢复guard或明确拒绝，而非继续复用旧plan。

## 14. 利用线性把hit工作搬到refresh，但仍检查舍入

FasterCache固定频带、各band统一scalar的配方中，可将频谱差的IFFT预先做成两份空间residual；hit只更新scalar并相加。它借鉴的是线性变换与不变量的交换，不是对任意谱滤波器的捷径。[频谱→空间研究](case-studies/spatial-cache-mlx-study.md)已计入更贵的refresh，并保留大偏置误差拒绝。

来源：[FasterCache/S46](sources.md#s46)、[MLX FFT/Real/S47](sources.md#s47)。最小验证：固定mask、Hermitian对称、轴/布局、scalar日程；与FP64数学参考及原FP32公式分别比较，加入大offset/小delta。变mask、逐frequency gain、非线性或不同shape时重新推导；逻辑complex→real容量减半不证明物理峰值减半。

## 15. 强分母、配对统计与双重质量门槛一起设计

优化候选必须同时面对“优化前原实现”和“去掉可消除浪费的强对照”。NNC合并读回能改变缓存分母的胜负；T-GATE必须与CFG-only比较；DeepCache interval1比stock更慢，不能用它放大近似收益。源码与证据见[NNC/S56—57](sources.md#s56)/[gate成本](case-studies/nnc-gate-controller-cost.md)、[T-GATE成本](case-studies/tgate-mps-resident-cost.md)、[DeepCache完整请求](case-studies/deepcache-sd15-mps-quality.md)。

最小验证：冻结分母/窗口/顺序，每个trial算`baseline_time/candidate_time`后汇总，保留IQR和反收益，不用两个中位数相除。setup/refresh/cleanup、失败和被排除pilot分别记录，完整周期不能只挑hit。

另冻结两道独立门槛：新后端/变体对**同配方**参考的实现误差；近似方案对**full与实际任务**的质量预算。VDE变体byte等值只过前一道；DeepCache真正生成了PNG但全部有损输出拒绝参考图预算。最小复用模板见[评测合同](evaluation.md)，不能用finite、CPU parity或重复hash替代第二道门槛。

## 16. 在实际省算边界计数，不把观测器事件当硬件事件

root/module hook在直接返回旧cache时也可能触发；DeepCache用未包装的Conv2d/Linear body核查深层确实跳过。VDE一个逻辑`.item()`在默认设备mode下可产生两次Python hook entry；NNC签名相同也不是直接观察compile调用。这三类计数回答不同问题。

来源：[DeepCache/S70](sources.md#s70)、[Torch观察器重入/S69](sources.md#s69)、[NNC签名/S58—59](sources.md#s58)；解读见[body跳算](case-studies/deepcache-unet-boundaries.md)、[item校准](case-studies/vde-mps-controller-cost.md)、[packing反证](case-studies/nnc-packing-signatures.md)。

最小验证：full与hit各做一条已知路径；对observer加嵌套depth/逻辑call ID和不启用default-device的控制；所有临时patch用`finally`恢复。软件事件只写API/body/entry，物理dispatch、copy、compile、共享内存和峰值必须有各自独立观测。

## 把技巧迁移到新模型时的交付清单

先交付一条窄而完整的链：固定来源/许可 → 画出省算边界 → no-op与fallback正反控 → 冻结任务质量预算 → 强分母完整周期/请求 → 报告失败与重开条件。数学fixture、随机网络、真实权重组件和真实完整生成分别命名，不沿用其它项目的设备或质量资格。

不建议把本页16条变成同时启用的开关。先选一个瓶颈、一个改动、一个可否证假设；通过后再做组合。当前证据矩阵与尚未闭合边界见[算法入口](README.md)。
