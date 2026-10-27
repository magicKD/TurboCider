# 真实Qwen FFN：正确的图与值得采用的图不是一回事

<!-- knowledge
id: technique-gpu-ane-case-studies-qwen-ffn-tiling-and-concurrent-cost
type: case-study
aliases: evidence-qwen-ffn-tiling-20261015
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[研究工作台](../../MAINTENANCE.md) · 全部样本与身份 · [图合同](../../ane/coreml-graphs.md) · [固定来源卡](../../sources/README.md)

结论限定于本机、此图与此消费者：**exp形式通过组件数值检查；并发能改善同一切分的串行执行，但仍明显慢于完整融合GPU基线，所以不改产品默认、不晋升新后端。** 这不是“所有ANE/所有shared IO都无效”。

## 实际输入，不用重复小样本冒充大shape

2026-10-15在Apple M4 Max、40 GPU cores、64 GiB、macOS26.6.2/25G83上，使用既有CLI：其构建manifest中的648个native source hashes与当前源码匹配，CLI/library另以SHA固定；manifest不是二进制签名或可复现构建证明。实际完成20-step/512px/base/seed42请求，cache off、强制GPU FFN，捕获layer0/15/31 × step1/10/19的9个原始`[1,1024,4096]` BF16矩阵，约72 MiB。二进制、原始capture、checkpoint中选定tensor与FP16转换结果的SHA均保留。

本轮只重放layer0完整`M=1024,H=4096,F=12288`：原checkpoint fused gate/up `[24576,4096]`与down `[4096,12288]`，没有随机权重、没有裁剪rows或重复行。layer15/31尚未做同样图候选重放，不能继承layer0资格。捕获本身有采样开销，28秒左右的wall不是普通请求分数。

实际Python依赖为3.11.16、NumPy2.0.2、MLX0.32.0、coremltools8.3.0。完整SDK和环境字段以JSON为准；不将clone源码HEAD冒充所运行的库版本。

## 受保护参考与边界

GPU基线是compiled fused gate/up→SiLU×up→down，不是分开的eager GEMM。各候选共用FP16 source；FP32 MLX提供全输出参考，CPU FP64 einsum在4个真实row、4个输出channel上完成所有归约，检查独立16点见证；再比较原BF16 compiled GPU replica的全输出。

固定门槛：同FP16 source全输出relative L2≤0.02、独立FP64抽样≤0.02；原BF16 replica relative L2≤0.03且cosine≥0.999。前者是组件门槛，后者检查BF16→FP16近似的局部代价；都不是端到端画质许可。发现失败后没有放松门槛。

目标边界从ready GPU FP16输入与resident权重开始，计入Core ML输入读取/必要转置、实际prediction返回值、输出恢复、GPU suffix与join，直到`mx.eval`完成。初始权重转换、compile/load、独立oracle和最终NumPy读回在边界外，另有setup数据。**不包含每层权重streaming/重新stage，也不代表原生IOSurface零拷贝。**

每次独立进程；4次warmup，8次ABBA测量、4个matched pair。源码权重/同pair输入一致；step1作discovery，step19另起进程确认。bootstrap区间只是4对样本的探索性不确定性，不能替代更大确认集或产品门禁。没有清理系统/driver缓存。

## 全部方向都保留，不只展示获胜部分

下表为各自run的warm中位wall，单位ms；倍率是median of paired ratios（不是两中位数相除）。完整样本、warmup、误差、环境、失败与exploratory interval在JSON中。

| 图/执行 | 对照GPU或串行 | 候选 | paired倍率 | 解释 |
| --- | ---: | ---: | ---: | --- |
| frozen、2D、monolithic、sigmoid | — | — | 不输出 | 抽样FP64门槛拒绝；只有warmup失败样本，不计性能资格 |
| runtime、4D、K2048/N4096、sigmoid | — | — | 不输出 | 同样被固定抽样门槛拒绝 |
| frozen、2D、monolithic、exp | 26.23 | 231.13 | 0.115× | 数值通过，但更慢 |
| runtime、4D、K2048/N4096、exp | 26.97 | 223.83 | 0.121× | 数值通过，不能因计划全在ANE就认定更快 |
| runtime半宽6144、serial、exp | 21.70 | 139.84 | 0.155× | 含GPU suffix/restore/join的完整输出 |
| 同切分hybrid，step1 | 25.11 | 120.91 | 0.208× | 仍明显慢于完整GPU |
| 同切分hybrid，独立step19 | 21.79 | 120.29 | 0.181× | 晚期真实输入确认负收益 |
| 清理自有编译副本后，step19再确认 | 21.92 | 120.50 | 0.182× | 记录磁盘admission；仍约5.5倍耗时 |
| 同一图、同一进程serial→hybrid配对诊断 | 139.92 | 120.46 | 1.163× | 并发减少相对serial的wall，不是相对GPU的加速 |
| 带源码快照，step19再确认 | 21.87 | 120.73 | 0.181× | source前后匹配；负收益不变 |
| 带源码快照，同图serial→hybrid诊断 | 141.34 | 121.13 | 1.172× | 仍只证明相对serial的局部收益 |

带timeline的两个后续serial→hybrid配对诊断得到1.163×与1.172×；最初一次同类诊断为1.196×，均保留在JSON。初轮13个report包含数值失败与资源失败；2026-10-16追加CPU控制重试，成为14个report，并另有同bundle policy诊断。不能用两个独立run的中位数直接证明重叠，因此额外做了同进程、同图、同输入的serial/hybrid ABBA。背景服务没有被终止、电源/频率没有锁定；n=4的探索性区间保留这种波动，不能据此宣传稳定生产收益。

## 数值负结果与profiler实际告诉了什么

sigmoid图的全输出误差约1.48%，仍低于2%的全局门槛，但16点独立见证约2.83%，所以拒绝；不能只展示全局平均而隐去失败见证。编译后的MIL将该表达融合成`silu`，exp图保留exp/add/div表达。这说明等价表达可触发不同lowering；**尚未定位到逐算子根因，不能称“ANE sigmoid硬件错误”。**

初次CPU_ONLY控制在转换阶段发生`No space left on device`，没有有效prediction，该失败继续保留。2026-10-16存储余量恢复后，未改门槛重试同一图：CPU_ONLY完整输出与独立抽样检查均通过；完整消费者warm中位约50.33ms，仍慢于该run的FP16 GPU对照27.25ms，不作为新性能候选。

进一步用policy_probe.py让**同一个compiled bundle、同一个输入**按CPU_ONLY/CPU_AND_NE/CPU_AND_NE/CPU_ONLY调用。每次确认输入hash未变，全部编译资产before/after SHA一致；runner另留快照。独立FP64抽样误差重复为0.4769%/2.8337%/2.8337%/0.4769%，前者通过2%门槛、后者不通过。两策略全输出相对差约1.5485%。这说明本环境有policy/path相关的数值行为，不能把旧负结果泛化成所有Core ML图都不正确；尚未定位到单个算子或物理ANE硬件原因。该probe的return0只表示采集完成，host call spans不作正式性能分数。

MLComputePlan对runtime tiled-exp的所有非const操作都给出ANE preferred device，完整消费者仍约224ms；frozen-exp的首投影/拆分计划在CPU，且cost均缺失。这里device/cost是`predicted`，数值和wall是另一次实际执行证据；null不改成0，不把cost乘wall伪造每op实测。

host-only trace记录prediction调用、suffix提交与restore/join的CPU span，使用同trial的perf_counter域；可说明host提交顺序，不能证明GPU/ANE物理同时执行。未获得Xcode GPU timeline、PMU或逐kernel数据；这些仍待研究，不以API异步代替硬件证据。

## 存储与复用

遇到磁盘不足后，只移除本轮两个已终止实验的4个可再生成Core ML package/compiled目录。事先保留逐文件hash；原模型、9个capture、FP16 fixture、原始reports/compute plans和旧实验均未删除。原始编译副本可按脚本、source与工具链重建；APFS实际占用不能简单相加目录逻辑大小。

新runner预留runtime至少512 MiB余量；frozen另预留4份原始权重大小。这是保守admission政策，不是编译器磁盘硬上界。最后GPU/hybrid独立确认前可用约1 GiB，负收益不依赖失败的初次CPU控制；之后余量恢复至约2.8 GiB才满足frozen重试admission，未为重试删除旧数据。

可复现入口：图构造、完整消费者基准、显式超时监督、只读汇总、host trace适配。产品捕获与权重导出只在[可选integrations](../../examples/README.md)，不属于核心离线依赖。

早期探索保存了runner/graph hash，但未为每次改动归档完整源码，因此不冒充完整候选源码谱系。之后增加每run的3文件源码快照、结束一致性检查与汇总时hash复核；关键runtime独立确认和serial/hybrid诊断另用带快照的新进程重跑。初期数据仍保留，`source_snapshot_status`区分其证据强度；重建Core ML随机metadata/缓存产物不保证字节同一。

下一步不是放宽质量门槛，而是分层验证native shared IO、权重表示/量化、完整LoRA、其他层深和shape，并继续GPU kernel与profiler研究。只有通过新的组件、真实消费者与请求质量/性能门禁，才考虑产品候选；当前保持GPU路径。
