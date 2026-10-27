# ANEForge 实测：缓存负控、SiLU 缩减与有界 FFN

<!-- knowledge
id: technique-ane-case-studies-graph-rewrite-search-cost
type: case-study
aliases: evidence-aneforge-20261016
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[总目录](../../README.md) · [技术卡](../graph-rewrite-search.md) · [固定源码](../../sources/repos/aneforge-rewrite-search.md)

2026-10-16，本机 M4 Max/64 GiB、macOS26.6.2、CLT/SDK26.5、Apple clang21、Python3.11.16、NumPy2.0.2、MLX0.32.0；GPU实际报告 `applegpu_g16s`。上游固定 `896c4943...`，原树只读；在 ignored 副本构建同步 e5rt shim，没有安装包/下载模型/替换 App 或改安全设置。

## 记录、计时和资格

15份 receipt：13组真实设备比较、1份成功 CPU contract、1份初始 Python 参数错误。13组各保存3 arm×(1 first +2 warmup +6 measured)=351条完整输出检查；所有失败 arm 留下，但不暴露其倍率。实际 pattern 为三 arm 六种排列，既不是简单 ABBA，也不从最佳单次时间挑 winner。

所有正式 wall 为**host FP16 数组 → 新 host FP32 数组可读**；各框架的输入 copy、同步与输出恢复计入。冻结权重 prepare、MIL emit/compile/load、GPU wrapper setup 分列；MLX lazy compile 留在 first call。没有清理系统 cache，首编译不是严格 cold start。后台系统备份/更新进程在 setup 时可见，未停止；没有连续 thermal/功率测量，因此短时观察不能当长时保证。

runner为保持确定性先生成三组权重再生成输入；activation/reduction虽记录这些RNG权重hash，实际图并不消费它们。这里只把MIL/blob中实际存在的常量当编译内容，不把hash列表误读为每个case都执行了三个projection。

独立 CPU full FP64 einsum oracle，原始权重/输入来自确定性 FP16 source；门槛为 full-output relative L2 ≤0.01。它不是声称候选保持每个 FP16 中间舍入，也不是请求图像质量。物理 ANE 放置未独立观察；mask0x4、同步返回和正确输出是 API/数值事实，不是 PMU/trace 证据。

## 三类缓存与返回值反例

真实上游 `_graph_key` 在两个不同 W 上相同；MIL hash 相同，weight blob 与 SHA256(MIL+blob)[:24] 不同。mock cache hit 确认 build1次、measure0次；没有加载 dylib。另一个明确 mock 让所有候选 error0.2、target0.001，`tune_precision` 仍返回 most-accurate model，reason 保留未达标。不能把 mock 的10us/123us当测量，也不能把返回 model 当 numerical gate。

causal SDPA 在 tuner route 候选中被排除；直接 decomposition helper 仍能去掉 sdpa 节点，不自动继承该 guard。未执行不安全的 masked 变换。

## 失败缩减与实际数字

| Case / 完整边界 | 数值观察 | 时间/性能观察 |
| --- | --- | --- |
| matmul32×128×256，identity | ANE最大L2约0.000449，GPU约0.000211 | identity约1.006×、区间跨1，不能晋升噪声 |
| 同shape INT8 W | 最大L2约0.006516，通过局部门槛 | 对FP16约1.018×、区间跨1；不是稳定量化收益 |
| reduction32×512，cancellation | reduce-sum与ones-matmul均exact | 约0.973×、区间跨1，没有证实改写优势 |
| 原生SiLU + 完整FFN128×512×1024 | baseline最大L2约0.04386；INT8版约0.04561，均拒绝 | 保留耗时，不给这些 arm speedup |
| 仅activation / projection+SiLU | 原生表达约0.04246/0.04347；exp版约0.000439/0.000622 | 缩减把失败保留到单算子程序，不推断内部物理原因 |
| exp FFN，seed17 | 最大L2约0.004235；FP32 GPU约7.1e−7 | 增强后的FP32 GPU约0.863ms，ANE约0.283ms；配对约2.969× |
| exp FFN，seed29确认 | 最大L2约0.004267；FP32 GPU约7.1e−7 | FP32 GPU约0.933ms，ANE约0.290ms；配对约3.265×，探索性区间[3.035,4.141] |

seed29另有 FP16 GPU确认，约0.869ms vs0.282ms，配对约3.067×、区间[2.647,4.133]。FP32 GPU是对同一原始FP16参数的精确拓宽，不换 checkpoint；更高 compute precision不是同一 FP16 中间配方，分别报告。两份确认的 W/input seed 都变了；这不是完整请求确认。

早期 GPU路径在 compiled output 外另做 GPU cast；随后分别测 host cast、图内 cast 和 FP32 compute，所有记录留存，未把旧 wrapper 的 hash 换成当前源码。prepared GPU诊断也留下全样本/数值，但和 host-fed ANE边界不同，明确不作配对倍率。这里的 compiled MLX API对照不是宣称所有可能的 GPU kernel 都已最优。

在32×32的常量激活见证中，原生 `SiLU(0)` 返回−0.001704216，而 exp版和GPU为0；`SiLU(-0.5)` 的原生值−0.203613281，数学值−0.188770334。6点原值/范围均在JSON；它们是合成常量，不是用户 tensor。此结果只绑定当前 private e5rt/MIL/program 配方，不宣布通用硬件 bug。

## 所有权、失败和来源

活 Program 的 output view 会随下一 call 更新；Model 返回副本稳定；输出 poison 在成功同步 execute 后全部覆盖，并与第一次 Model结果相同。未在 release 后访问 view，未开启 experimental async、多 op共享、profiler 或 fork-safety override。

初次 CPU frontend 用错 `sdpa(causal=...)`，应为 `is_causal=...`；该错误保留，不标为 ANE 不支持。此最早失败只有 runner hash、没有原源码快照，显式记录较弱 lineage；后续所有设备 run 都有 runner/helper snapshot。最终 curator 比较了58个隔离源码文件、各 run关键源码前后 hash、shim与MIL/blob hash、完整调用顺序和 gate；它是 data verifier，不是重新执行硬件/输出 tensor的 verifier。

## 对 TurboCider 的采用决定

吸收：三类 cache 分离、未达预算严格拒绝、独立 baseline oracle、单算子失败缩减、借用 view owner 合同和候选全样本确认。exp-SiLU是值得研究的有界表达，不修改产品默认。

不晋升：当前没有真实 Qwen checkpoint/activation、GPU→ANE actual-backing/handoff、并发、LoRA、全请求或长时内存/热状态证据；不能拿本页约3×覆盖[真实Qwen公共CoreML负结果](../../gpu_ane/case-studies/qwen-ffn-tiling-and-concurrent-cost.md)。后者的形状、后端和生产者边界不同。

剩余：更宽激活/极值域的exp数值、精度岛、真实消费者、编译失败/取消生命周期、PMU语义及与GPU重叠，均继续研究。没有清模型、旧实验、原始数据或系统 cache；本批后可用磁盘约0.9GiB，后续大图仍需 admission。
