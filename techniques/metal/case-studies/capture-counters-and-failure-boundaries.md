# M4 Max GPU profiling：真实capture、stage timestamp与parser反例

<!-- knowledge
id: technique-metal-case-studies-capture-counters-and-failure-boundaries
type: case-study
aliases: evidence-gpu-profiling-20261016
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[研究工作台](../../MAINTENANCE.md) · [公共实验](../../profiling/gpu-timeline-and-counters.md) · [来源卡](../../sources/repos/gputrace-timing-provenance.md)

2026-10-16，Apple M4 Max / 64 GiB，macOS 26.6.2，SDK26.5 / CLT。此批是synthetic operator与profiling能力研究，非Qwen新性能资格、ANE物理证据或产品默认变更。前一整理提交为`5ed9424`；继续完整roadmap，不因本批成功而宣称全目标完成。

## 实际执行与身份

- 原有native lab增加显式诊断开关，不替换生产runtime；每run新进程、源码快照、binary hash、ABBA输入匹配、4 warmup与8正式编号样本（smoke为4）。所有正式比较与诊断串行，没有同时做Go build/test。
- 11份native记录：9个三工作负载×三模式；1个小shape stage smoke；1个不支持的dispatch sampling失败。失败保留，不静默降级。capture只取trial0 baseline。
- 10份capture解析：3份native、最初MLX v1及6个后续MLX控制。所有raw capture与源码/编译cache留在本地ignored输出，不提交UUID、buffer或trace原件。
- `gputrace@3d7d32e...`隔离构建成功，两个选定parser package测试通过。普通stats/timing执行，不使用全局安装、App、Xcode replay或默认shader source扫描。原参考checkout与隔离源码副本均保持tracked clean。
- `xctrace list templates`退出72；CLT环境不提供该Instruments入口，但公共programmatic capture与stage counter实际可运行，不能混为“没有任何profiling能力”。

## 1. 三工作负载与结构ground truth

| 工作负载 | 未instrument baseline / candidate host中位数 | 捕获中预期 / 解析dispatch数 | 计时解释 |
| --- | --- | --- | --- |
| GEMM 256³，scalar / 教学SG8 | 0.907 / 0.801 ms | 1 / 1 | 编码、submit、wait包含；非MLX优化GEMM对照 |
| vector 1,048,579 floats，scalar / float4 | 0.892 / 0.838 ms | 1 / 1 | tail通过，仍是小型host边界 |
| 同规模16层chain，逐层wait / 单CB | 5.895 / 1.493 ms | 16 / 16 | 相同数学工作，主要改变submit/wait边界 |

两列中位数之比不等于paired倍率中位数；JSON保存全部样本与四个pair ratio，不拼分母。样本少且未做独立性能确认，不从表推广产品收益。三个模式的输出误差最大约`2.04e-7`、`2.99e-8`、`0`，finite/tail通过。capture模式只有一个sample在捕获中，不据其run median计算“采样overhead”。

公共API实际枚举到仅timestamp集合，stage boundary=true、dispatch boundary=false。三份主counter诊断的144个正式编号stage窗口有效（含warmup为216个）；smoke另有8个有效窗口。全部是本设备原始值，sentinel/缺失检查独立；没有把它们转换成ns/seconds或占用率。graph每个arm都采16个encoder，baseline有16个CB，candidate只有1个。

## 2. MLX：文件存在与可用结构是两个验收

相同seed29，FP32 `tanh([128,256]@[256,384])`，独立完整FP64 oracle最大绝对误差约`1.32e-7`。

| 控制 | stats结果 | 可解释范围 |
| --- | --- | --- |
| 初始v1、v2 baseline、eager、fresh inputs、仅开始前同步 | 都因capture header太小退出1 | 文件存在、数值通过；不能证明capture包含可解析dispatch |
| v2两端同步、v3仅停止前同步 | stats退出0，1 CB / 1 compute encoder，parser识别dispatch=0 | 结构可读但dispatch覆盖仍未验证；不能称物理执行了0次 |

这是对本wheel/OS的边界观察，不把原因推广到所有MLX/Metal版本。release源码`eval`等待output event，`synchronize`额外end/commit/wait，解释方向一致但不是wheel构建等价或逐调用根因证明。所有失败对照仍在JSON，未删除旧v1或改写其原报告。

## 3. gputrace：synthetic是预测，不能变成性能分数

native trace解析出的“unique kernels”数为3、2、2，多于已知实际dispatch的函数种类；它包含库函数与identifier-like字符串，不是执行计数。对应timing总量为11ms、2ms、2ms，均明确synthetic。

同步后的MLX trace读出16,446个函数名/identifier候选，synthetic总量为25.731秒；空capture的timing则成功退出并输出0 rows / synthetic总量0。两种都不能解释为GPU真实耗时、性能退化或零工作。parsed dispatch=0可能是覆盖限制，不是GPU没有执行。

本地adapter把heuristic值保留为predicted，空/未知物理时间保持null，遮蔽路径/UUID/明显credential label，并保留输入/规范化内容hash。profiler源也不自动获得物理端点资格，因为已读代码会串接duration来重建时间轴。`pprof`路径对extracted/store0的boolean与metrics路径存在差别，因此采用固定commit的来源策略，不盲信单字段。

公共证据只嵌入大名录的前8行、明确省略数和完整内容hash；完整原件留本地。归一化器没有运行Xcode、读取其他源码目录或修改任何trace。

## 后续与重开条件

优先做timestamp单位/误差校准与尺寸/重复缩放，再把同一工作负载的未采样性能、合法计数器和trace窗口关联。后续独立验证profiling streamData与限界source mapping、真实Qwen attention/FFN consumer、native GPU/ANE handoff、ANEForge/PMU、LoRA/更多层/请求和媒体内存组合。官方plan、raw counter、派生比率与私有PMU不互相冒充。未修改默认、不自动commit/push，本轮不是研究终点。
