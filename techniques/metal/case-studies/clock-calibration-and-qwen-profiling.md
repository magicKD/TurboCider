# Metal时钟校准与真实Qwen FFN host profile

<!-- knowledge
id: technique-metal-case-studies-clock-calibration-and-qwen-profiling
type: case-study
aliases: evidence-gpu-clock-and-qwen-20261016
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[工作台](../../MAINTENANCE.md) · [前一批profiling](capture-counters-and-failure-boundaries.md)

2026-10-16续接，M4 Max / 64 GiB，macOS26.6.2 / SDK26.5。上一轮是progress，不是完整目标完成。本轮没有继续增加raw capture大文件，也没有删除旧记录/模型、修改系统设置或产品默认。

## 1. 12个时钟控制，不把首个采集bug丢掉

首次clock-reference-smoke-v1的native计算、数值和counter原值正常，但新增的command-window数组被后续trial复用覆盖；只有最后一个trial对应正确。归一化器辨出ns单位，却拒绝整份窗口关联。该记录与源码快照保留，是采集器缺陷，不是硬件clock故障。

改为不可变copy后，新smoke加10个规模控制通过：GEMM64/128/256，vector65539/262147/1048579 elements，graph262147 elements、1/4/16/32 stages。每个主run保留4 warmup+8正式编号diagnostic样本，不另拼正式性能倍率。11份通过记录共716个stage窗口。

实查CPU raw落在CLOCK_UPTIME_RAW的ns区间，不在125/3 timebase的Mach tick区间。GPU affine比例观测为1 ns/raw tick。reference bracket最大875ns；单stage/CB的duration残差最大约41.5ns，多stage则检查在同一CB内包含。**这些不是总误差上界，也不是独立物理传感器验证**；不能据残差宣称41ns硬件精度、GPU频率或跨OS/SKU单位恒定。

固定门槛与实验规模在[candidate控制（过程原件已移出）](../../MAINTENANCE.md)，实现及数学边界在[校准说明](../../profiling/timestamp-calibration.md)。旧raw-counter报告没有reference bracket，继续保留null秒数，不做事后升级。不同stage数只验证记录数量、关联和数值，不要求elapsed严格线性。

## 2. 真实Qwen，而不是把native toy当模型profile

两次独立进程读取原始BF16 checkpoint、真实layer0 capture与已有fixture；第一批step1/19，后续step10。完整消费者M1024/H4096/F12288、`[1,1024,4096]` BF16、base无LoRA。所有选定权重/activation SHA、checkpoint stat、源码快照与结束身份核对通过。

| point | compiled output-ready中位数 | staged gate/up / epilogue / down host中位数 |
| --- | --- | --- |
| step1 | 21.660 ms | 14.587 / 0.967 / 7.422 ms |
| step10 | 22.078 ms | 14.832 / 0.725 / 7.683 ms |
| step19 | 21.667 ms | 14.449 / 0.489 / 7.346 ms |

每point有2 warmup+4 compiled正式编号样本、2 staged样本。所有compiled/staged输出对BF16 replica误差为0；独立完整K的CPU FP64抽样过固定3%局部门槛，完整finite检查通过。仅抽样FP64，不是全输出FP64，更不是原native请求bit-exact或图像质量。

表右列是特意拆成三个phase并每次drain后的host程序，不是原compiled图的GPU kernel时间；没有speedup或“profiling overhead”比率。它说明在这个诊断程序里gate/up占较大部分，可据此规划布局/融合/attention对照，不能宣称定位了DRAM/ALU瓶颈。没有将新native timestamp资格套到MLX queue，physical timestamps仍为null。

源码与runtime资产hash、全部样本与phase事件在JSON。独立复核，不从新probe导入benchmark或自动运行。`host_trace.py --point ... --trial ...`可只读导出一个diagnostic CPU lane；未启动viewer。

## 可复用结论与尚未完成

单位验证与同trial窗口关联是两个门；输出ready、queue drain、完整request是三个边界；采集元数据必须冻结owner状态。正式分数与诊断拆分既保留有价值的局部信息，也避免把观察程序当优化后的原程序。

下一步仍包括真实Qwen attention数据/多个head/shape与mask、native GPU/ANE handoff、ANEForge与PMU、更多层/LoRA/完整请求、媒体/VAE/内存/CPU组合，以及多维metadata/aliases/version claims/薄AI skill。没有新增产品加速资格，本轮未自动commit/push，整体目标保持进行中。
