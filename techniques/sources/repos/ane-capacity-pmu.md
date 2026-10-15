# measure_ane_capacity：序列化、fallback 与 PMU 计分

[来源目录](../README.md) · [原 PMU 工具](ane-pmu-provenance.md) · [实验合同](../../profiling/ane-pmu-lab.md) · [本机报告](../../review/ane-pmu-20261016.md)

补充来源：`freedomtan/measure_ane_capacity@2d3809ba2a3654db71d6790574c9b0247d5b4493`，2026-10-16新建 sparse checkout，未改变既有参考项目。完整读取 `measure_ane_pmu.m`723行，另读 README1–170、App bridge 的选定 PMU段、interpretation guide1–80、Makefile和LICENSE；不宣称运行整个 App、FP8/INT8或审计全部仓库。逐文件 SHA 与范围在[来源登记](../../references/sources.json)。

## 数据流中值得学的部分

MPSGraph placeholder/常量 conv → `preferredDevice=2` 的编译描述 → warmup → 查找 serialized ANE region → package export → 独立 `_ANEClient` 加载 region / IOSurface / stats。这比只看最终 TOPS 表更有价值：**每个阶段是否真的产生了被下游使用的资产必须单独确认**。

临时目录查找先对比 before/after，失败后会选最新含 `.bc.mlir` 的目录；不绑定当前PID，可能误选其他工作。本轮实际执行时用全新 owned `TMPDIR` 和 package 路径隔离这个搜索范围，没有扫描共享临时目录或复用他人 bundle。serialization/delete目标也只在本任务新建目录，不运行 Makefile clean。

PMU仅在 `profilePMU && activeBundlePath` 时进入。没有 bundle 时程序仍打印汇总表，latency使用 `anePmuTimeMs` 或 fallback到MPSGraph wall，未取得的 counters保留初始化0。因此表格中的0、ANE标签和正常退出都不能证明PMU已采到。

本机明确 FP16/B1/C32/H=W32/K3/L1、2次重复、`--no-gpu`，原源码隔离编译成功。MPSGraph运行并导出5文件package，但没有 `ane_bundle`，PMU分支marker未出现。该记录是 **PMU跳过**，不是0 cycles/0 DRAM，也不因 `--no-gpu` 就证明 MPSGraph执行没有GPU路径。未独立验证该上游输出数学，不采用它的性能分数。

## 测量与呈现的反例

- GPU与ANE的 compile optimization level分别为0和1，不能直接将表格当匹配的最强GPU分母。
- PMU路径 warmup/timed的某些 BOOL 返回值未检查；缺失NSData时用全零数组继续相减/计算派生值。
- 核心CLI仍将host evaluate wall解释为silicon latency，并固定乘/除16、引入名义clock/峰值假设。这些是待验证模型，不是物理单位校准。
- App `MILCapacityEngineBridge.mm` 的 `minLatencyMs=avg*0.95`、`maxLatencyMs=avg*1.05` 是构造的范围，不是实测极值；该路径的“非零输出”检查也不等于 finite或独立数值对照。
- 报告的SRAM阈值、整数乘法器/能耗/峰值等解释保持 source-reported；本机没有取得counter，不能借这些文字补出实测结论。

借鉴：固定source、独立fill/验证、同workload多入口、显式 serialization身份；同时为skip/failure/不同边界保留null与状态。不要复制庞大引擎或把一段理论计算变成硬件attestation。

许可：根LICENSE为BSD-3-Clause，Copyright (c) 2025 Koan-Sin Tan；第三方coremltools schema另有许可。这里只发布原创说明/客户端与脱敏摘要，不复制shader/App/模型或framework。编译binary和package仍保留本地。

[固定CLI](https://github.com/freedomtan/measure_ane_capacity/blob/2d3809ba2a3654db71d6790574c9b0247d5b4493/measure_ane_pmu.m) · [MIL bridge](https://github.com/freedomtan/measure_ane_capacity/blob/2d3809ba2a3654db71d6790574c9b0247d5b4493/ANECapacityApp/ANECapacityApp/MILCapacityEngineBridge.mm)。
