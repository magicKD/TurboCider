# GGML：同输入、同图、同 buffer 的融合计时

<!-- knowledge
id: technique-gpu-ane-case-studies-ggml-matched-fusion-timing
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [few-row 源码](ggml-fewrow-mma-and-reuse.md) · [原始缺陷回归](ggml-residual-identity-regression.md) · 全部确认样本 · [源码/构建](../sources.md#ggml-fusion-timing)

研究从 2026-10-16 延续至 2026-10-17，设备运行完成时间另存 UTC；ignored artifact 目录保留开始日期，不将 10 月 17 日整理伪装成 10 月 16 日完成。本机仍为 Apple M4 Max、macOS 26.6.2 / 25G83、SDK 26.5；固定 GGML ref `988190680d5a89fce97de3c20df2c2813731fd61`，不声称是当日最新 HEAD。

本轮从正确性回归推进到受控组件计时。结论很局部：M=1000/K=1024/N=8 的两个量化类型，长块反序确认均观察到约 1.12–1.16× 调用速度比；更大矩阵多数接近持平。没有全局融合胜出、完整 FFN/模型、ANE 或 M5 性能结论。

## 测的是完整小图，不是仅重复最后一个 ADD

原创 host直接调用未修改的 GGML，图为：

```text
同一份 X ──→ X @ WA.T ──┐
         └─→ X @ WB.T ──┴─ ADD → Y
```

每个 `ggml_backend_graph_compute` 都重新执行两个 matmul 和 ADD，而不是只把最后一个 out node 插进 graph 多次。后者是所读 `test-backend-ops::eval_perf` 的一种摊销方式，不能直接解释成两个 matmul 每次都执行的完整图耗时。

两臂仅切换原始 backend 注册的 `ggml_backend_fusion_*` opaque handle；这是**内部、尚非稳定公共合同的调试接口**，不当产品 API。单进程内使用相同 graph、tensor、权重与输出 backing，不重建另一套随机输入。权重另用标记为 WEIGHTS 的 buffer，图输入/临时值/输出用另一个 buffer；不能把整个 buffer 都标 WEIGHTS 而错误排除 residual 融合。

`ggml_backend_graph_compute` 在本 ref 中调用 async submit 后 synchronize，所以计时是 host 看到的 encode/submit/等待完成的 graph-call wall time，不是 device timestamp。每块结束再读回输出；统计开关/reset/read、输入生成、分配、H2D、读回、校验不在计时区。fusion counter 的轻量 bookkeeping 仍在 encode 中，未声称完全等于关闭调试的产品路径。

未调用模型 scheduler/graph optimizer；三节点图已经是可检查的邻接顺序。完整底层代码、35 个库的加载、额外 command buffers 和内部 allocator 行为仍由原始 backend 决定；调用数不等于 GPU command-buffer/dispatch 数。

## 输入固定到 packed bytes，而非只固定原始 FP32 权重

两种格式 Q4_0/Q8_0，(M,K)=(1000,1024)、(4096,14336)，N=1/8/16；共 12 case/进程。N=1 不满足本 ref 的 few-row MMA/ADD 融合门槛，是零融合对照，不能标成“融合也加速了单 token”。

fixture 使用确定性 uint32 mix、不同 WA/WB seed、每 32 元素 block 的非恒定 half scale。按原格式手工生成 Q4_0 low-half/high-half nibbles 或 Q8_0 signed bytes，不运行原 quantizer，也不拿原 FP32 checkpoint 当量化 reference。scale 为可精确表示的小数，激活则用整数/333 后舍入 F32，补充上一轮短小 dyadic fixture 未覆盖的情况。

独立 Python/NumPy 实现重建每一个 packed byte 和 F32 输入，并核 SHA-256；CPU FP64 reference 计算 `X @ (WA+WB).T`，其权重由独立数学式重建，未调用上游 decoder。两个投影在数学参考中合并后求和，不要求与两次 FP32 reduction 逐位一致。

完整 actual 输出保存为 `.f32` sidecars；确认阶段两进程×两臂共 **1,019,200 个输出值**全部与 FP64 比较：最大绝对误差 `1.4397783525055274e-4`，最大 relative L2 `5.399129275600726e-7`，均低于预设 `1e-3 / 1e-5`。两臂及反序进程的 actual 字节均相同，不等于对 FP64 exact，也不授予模型量化质量。

每个计时块末的完整输出还与该臂预热结果 bitwise 比较，输入与权重在该组结束后读回核对未变。没有逐次读回块内 64 次调用的中间结果，也没有外部 OOB guard、fault/cancel/drain 资格。

## 短 pilot 揭示的噪声没有丢弃

先运行 6 轮 ABBA/BAAB、每块 8 次调用。虽然全量数值通过，某个 N=1 未变 kernel 的 `t(off)/t(on)` 达 1.0986，说明短窗口存在明显漂移。原始 pilot、源码/可执行文件快照和完整 logs 都保留，确认数据里包含全部 12 个 pilot 比较，不拿它替换成后来更好看的表。

在确认阶段开始前写入计划：保持输入、图、shape、4 次/臂预热、3% 描述性中性带不变，增加到 8 轮、每块 64 次调用。两进程分别用 ABBA 与 BAAB，并反转 case 顺序。计时块全部保留，不删 outlier、不取 best trial。

确认阶段共 **768 块、49,152 次同步 graph 调用**；pilot 另有 576 块/4,608 次。768 块不是 768 份独立机器实验；3% 不是置信区间，更不是通用硬件误差阈值。

## 确认结果：R = 未融合 median / 融合开关开启 median

每格为 ABBA / BAAB 两个进程的 R。N=1 两臂实际上都不触发融合。R>1 是调用速度比，不是同比例 latency reduction。

| 格式与 (M,K) | N=1 对照 | N=8 | N=16 |
| --- | --- | --- | --- |
| Q4_0 (1000,1024) | 1.020 / 0.998 | 1.164 / 1.120 | 1.005 / 1.021 |
| Q8_0 (1000,1024) | 0.993 / 0.998 | 1.129 / 1.117 | 0.997 / 1.021 |
| Q4_0 (4096,14336) | 0.998 / 0.993 | 1.014 / 1.014 | 1.008 / 1.007 |
| Q8_0 (4096,14336) | 1.003 / 1.007 | 1.013 / 1.006 | 1.003 / 1.002 |

小矩阵 N=8：Q4_0 两轮 on/off 分别约 110.02/128.02 μs 与 112.30/125.75 μs；Q8_0 为 109.65/123.78 μs 与 112.07/125.17 μs。只有这两个 case 在两种顺序下都超过预先描述的 3% 带；不是 p-value 或稳态长期保证。

大矩阵 N=8 的 on 从约 325 到 462 μs 不等，但相对优势只有约 0.6%–1.4%；同一 Q4 shape 的两进程绝对耗时也有漂移。其余 case 不据此授予稳定速度收益。融合 counter 的正确增量说明开关有效，不能把所有小差异都当融合效应。

这与小图中一次 ADD/绑定/交接占比更高的解释相容，但没有 GPU counter，不能诊断精确瓶颈。它给强 GPU 分母的一个组成提供证据，不等于这个小图已代表完整 FFN、最优 llama 调度或真实模型分母；更不能把局部倍率乘进 GPU+ANE 预期收益。

## 准入、构建与 oracle 失败分别处理

五次设备进程（smoke、两 pilot、两确认）各在五次一秒总 GPU 占用均为 0% 后启动；串行执行。CPU reference 只在设备实验结束后运行，不与正式 GPU 窗口争内存带宽。粗占用不是物理独占或全程排除外部短任务的证明。

原始 GGML 的 178 个复制 tracked 文件全部匹配固定 Git blobs；12 文件为本轮明确阅读范围，二者不混算。参考 checkout、产品后端、全局环境和系统缓存未改动。

本地 CMake 首次只启用 OBJCXX，使上游 `.m` 被按 C++ 编译而失败；仅将自有 project 声明改成同时启用 OBJC/OBJCXX 后通过。五个 configure/compile receipt 的返回码为 0、2、0、0、0，失败日志与编译 flags 保留，不当上游默认 build bug。pilot/确认可执行文件分别保存；两者变化只是 host 的计时次数/profile，原始 GGML 未修改。复制目录的 Git stamp 问题仍按上一轮处理，不用 banner 自证来源。

CPU 环境 NumPy 2.0.2 的 BLAS matmul 曾报 divide/overflow/invalid 警告。独立诊断全结果有限，且与非 BLAS 路径 max_abs=0；保留三条警告，不称它们是 GPU 错误或已定位整个 NumPy 根因。正式 oracle 改用 `einsum(optimize=False)`，核对安装源码确实直达 `c_einsum`，并检查所有参考值/指标 finite、选定位置与标准库 `math.fsum` 一致。这里没有屏蔽真实 NaN/Inf 来通过验证，也不使用 BLAS norm 作为错误判断。

## 复核与未测边界

普通离线测试不需 NumPy/参考仓库/GPU；完整数组复核需显式指定原 artifacts 和已安装 NumPy，运行 CPU，不重跑设备：

source capture 另用 只读入口，禁用 lazy fetch；[构建与版本证据](../sources.md#ggml-fusion-timing)和[验证摘要（过程原件已移出）](../../MAINTENANCE.md)绑定 host、原始库、副本、二进制、当前/历史 curator 及原始输出。

设备复测必须换新目录、串行运行并复用 guard，不能绕过忙/缺 counter 拒绝。没有 device-timestamp kernel time、cold first request、功耗、峰值总驻留、长跑置信分布、MMA 对 legacy MV 的同输入计时、模型/LoRA、ANE/MPP 或物理异构 overlap。下一步应放到真实消费者中重新测关键路径，而不是把这张表当产品启用许可。
