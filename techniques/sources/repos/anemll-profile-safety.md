# anemll-profile：先审运行副作用，再读性能数字

[来源目录](../README.md) · [公共计算计划探针](../../labs/ane/compute_plan.py) · [测量口径](../../profiling/measurement.md)

来源：`Anemll/anemll-profile@bf8528105a0e61fda2c80d2c9b19615ecab9b7fa`，`anemll_profile.m`阅读1–150、170–460、1050–1270行；SHA-256 `fcf70badf2acbd5857952b6c91f9024fed810a803a7b1a252dcdade2015830aa`，工作树干净，2026-10-15。根许可MIT，Copyright 2026 Anemll。

## 为什么本轮不直接执行原工具

`startLogCapture`启动整个`com.apple.espresso` subsystem的log stream，并开启私有日志显示，不只针对自己的PID。主路径删除用户目录下此工具的e5bundlecache以促使日志重发。即使目录属于该工具，这也违反本研究不清理系统/应用缓存的边界；全局日志还可能混入其他App的数据。

后面的prediction helper只对第一次调用检查NSError，后续timed calls传error:nil；dummy MLMultiArray未在已读构造路径明确写入确定性数值。因此不能把其time字段直接当作“相同有效输入的成功预测”。

不通过重定向HOME或关闭系统保护绕开这些问题。本轮只执行[自己的公共MLComputePlan客户端](../../labs/ane/compute_plan.py)和确定性实际FFN；上游原工具保持`not-run-unmodified / safety-hold`，未修改参考checkout。

## JSON适配必须拆开三类证据

- 实际prediction wall：只有逐次成功、输入确定、完成边界明确时才可列`measured`。
- weightMB/elapsed、逻辑GFLOPs/elapsed：`derived`，不是DRAM PMU、频率或整数MAC证据。
- MLComputePlan、CostModelFeature、默认边界惩罚、`steadyMs > grandRT*3`判fallback：计划/模型/启发式，标`predicted`，不能称物理实测。

缺失CostModelFeature条目应为unknown，不是零；不能把未记录op排除后计算一个貌似完整的“覆盖率”。没有counter也可依靠完整消费者配对实验优化，但需保留归因限制。

将来若要执行上游实现，应在独立副本中保留许可与patch SHA，关闭删缓存和全局日志，使用已编译的自有小图，确定性初始化输入，并检查每次prediction返回值。原始日志不得加入公开知识库。此页不是对项目所有代码的完整审计。

[固定版本源码](https://github.com/Anemll/anemll-profile/blob/bf8528105a0e61fda2c80d2c9b19615ecab9b7fa/anemll_profile.m)。
