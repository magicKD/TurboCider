# coremltools：计算计划不是硬件逐算子计时

[来源目录](../README.md) · [测量工具](../../profiling/tools.md) · [公共探针](../../labs/ane/compute_plan.py)

来源：`apple/coremltools@a207b1a3452eafe02f2d870f486287834af2c82d`，`coremltools/models/compute_plan.py`全文448行，工作树干净。SHA-256：`977c84697df762a1958bd1c8e562db8e9cb95270254508983e11519c57f8e497`，2026-10-15。本机实际执行的是已安装coremltools 8.3.0；该Python文件与参考checkout内容相同，不代表两套包的native binary或其他源码完全相同。

## 返回值的真实语义

`MLComputePlan.load_from_path`调用native proxy获取指定compiled model与compute-unit政策的计划。program含functions、block、operation与嵌套blocks，不能只统计main第一层。

`get_compute_device_usage_for_mlprogram_operation`返回anticipated preferred/supported设备，可能为None；`get_estimated_cost_for_mlprogram_operation`返回相对成本weight，可能为None。它不是实测ns，也不能任意乘某一次wall time再称逐op硬件时间。

我们的[归一化探针](../../labs/ane/compute_plan.py)将device/cost标为`predicted`，operation count标为对计划数据的`derived`，缺失值保留null。不会删除系统缓存、抓取统一日志、运行prediction或读取其他App信息。

## 与真实时间关联，但不混为一谈

[tiling_bench.py](../../labs/ane/tiling_bench.py)另测实际predict与完整GPU消费者：相同ready输入、Core ML返回结果、布局恢复、可选GPU suffix、join后`mx.eval`完成。CPU encode、API call、GPU完成不能由plan的cost替代。

一个图所有非const op都计划在ANE，不保证它比compiled GPU快；更小tile也可能增加切片/临时数组与归约。计划用于提出假设，原因需要消融；真实placement或物理并行仍需独立trace/counter。

当前实验中发现有些图能给设备计划却不给任何cost；这些值必须保持unknown，不能输出零成本。计划加载发生在首次prediction前，也可能影响首次调用的缓存状态，所以报告显式记录顺序；不称真正冷启动。

重新验证条件：OS、Core ML/coremltools、graph、shape、dtype、compute-unit policy或编译缓存变化。许可为BSD-3-Clause（LICENSE.txt）；本地探针为原创API客户端。

[固定版本源码](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/models/compute_plan.py) · [Apple计算计划优化讲解](https://developer.apple.com/videos/play/wwdc2024/10161/)。
