# Metal timestamp校准：先辨单位，再关联同一trial的窗口

[公共采集](gpu-timeline-and-counters.md) · [来源归一化](trace-normalization.md) · [官方合同](../sources/official/metal-counters.md) · [本机证据](../review/gpu-clock-and-qwen-20261016.md)

## 单位不是由字段名决定

上一批raw counter保留null秒数，因为没有独立单位参考。新增[labs/metal/bench.mm](../labs/metal/bench.mm)在`sampleTimestamps`前后同时读取`mach_absolute_time`和`clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`，并记录实际`mach_timebase_info`。数值必须落在实测区间，不能因“都叫timestamp”就合并时钟。

[calibrate_metal_timestamps.py](../scripts/calibrate_metal_timestamps.py)比较两个CPU单位候选：纳秒或Mach ticks；timebase=1导致数值等价时明确标注。只在两端参考一致时，计算GPU raw delta到CPU ns的有理数比例。转换后的值标为`derived`，原始counter仍是`measured`；不把猜测单位或其他进程时钟塞进来。

本机新数据的CPU候选为ns、Mach timebase为125/3，GPU映射比例观测为1 ns/raw tick。但这是这些窗口/配置的验证，不是所有Apple GPU的恒定周期。若对已经是ns的数再应用125/3，会重复换算；不能套一个网上通用公式。

## 两点配对还不够

转换后还要检查每个stage是否落在**同一trial、同一个command**的完成时间窗口内：

- 一encoder/一CB：包含关系加duration一致性检查。
- 多encoder/一CB：只检查各stage被CB包含，不把CB总时长逐个复制给所有stage。
- 多CB/逐层wait：按编码时登记的slot关联，不按label或排序猜关联。
- 缺参考、sentinel、0/非递增counter、过宽bracket、未知source或不支持的关联保持未通过。

固定门槛为参考bracket≤1ms，端点绝对余量2µs，一stage一CB的duration差≤max(2µs, 5%)。全部误差都保存，不靠改变门槛找冠军。完成CB时间与counter不是独立物理传感器；通过仅说明公开API读数和时钟映射一致，不等于获得ns级绝对精度、occupancy或跨设备资格。

## 采集器本身也需要负控

最初clock smoke把可变`completed_commands`数组直接放入多个sample，后续trial清空/写入使旧记录全指向最后一个窗口。单位依然辨为ns，但7/8窗口关联失败。修复为每trial发布不可变copy后，新smoke和规模控制通过。该问题只影响最初新增的clock-reference-v1，不回写或否定此前未保存这些窗口的raw-counter记录。

因此必须快照事件数据/owner，而不只是保留指针；最后一个trial碰巧正确不能让整批通过。测试同时覆盖错误窗口、重复换算方向、缺参考、sentinel与historical不回填。

## 可复现实验与复用边界

```sh
python3 -B techniques/labs/metal/run.py graph --count 262147 --steps 16 \
  --diagnostic counter-stage --output outputs/my-new-clock/report.json
python3 -B techniques/scripts/calibrate_metal_timestamps.py \
  outputs/my-new-clock/report.json --output outputs/my-new-clock-normalized.json
```

本轮12份记录含1个有意保留的采集失败，11份通过记录共716个stage窗口；GEMM64/128/256、vector三个规模、graph1/4/16/32阶段都保留。GPU时间无需与FLOPs或次数严格线性，cache/occupancy/频率/调度会变；首先验证采样数量、包含关系和单位。

旧数据没有bracket就继续为null，当前通过也不自动适用于MLX内部queue或ANE。完整原始来源/转换数据在[clock evidence](../review/gpu-clock-evidence-20261016.json)，默认CPU检索和测试不运行设备实验。
