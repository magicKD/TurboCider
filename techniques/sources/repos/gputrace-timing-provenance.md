# gputrace：时间来源比漂亮时间线更重要

<!-- knowledge
id: source-upstream-gputrace
type: source
-->

[来源索引](../README.md) · [归一化合同](../../profiling/trace-normalization.md) · [本机证据](../../metal/case-studies/capture-counters-and-failure-boundaries.md)

固定读本为[tmc/gputrace](https://github.com/tmc/gputrace/tree/3d7d32e66605ba994fb0765bf9f287682bd7ccc7)，MIT，Copyright 2024–2026 Travis Cline。SHA与阅读范围登记在[sources.json](../README.md)。已有checkout只读；构建使用任务隔离副本、独立Go module/build cache与`GOTOOLCHAIN=local`，不安装全局CLI或App。

## 按入口追到真正来源

- `internal/timing/metrics.go`的`TimingMetricsExtractor.Extract`优先读取profiler，随后`ExtractTimingData`，最后`GenerateSyntheticTiming`。`TimingSource.IsApproximate`将所有非profiler来源标为近似。
- `internal/timing/timing.go`在label附近按固定偏移扫描“像Mach timestamp”的整数，并用范围/字节模式过滤；字段叫`DurationNs`并不证明扫描到的整数属于同一时钟或真是ns。
- `internal/timing/store0.go`同样明确为heuristic：扫描大整数，按相邻位置成对，按序猜kernel关联。不是硬件counter语义证明。
- `internal/timing/synthetic.go`按名称赋常数，例如包含`gemm`的名字取5ms，并添加人为gap。它遍历`KernelNames`，不是物理dispatch记录。
- `internal/trace/trace.go`的`extractLabels`会把device resources和MTLB库函数加入`KernelNames`。库中存在函数不等于本次执行它；本机甚至出现identifier-like字符串被列为kernel，公共归一化输出已遮蔽。
- `internal/mlxprof/gputrace.go`的`selectGPUTraceTimings`还尝试store0；该读本中只有synthetic分支显式设`approximate=true`。因此不能只相信不同CLI/pprof出口的一个boolean。

即使选到profiler，`metrics.go`也可能把各encoder的duration从0开始串起来生成start/end。duration来源与端点/并发关系必须分别验收；有真实duration不等于拥有真实overlap时间线。profiler-only CLI转换还可能缺少完整source标记，adapter不能猜。

## 真实执行了什么

Go 1.26.5 / darwin arm64下，固定源码CLI构建成功，`internal/timing`与`internal/trace`两个package测试通过；非verbose日志不能证明所有fixture或counter测试都运行，也不是全项目验收。

实际只运行了`stats --json`、`timing --json - --table=false`及版本读取。三份原创native capture中解析出的dispatch数量为1、1、16，与受控程序一致；所有timing来源仍是synthetic。MLX捕获还展示了空数据成功退出、库函数名膨胀与边界同步差异，见[完整实验](../../metal/case-studies/capture-counters-and-failure-boundaries.md)。

未运行`xcode-profile`、replay、`clear-buffers`、GUI或权限修改。`timeline`和`FromGPUTrace`默认会建立shader source mapper并扫描内建路径，故这轮也没有直接调用；未来需先有显式受限路径策略。普通离线解析与Xcode自动化是不同权限边界。

可复用实现是严格只读adapter，不是复制整个profiler。固定版本变更、capture格式改变、取得真实streamData或经校验时钟映射后，应重新审读并扩大证据，不沿用当前资格。
