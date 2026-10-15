# GPU trace归一化：来源、单位、端点与资格分别保存

[工具地图](tools.md) · [公共counter实验](gpu-timeline-and-counters.md) · [gputrace源码卡](../sources/repos/gputrace-timing-provenance.md) · [完整本机证据](../review/gpu-profiling-20261016.md)

## 为什么不能把JSON数值直接当GPU实测

本机受控native capture确实包含1、1、16次dispatch；同一parser的timing却按库函数名给出11ms、2ms、2ms的synthetic总量。MLX库名录更产生25.731秒，而其实际小型运算的host-ready样本只有亚毫秒量级。两者不是同一测量边界，不能相除成“profiler overhead”，更不能用synthetic排序指导kernel替换。

| 输入字段/来源 | 可保存的事实 | 必须保持未知的东西 |
| --- | --- | --- |
| native完成后的GPUStartTime/GPUEndTime | 指定command buffer的host-seconds时间窗口 | ANE时间、request全时长 |
| public timestamp counter | 本设备原始timestamp与采样点、sentinel检查 | 未校准的seconds与跨域overlap |
| gputrace `extracted` / `store0` | exporter的heuristic观察与声明单位 | 物理kernel elapsed time |
| gputrace `synthetic` | 根据函数名生成的预测值 | 执行次数、真实性能排序 |
| gputrace `profiler` | 上游声称的profile duration | 未独立核对的时钟、端点、原请求性能 |

profiler duration可能来自真实profile，但该出口仍可能重建串行端点；adapter没有凭空补齐这种信息。形状、设备、实际dispatch数缺失时保持null，不从kernel名字猜测。

## 只读适配器

[normalize_gputrace.py](../scripts/normalize_gputrace.py)读取明确指定的`timing --json`文件，不读取整个trace、不启动CLI、GPU、viewer、下载或源码扫描。

```sh
python3 -B techniques/scripts/normalize_gputrace.py "$TC_TIMING_JSON" \
  --parser-commit 3d7d32e66605ba994fb0765bf9f287682bd7ccc7 \
  --output outputs/my-new-normalized-timing.json
```

已审读版本的synthetic/extracted/store0标为`predicted`；即使输入把approximate设成false也不会晋升。profiler与未审读版本保留`unknown`及上游claim，等待独立验证。物理GPU秒数始终null；空输出或零duration是缺失，不是“GPU用了0秒”。Go `time.Duration`数字的声明单位与硬件时钟验证分开。

native报告另用[labs/summarize.py](../labs/summarize.py)描述已验证输出的host边界；该入口保留`diagnostic_mode`与正式分数标记，拒绝源码在运行期间变化。它不把profiler JSON或raw tick当作同一种报告。

默认只导出白名单字段，不带metadata UUID、device ID、queue地址或本机trace路径；路径型、UUID型与明显credential型label被遮蔽并保留hash。该过滤不等于通用匿名化，发布前仍需检查自由文本。

## 与真实证据绑定

[curate_profiling_campaign.py](../scripts/curate_profiling_campaign.py)显式选择run ID，复核源码快照、capture内容hash、全部native样本、输入配对、counter sentinel与失败receipt。大MLX名录在公共证据中只保留有界前缀、明确省略数量、完整raw输入hash与规范化内容hash；没有把几万个猜测值包装成物理trace。原件留在ignored输出中。

复用[知识查询](../scripts/README.md)的`source_ids` / `evidence_ids`进入源码和实验，不由查询触发任何采集。反例测试在[test_gputrace_normalization.py](../tests/test_gputrace_normalization.py)，同时覆盖false marker、未知版本、sentinel、空结果、隐私字段和profiled ratio拒绝。

重开条件：支持实际streamData、独立timestamp单位/端点见证、parser commit或capture格式变化。只有扩大验证范围，才能扩大`measured`含义；不能只删除warning。
