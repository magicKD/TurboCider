# 调度：让有用工作重叠，而不是让等待换位置

[总目录](../README.md) · [校准](calibration.md)

## MLX lazy 执行的先后关系

构建 tensor 表达式不等于 GPU 已执行或提交。Hybrid 启动前应明确物化共同输入/修正，提交实际 GPU suffix/head，再让后台 ANE worker运行；必要依赖到位后再 join。提前取下一层权重若挤在 current GPU work 前，可能把当前关键路径拖长。

本仓库保留稳定层 async head、readiness 合并、独立 stage queue；没有采用的 GPU-first/输出恢复并行不能仅因“更异步”恢复到默认。参考 [后端维护说明](../../native/backends/README.md)。

## 四种预取要分别消融

| 机制 | 隐藏什么 | 新增竞争 / 内存 |
| --- | --- | --- |
| disk layer lookahead | 下一层 IO / decode | SSD/page cache、CPU/GPU decode、typed slots |
| W-bank prefetch | 下一层 W8/FP16 staging | GPU queue、UMA bandwidth、第二 W bank |
| A8 lookahead | 后续输入 chunk 的 pack/rotation | 依赖输入就绪、第二 activation slot |
| prepare/load-only | 下一请求 / 模型的 setup | 与当前请求抢内存、hash/compile 工作 |

每种都测 off/on；组合再独立测。hit 数说明机制命中，不说明减少关键路径。

## 已命中但更慢的真实案例

[Private W prefetch](../../docs/status/private-ane-prefetch-2026-10-03.md)：Z512 开启后 744 submissions/744 hits、暴露等待极少，却把该窗口 Private warm 从约 6.603 增到 6.985 s；同窗口 GPU 约 6.99 s。原因为共享工作/资源竞争的解释与观察一致，但没有 device trace 不把竞争归因为唯一硬件原因。默认恢复 explicit opt-in。

这一案例最适合用来反驳“prefetch hit 接近 100%，所以带宽问题已经解决”。

## 自适应控制应该观察完整 block

[RowScheduler](../../native/backends/ane_scheduler.hpp)按 `(layer,rows)` 隔离状态，不让短 decode 的历史调节长 prefill。它会 warm shape、采普通 GPU probe、EMA 更新、周期复测，并用迟滞减少反复开关。当前源码的 0.98/0.95 block ratio 与 32 visits 是该控制器策略，不是所有模型的产品门槛。

关键点：

- `peek_plan()` 只预测，不消耗下一层 visit；预取不能偷偷改变采样序列。
- 新 partition 先排除首次编译/warm，不能用旧 partition 的坏成绩提前否决它。
- 自动禁用时回普通 unsplit GPU block，不带拆分的 timing fence。固定 zero-chunk 消融则刻意保留边界，两者不是同一个分母。
- 稳态不每 block 强制同步；周期诊断才采 full-block wall。
- adapter/source/precision/rows 变化隔离 key；只缓存 CPU metadata，限制 capacity/TTL，不保留 executor tensor。

## 按 phase 决策

Qwen 编辑：首步参考/文本/目标长序列可能适合 channel split；prefix KV hit 后 rows 减少，完整 compiled GPU 可能更快。40步 base 的后39步仍有大量 FFN，不能把六步 LoRA “后续GPU优先”写成所有 Qwen 的固定规则。

不同参考数量、caption rows、cache hit 和 bucket 组合要重测 share。[最新 phase 证据](../../docs/status/performance-progress-2026-10-10.md)说明单/双参考的最优候选比例不同，部分排序反转，保持实验而非普遍自动。

## 成功计数与错误路径

只有真正成功发布完整输出才增 successful/deferred blocks；requested flag、enqueue 和 future hit 都不能代替 actual call。退化/取消/超时不继续拿旧调度收益标签。晚期失败完整 GPU 重算并记录次数；fallback 时间必须留在 E2E。

## 实现参考与实验

可读实现：[参考代码](../labs/systems.py)；[实验步骤与边界](../labs/scheduling.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py overlap-dag
```

本项控制：legal vs missing dependency；serial/two GPU queues and measured timestamp intersection。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：different contention, prefetch frontier, phases or queue schedule。历史负结果限定于其来源/shape/边界，不关闭该方向。
