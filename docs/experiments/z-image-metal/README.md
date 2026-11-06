# Z-Image Metal 历史实验

这里集中存放 2026 年 9 月的 Z-Image GPU/Metal 实验正文。记录中的源码、设备、
会话号和运行状态只说明当时的实验；归档不会继续运行实验，也不会把待验证项改为通过。
当前产品能力应从[项目文档](../../README.md)和[加速状态](../../status/acceleration.md)进入。

## 先确认哪些证据可用于性能分析

[2026-09-23 性能账本](../../design/validation/z-image-performance-ledger-2026-09-23.json)
将下面 11 篇记录的计时数据标记为无效：它们在重启前异常系统状态下采集。
即使原文标题写有“completed”“promotion”或给出 ABBA 加速比，也只能作为实现、
逐值一致性或诊断历史，不得用于内核选择、加速比、回归阈值或容量判断。

- [重启对照](z-image-reboot-control.md)：同一旧二进制在重启后恢复性能，解释旧计时失效的原因。
- [健康状态复验结果](../../design/validation/z-image-clean-performance-2026-09-23.json)：
  性能账本指定的该次 M4 Max 复验结果；仍受其中设备、源码、分辨率和 workload 限制。
- [M4 Pro 四图 screening](z-image-m4-pro-metal-2026-09-27.md)：独立的 2026-09-27 探索，
  每个方案只有一个 warm 样本，不能与 M4 Max 结果合并，也没有扩大自动设备默认策略。
  [原始摘要](../../design/validation/z-image-m4-pro-metal-2026-09-27.json)保留原始二进制和图片哈希。

## 仅作实现或诊断历史的记录

| 记录 | 主题 |
| --- | --- |
| [Command-buffer 阈值](z-image-buffer-mid-screen.md) | 中间 buffer 上限的探索 |
| [Context 缓存](z-image-context-cache-abba.md) | 请求内不变量复用 |
| [1024 累计对照](z-image-final-default-1024-abba-pending.md) | 先前标为完成的最终库 ABBA |
| [父进程与存储控制](z-image-parent-storage-diagnostic.md) | 独立调度与 Metal storage 诊断 |
| [回归遥测](z-image-regression-telemetry-2026-09-22.md) | 系统异常状态下的只读观测 |
| [复测与算子 profile](z-image-retest-profile-2026-09-23.md) | 重启前的复测与热点分析 |
| [宽 SwiGLU screen](z-image-swiglu-wide-screen-2026-09-22.md) | 未晋级的候选几何 |
| [Vector QKV](z-image-vector-qkv-pending-2026-09-22.md) | 测量后拒绝的候选 |
| [Virtual256 晋级记录](z-image-virtual-default-promotion-2026-09-22.md) | 当时的重构建与派发检查 |
| [Gate/norm 融合](z-image-virtual-gate-norm-2026-09-22.md) | 未晋级的精确融合候选 |
| [Virtual256 ABBA](z-image-virtual256-abba-pending-2026-09-22.md) | 已结束但计时失效的对照 |

## 存储与兼容

正文是唯一的完整副本；旧的 `docs/design/validation/*.md` 入口保留为转发页。
[验收摘要目录](../../design/validation/README.md)中的 JSON、性能账本和哈希保持原样，
避免破坏已有消费者。原文中仅写文件名的 JSON 引用仍以该目录为准，完整相对路径
则按记录的仓库根目录解释。所有原数值、失败结论和历史过程说明均保留。

原始日志、张量和生成图片位于各记录写明的本地 `outputs/` / `/private/tmp/` 路径，不保证随仓库提供。
复现命令应从仓库根目录运行，并核对原记录的源码与模型身份；本次迁移没有重跑模型。
新实验应按[归档约定](../README.md)另建记录，不覆盖这组历史证据。
