# 调度：让有用工作重叠，而不是让等待换位置

<!-- knowledge
id: technique-gpu-ane-scheduling
type: method
stage: pipeline, ffn
operator: scheduling
backend: mlx, metal, coreml, private-ane
device: cpu, gpu, ane
symptom: straggler, lifetime, memory-bound
semantics: contract-only
boundary: gpu-ane, producer-consumer
api: public, private
-->

[总目录](../README.md) · [校准](calibration.md)

## MLX lazy 执行的先后关系

构建 tensor 表达式不等于 GPU 已执行或提交。Hybrid 启动前应明确物化共同输入/修正，提交实际 GPU suffix/head，再让后台 ANE worker运行；必要依赖到位后再 join。提前取下一层权重若挤在 current GPU work 前，可能把当前关键路径拖长。

“等待 ready event 的值”也不同于“新建一个等待 ready 的 command buffer，再等它完成”。若同一队列已提交 `upload → signal ready → wait ANE done / restore`，尾部新加的等待仍排在 restore 后面，会把 output 依赖一起挡在 GPU complement 提交前。[Qwen 组合案例](../../docs/status/qwen512-qkv-private-combo-2026-10-18.md)用直接 event-value 等待消除了这一额外串行边界，但没有证明等待本身有稳定净收益。检查实际 enqueue 顺序，保留真实 signal、成功/physical-completion 与所有权门；event 提前被观察到不等于物理引擎并行。

本仓库保留稳定层 async head、readiness 合并、独立 stage queue；没有采用的 GPU-first/输出恢复并行不能仅因“更异步”恢复到默认。参考 [后端维护说明](../../native/backends/README.md)。

## 四种预取要分别消融

| 机制 | 隐藏什么 | 新增竞争 / 内存 |
| --- | --- | --- |
| disk layer lookahead | 下一层 IO / decode | SSD/page cache、CPU/GPU decode、typed slots |
| W-bank prefetch | 下一层 W8/FP16 staging | GPU queue、UMA bandwidth、第二 W bank |
| A8 lookahead | 后续输入 chunk 的 pack/rotation | 依赖输入就绪、第二 activation slot |
| prepare/load-only | 下一请求 / 模型的 setup | 与当前请求抢内存、hash/compile 工作 |

每种都测 off/on；组合再独立测。hit 数说明机制命中，不说明减少关键路径。

即使只有同一组bank，也要单独比较producer的提交前沿：GPU complement前、
hidden就绪后和整个complement提交后不是等价时机。前移可能缩短下一operator
的weight join，却拉长当前分支；必须用包含两侧真实完成的依赖consumer裁决，
不能把不同策略各自最短的wait拼成理想收益。原输入/权重、future身份和完成门
保持不变，调度fixture若使用合成的后继输入映射，应与自然模型forward资格分开。

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

固定计算图与按phase调度并不矛盾：同一row-independent图接受动态权重，长序列
以固定chunk重复调用，GPU接余数；控制器仅缓存有界CPU metadata。短于chunk
而没有合法share的调用应在stage/drain之前跳过，也不将它的GPU probe混入长
prefill的section历史。一个已armed的模块在短decode上可能零prediction却仍有
同步成本；见[vpipe后续源码对照](../gpu_ane/case-studies/vpipe-runtime-weights.md#后续源码对照固定图不等于固定控制状态)。

跨请求prefix复用与计算图缓存也应分开：保留整个Transformer，和只导出紧凑K/V
叶值，是不同的owner/内存合同。runtime回调须先真实drain并解除，再导出值；
bank同时检查算术配方与conditioning身份，不能只用shape或一个recipe digest。
普通GPU不应隐式消费runtime算术的prefix。另须沿公共module准入检查到底：
Session里存在一个调用点，不代表真实请求被允许；被拒绝的组合是负控，不是
速度样本。实现例见[显式runtime-prefix组合](../../docs/status/qwen-base512-runtime-prefix-2026-10-18.md)。

还要固定cache命中后的精度路由：省掉prefix后rows可能进入新的ANE准入区间，
使首步新增W8投影。cache代数虽独立，整个consumer的数值路径却已改变。用
off、cache＋原分工、cache＋保留首步算术三arm隔离，核对实际GPU credit/预测数，
不要只保持总steps或总evaluate数。拒绝计划在stage前走普通GPU，同shape的
compiled-block key也必须区分外部projection；policy进入prefix recipe。当前
[Qwen首步对照](../../docs/status/qwen-base512-text-prefix-qkv-step0-2026-10-19.md)
在有限样本恢复exact，不外推为所有prompt/shape/硬件的普遍无损证明。

## 成功计数与错误路径

动态权重/固定shape不自动意味着禁止热重建：溢出后的headroom修复也可能创建
新program。应将图形状、错误策略和owner身份固定，在正常、数值错误、准入失败
及同配置下一请求分别验证。Program构造计数与实际compile/load API是不同证据；
缓存命中也可能仍有replacement load。失败重算完整operator，不能发布部分
scratch；保存失败状态可避免下一请求反复初始化。代价是后续温和输入也可能
暂时走GPU，恢复边界必须显式说明。见[固定Program失败合同案例](../../docs/status/qwen-base512-fixed-program-2026-10-18.md)。

只有真正成功发布完整输出才增 successful/deferred blocks；requested flag、enqueue 和 future hit 都不能代替 actual call。退化/取消/超时不继续拿旧调度收益标签。晚期失败完整 GPU 重算并记录次数；fallback 时间必须留在 E2E。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/systems.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：legal vs missing dependency；serial/two GPU queues and measured timestamp intersection。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：different contention, prefetch frontier, phases or queue schedule。历史负结果限定于其来源/shape/边界，不关闭该方向。
