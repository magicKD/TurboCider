# 调度实验：合法窗口、overhead、bank与share

[实验室](README.md) · [调度/状态代码](systems.py) · [控制器代码](policies.py)

## 1. 用可执行DAG画真实依赖

```sh
python3 -B techniques/labs/run.py overlap-dag
```

任务记录`id/resource/duration/deps`，scheduler选择依赖已满足且资源空闲的任务。对照串行资源、合法fork-join和带显式slowdown的共享带宽场景；移除attention→FFN输入边会被模型所需edge检查拒绝。

这些duration/slowdown是合成假设，输出timeline不是device trace。把实际solo/concurrent spans代入可以作探索，但不能仅由两条resource lane断言物理overlap。真实GPU/GPU共享event例子见 [Metal](gpu.md)，GPU/CoreML公共worker见 [ANE](ane.md)。

## 2. Slot不是一个可随时覆盖的数组

```sh
python3 -B techniques/labs/run.py slot-ownership
```

`Slot`状态：empty→filling→ready→in_flight→retired；timeout转quarantined但readers仍持有generation。必须所有reader完成才可refill，failed producer不能发布ready，late旧generation不能改新slot。

实际调度还需device ticket/producer/consumer三侧物理完成；state machine不能靠自身布尔值证明硬件已结束。未来实现可用event、callback或显式同步，但不以timeout安全signal冒充结果有效。

## 3. Share不是越大越好

```sh
python3 -B techniques/labs/run.py share-calibration
```

学习实现只对两份已测share内插`GPU/ANE/both`，用complete span和memory-admitted集合选near-optimal小share；更强GPU分母会使同一个候选decline。区间外拒绝推断，意味着需要新增采样点，不是禁止探索新区间。

生产 [ChannelCostModel](../../native/backends/ane_cost_model.hpp)保留带宽不确定性的polygon envelope，比教学内插更保守；实际 [model calibration](../../native/backends/ane_channel_calibration.cpp)进一步建独立executor试验。拟合、proposal、actualshare、完整FFN、整请求分别报告。

`(T4-T1)/3`仅在相同shape/固定开销近似可消时成立；T4≤T1、非有限、不同warm/cold或交接边界不同的证据不适合直接代入。可以改用多点回归/真实trace，而不是放宽输入校验。

## 4. Bucket与cold成本一起选

```sh
python3 -B techniques/labs/run.py bucket-selection
python3 -B techniques/labs/run.py startup-amortization
```

bucket case明确1024image+31text→1056、4096+31→4224、再返回1056；长caption可能跨档。startup case按`setup + N*hot`算crossover，热态收益不能代替one-shot体验。

下一步探索fixedbucket/flexible shapes、multifunction models、load-only、phase-scoped bank、复用weights或runtime模板；每种变化会改变cold/steady/memory，不预设哪一种普遍最优。

## 5. 从lab到真正异构调度的SOP

先固定完整GPUblock → 合法cut和same-boundary GPU消融 → GPU-alone/ANE-alone/concurrent → 完整traffic（W/A stage、LoRA、restore、未来prefetch） → 独立候选试验 → phase策略 → E2E。

用户可探索更深overlap、不同queue、跨layer lookahead；先写下缓冲存活、下一层是否真正ready、失败后如何drain。原先预取全命中仍更慢的case是寻找资源争用的线索，不是“以后不得预取”。
