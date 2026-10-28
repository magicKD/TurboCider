# vpipe 深读：双层控制器、有限输出与失败之后的消费者

<!-- knowledge
id: technique-gpu-ane-case-studies-vpipe-controller-and-failure
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [第一轮方案概述](vpipe-runtime-weights.md) · [续研证据](../../MAINTENANCE.md)

固定版本仍为 `tgo-app-dev/vpipe@c2989a76841888e4d8d208fad4f476c487710962`。本轮不再只依据 header 的设计说明，而是读 [`ane-ffn.cc` 581–977](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L581)、worker 实现，以及 MiniMax-H3 transformer 6410–6510 的实际调用点。源码身份/范围在 [续研清单](../sources.md#runtime-case-studies)；没有执行上游或注入设备故障。

## 1. 第一层：是否启用分担，而不是只调 share

`plan_block()` 在 auto 模式先跑4个 split block，其中前2个不进入启停测量，然后2个 GPU probe；此后选较优方案，每48个 block 测一次另一方案。`_barrier` 标记从积压的纯 GPU 路径切到 measured split/probe 的边界，避免把以前未排空的工作记到当前候选头上。

`decide_()` 对最近9个样本取中位数，用3%迟滞决定启停：probe < 0.97×split 才关 ANE；split < 0.97×probe 才重开。这是项目的经验控制参数，不是通用统计显著性标准，也不是一次 probe 足以证明无干扰。

测量的 section 是 `drain + GPU分支 + finish中暴露的等待`。它不是整个 request；权重 staging 的一部分可以藏在 attention 下，但未被覆盖的部分必须通过 drain/完整请求体现。probe 保留中途提交，因此还应与原始 unsplit GPU 整体对照。

## 2. 第二层：整数 chunk 的负载平衡

`finish()` 分别维护 GPU 每row和 ANE 每chunk的平滑耗时：新估计为旧值与当前值各一半。枚举合法 chunk 数，预测 `max(GPU_rows×gpu_row_ms, ANE_chunks×chunk_ms)`；至少保留1个 GPU row，且只有预测改善超过5%才改变 chunk 数。

这解释了为什么“哪边晚到就增减一个chunk”会振荡：最优连续点落在两个离散配置之间时，每次增加都过冲。这里用预测+迟滞缓和振荡，但线性模型仍忽略形状跃迁、shared-bandwidth和部分固定开销；它只是在线候选选择器。

若某 block 发生 FP16 overflow retry，该次仍真实花了时间，但被排除在启停中位数之外，以免一次性找 scale 的代价永久关掉稳态更快的路径。评估者不能同样从冷请求/用户延迟里删去这次重试：**调参样本筛选与真实请求计时是两件事**。

## 3. 一个 dispatch worker，另一个转换线程池

`AneWorker` 用单个常驻线程承接同步 Core ML 调用，`dispatch` 在前一任务结束前会等待；这避免每block建立线程。转换池独立，按硬件线程数的一半、最多4个线程构造；小于65536元素的转换 inline 执行。

独立池的核心是避免 dispatch 任务调用 `parallel_for` 时等待自己。它不是“ANE硬件绝不能并发”的证明；单 outstanding 是这个运行时的资源和模型策略。转换线程数量、threshold、CPU/GPU共享带宽需重新测量，不按所有CPU core拉满。

## 4. 精度恢复不是 bitwise 无损

`begin` 逐chunk先将所有输入 BF16→FP16，再预测，再写回同一批row，因此 row-wise in-place 的读写不会覆盖后续chunk的输入。输出存在非有限值时，对 SwiGLU 的up或 GELU 的down projection及其bias缩小4倍，重跑当前chunk，恢复输出scale；scale上限4096。已经写出的早期chunk不重算，使用它们当时的scale。

写回 BF16 时，所读代码取 FP32 bits 的高16位，没有执行 BF16 round-to-nearest-even。例如可由 FP16 表示的 `1.005859375`，截断得到 `1.0`，BF16 RNE得到 `1.0078125`。这是确定的表示差别；不凭此宣称整模型质量失败，但不能继承“与原BF16实现逐位一致”。数值见证见 独立CPU模型。

第五轮又把scale追到emitter、LoRA合成和真实VAE consumer，见[bias/LoRA/headroom](vpipe-bias-lora-and-headroom.md)。SwiGLU的最终bd不随up缩小，恢复整个输出时有条件性bias偏移；GELU缩down及bd则是另一条代数链。上述有限输出与重试机制本身不保证这些语义已经正确。

## 5. 失败之后，谁拒绝发布结果？

在 `begin` 的 worker 中，一旦某chunk预测失败或仍有非有限值就返回，`_ok` 保持false；早先chunk可能已经写回目标buffer。`finish()` 会join、警告并返回false。

本轮读到的 MiniMax-H3 transformer 调用点却是 **`(void)_ane->finish(...)`**，随后继续 gated/residual。它确实检查了**启动前 staging 失败**并保留全GPU，但不能由此推断**启动后部分输出失败**也有同样回退。

这是该commit/调用点的静态观察：没有实机故障注入，不断言整个项目所有入口都有同一缺陷。可以确定的是，所读链路不足以证明“失败结果不会进入下游”。新的迁移实现必须检查返回值、使本代输出失效，并在安全drain后选择中止请求或完整重算；打印warning不等于事务提交门禁。

这个调用点还是`begin(s.nm,s.nm,rows)`的in-place形式：成功的早期chunk已经覆盖原norm输入。因此GPU重算不能直接把部分覆盖后的buffer当原输入使用；须保留/重建producer结果，或者中止请求。数学上row独立解决不了这种失败恢复的状态依赖。

## 6. 可迁移设计与最小见证

值得学习：持久worker、独立转换池、on/off与share分两层、离散chunk预测、有限buffer、分阶段计时。不能直接继承：固定threshold、用全局wired增量作精确归因、截断精度配方、未检查的失败返回值。

`runtime_case_studies.py` 只在CPU上演示迟滞、chunk平衡、BF16截断差异和partial-write发布拒绝；它不是移植上游实现，也不是上游故障复现。真实设备验证还需取消/异常/drain、每个chunk的generation、完整GPU分母和整请求质量。
