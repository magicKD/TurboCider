# 优化验收与回退：正确、可测、完整请求受益才采用

<!-- knowledge
id: technique-gpu-ane-validation-and-fallback
type: guide
-->

[方法导航](README.md) · [实验与反例](experiments.md) · [来源范围](sources-and-coverage.md)

## 三道独立的门

**语义门：**图变换、布局、量化与状态更新仍计算声明的函数。实数代数等价不保证浮点逐位一致；有损配方需要明确质量预算。

**测量门：**基线足够强，比较对象、时间边界、输入和统计口径可复核。host span、command数、compute plan与物理设备counter不能互换。

**采用门：**真实消费者和完整请求在可接受质量、峰值、失败恢复与维护成本下受益。通过前两道门，不自动获得第三道资格。

## 把反例变成回归

| 容易漏掉的问题 | 最小检查 | 实现案例 |
| --- | --- | --- |
| finite但小值行全归零 | 稠密输入、scale边界、非零参考 | [ccv scale-floor](case-studies/ccv-dense-precision-and-scale-floor.md) |
| 大residual掩盖坏FFN | 同时查完整Y与去residual分支 | [完整paired FFN](case-studies/tinygrad-strong-baselines-and-paired-ffn.md) |
| 融合删除了中间裁剪/舍入 | 比较原中间format与极端值 | [MetalPetal](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md) |
| 首token对，第二步state已错 | 多步递归、fresh/reused、A/B/A | [ANE-LM状态数学](case-studies/ane-lm-programs-state-and-numerics.md) |
| API返回成功却配置不匹配 | token实际ID、config与consumer合同 | [FastVLM元数据](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md) |
| 失败后原输入已被部分覆盖 | poison、部分失败、完整重算/rollback | [ggml-ane失败链](case-studies/ggml-ane-backend-cache-and-failure.md) |

这些案例的执行深度不同，应读各自范围说明，不合成一个“上游故障复现数”。

## 最小对照实验

冻结输入、权重/LoRA、dtype/quant、shape/stride和成熟GPU基线；每次只改一个机制。主实验保留全部样本，固定候选后新进程/反序确认，避免只选最快窗口。

报告同步完成的完整调用，并分开prepare、cold compile、warm、device/wait、join、readback与完整request。总体中位数之比与配对ratio中位数分别命名；不同机器、runtime或窗口不伪装成严格配对。

没有硬件counter就标unknown；CoreML允许CPU/GPU/NE或plan偏好，不是实际物理placement证明。设备测试另需显式准入，不由文档检查自动启动。

## 失败回退是事务，不是warning

回退前确认原输入/权重可重用，所有部分输出和递归state已丢弃或恢复。KV位置回滚不能自动恢复conv/SSM状态。不得把强制signal、scheduled、超时返回当已完成/drain。

资源持有到最后读者真实完成；标记generation，避免地址复用误认旧结果为新结果。不盈利允许回到原GPU，失败候选不得发布最快倍率。

迁移模板见[技巧总览](optimization-playbook.md#五把方法变成一个可验收的小任务)；运行[离线正反控制](examples/README.md)不授予设备性能或模型质量资格。
