# Delta 权重重载与 LoRA-as-input：Orion 的两条免重编译路线

[ANE 目录](README.md) · [训练](training.md) · [动态参数](training-dynamic-weights.md)

固定 [Orion@4dcc1c2a](https://github.com/mechramc/Orion/tree/4dcc1c2a626c9ee1f35576bd0b290d489db17d08)，阅读 [ane_runtime.m](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/core/ane_runtime.m)、[delta tests](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/tests/test_delta_compile.m)、[LoRA frontend](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/compiler/frontends/lora.c)的选定范围。本轮未执行原 runtime；不是对全仓库的正确性/安全认证。

## 1. 两种 delta 操作应分开

`orion_program_reload_weights` 对已存在 model 做 unload、修改权重文件、再 load，不再次调用 compile。`orion_program_patch_weights` 则建立新 descriptor/model，借 donor 的 `net.plist` 与编译布局，并写入新 blob/data，尝试直接 load。两者都依赖私有产物行为，不是 Apple 提供的稳定 weight-update 合同。

优势是保留常量图的计算结构，减少编译；代价仍含 unload/load、文件写入、FP16 blob 准备和状态切换。上游 README 的约8.5×指重编译/重载阶段，约3.8×指指定训练流程，**不是 ANE 算术吞吐提高3.8×**；本轮不复用这些数字为本机结论。

## 2. 源码里需要额外验证的假设

- patch 路径把第一个 weight blob 也写成 compiled `data`，多 blob 对应关系依赖实现假设；不能推广为任意常量折叠、压缩和布局后的权重都可原样替换。
- 当 framework identifier 相同，donor 的临时目录 owner 会转移；不同 identifier 时会删除/重建目标目录。执行前必须建立任务目录归属，而不是盲信一个字符串 hash 是可安全删除的私有路径。
- reload 路径所读 unload 返回值未作强 gate；文件写入也需检查失败、完成、并发 reader 和代际一致性。应先 drain，失败时整代无效，不能继续用半更新图。
- tests 比较 patched 与 fresh compile、检查 compile counter 不增加和多次 patch；这是好的控制意图，但不替代独立 CPU oracle、finite gate、全部 blob 和真实训练/LoRA 组合测试。

因此本轮保持 `source-inspected / upstream-not-run`。不将此方法直接加入 TurboCider immutable artifact 流程；两者的可变性合同正相反。

## 3. LoRA-as-input：冻结 base，只传小矩阵

对行向量约定，`Y = XW + (alpha/rank)*(XA)B`。base W 可是常量 Conv；A/B 是 runtime input，编译图 shape 固定，换 adapter 不需重新编译 base。所读 frontend 将 BC1S 与 matmul 维度来回映射，把 adapter 分支加到 base 输出。

这是参数表示与热切换方案，不自动等于 LoRA 训练已完成，也不等于所有 transpose 零成本。rank、实际 A/B 轴与 strides、alpha/rank、bias、原权重量化和输出 dtype 都须绑定 recipe。多个 LoRA 或不同 rank 可能需要新图/新 bucket。

## 4. 与全参数动态图的取舍

| 方法 | 每次变化的数据 | 更适合的候选场景 | 优先否决条件 |
| --- | --- | --- | --- |
| 全参数 runtime weights | 大部分/全部 W | 高频全参数更新、同形 kernel 复用 | 权重 staging 成为主成本 |
| delta reload | 常量 blob + runtime 重载状态 | 想保留特定常量 kernel 映射 | 新参数不生效、缓存别名、格式假设失效 |
| LoRA-as-input | 小 A/B，base 不变 | adapter 切换/低秩微调 | rank/axis 错误、base 非线性改写、质量预算失败 |

应共同测试 `base→A→B→base`，并用不同 adapter 的独立期望输出防 stale；另做取消、更新中失败、并发 session 和 cache 失效。切换速度、训练 update 速度、前向热调用和完整模型质量分别报告。

注意 Orion README 保留了“INT8 都先解成 FP16”的泛化表述，而官方 M4 W8A8 文档支持更窄、更准确的结论；已在 [INT8 专题](w8a8-int8.md)纠正，不能因其 runtime 代码有价值就接受全部硬件解释。

## 本地学习入口与复查

- [ane/training_math.py](../labs/ane/training_math.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [caches.py](../labs/caches.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py ane-training-math`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
