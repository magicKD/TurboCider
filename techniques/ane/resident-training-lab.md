# 驻留训练源码与实测：VJP、参数代际、绑定时序和独立完整验证

<!-- knowledge
id: technique-ane-resident-training-lab
type: method
operator: gemm, layout, scheduling
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: precision-loss, lifetime, measurement
semantics: contract-only
representation: fp16, fp32, iosurface
boundary: producer-consumer
api: reference-only
-->

[ANE目录](README.md) · [训练路线](training.md) · [本轮固定源码](sources.md#implementation-sources-20261016)

这是上一轮公共 Core ML 手写反向实验之后的深入验证。上一轮 matmul 计划首选 CPU，不能证明 ANE 训练；本轮直接调用固定 ANEForge/e5rt 路线，逐程序核对请求 mask=0x4。**请求设备仍不等于物理 trace，结果不称通用 ANE 训练框架、LLM 收敛或训练速度胜出。**

后续另行完成了[匹配K步窗口](matched-training-windows.md)，用无逐步observer/写盘的共同host checkpoint边界比较resident、device、原创SGD四步展开和编译GPU。那是独立新批次；本篇旧API跨度仍不构成速度对照。

2026-10-17另有[Adam非线性VJP、完整moments与状态恢复](adam-moments-loss-scaling-and-checkpoint.md)：48进程246次显式step（含72次恢复续步），默认epsilon有32份source-reference更新拒绝但全部finite；完整W/m/v/t恢复24次复现、48次错误恢复不复现。那是独立批次，不回填为本篇线性SGD已验证Adam，亦没有新增训练速度/跨进程checkpoint资格。

## 1. 从 Python 参数追到 native buffer

固定 `sbryngelson/ANEForge@896c4943c5ebf811c0e815f8d1381fd1e157fa7d`。主要入口：

- [autograd.py：Trainer](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py#L727)：默认编译 loss forward 和每参数 backward；device optimizer 再增加更新图；resident 把梯度和更新接进 fused multi-output 图。
- [runtime.py：share_buffer/set_input/execute/read_output](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_runtime.py#L181)：前者负责绑定，后面区分 seed/feed、执行与显式 checkpoint readback。
- [native share_buffer](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_lib/ane_e5rt_dispatch.mm#L687)：将 destination input port 绑定到 source output 的同一 e5rt buffer object。
- [finalize_program](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_lib/ane_e5rt_dispatch.mm#L476)：首次执行前才 encode，随后固定拓扑。**alias 必须在 finalize 前建立**，否则显式拒绝。

本例对应状态流：

```text
初始W写入参数input → output W_next绑定回同一input buffer
          → feed X/target/lr → 同步execute → 下一代W
          → 再feed新数据 → 同步execute → 下一代W
```

不再每步重传 W，不意味着 W 永久在片上 SRAM，也不等于编译器支持任意 in-place 算子。这里是一个 runtime buffer feedback 合同；同步、图内读旧写新顺序和类型必须正确。

## 2. native 所有权：为什么不会在正确路径双重释放

`port_alloc_and_bind` 对输入/输出分别分配至少64 bytes、按64 bytes取整的 buffer；这是 wrapper 分配策略，不是设备 tensor stride 的完整规格。share 时先尝试 bind，成功后才释放 destination 旧 buffer，把 destination 的 data pointer 指向 source，并清空 destination 的 owning buffer 字段；source 负责最终释放。

失败不先破坏旧绑定，是可迁移的“先建立新状态，再提交所有权转移”技巧。但所读 alias 接口没有完整的 typed shape/dtype 等价检查，调用方仍须保证相同大小/布局。**本轮只使用同shape FP16参数，不执行错size、UAF或跨dtype alias探针。**

普通 `Model.__call__` 返回新的 FP32 数组；`input_view/output_view` 是借用视图，不自动持有 Program owner。resident 参数 checkpoint 使用 `read_output(...).copy()`，而非释放 Program 后继续读取裸 view。

## 3. 三臂实际计算了什么

| 模式 | 图与参数更新 | 本轮检查 | 不自动成立的结论 |
| --- | --- | --- | --- |
| host | fwd/backward请求e5rt；host FP32 master SGD | 完整梯度、完整W、标量forward loss | optimizer算术也在ANE |
| device | 单独backward与更新图；参数每步回到host再喂入 | 完整梯度、完整W、标量forward loss | 没有状态往返、等同host FP32配方 |
| resident | 梯度+更新fused；更新输出alias回参数input | 完整W、更新delta、标量forward loss | 直接观察到了resident内部梯度或片上驻留 |

host/device 的原始梯度被原创 adapter 在调用原 optimizer 前记录和验收，没有修改上游源码。resident 不暴露梯度，证据中对应字段为 null，不用另一次 backward 冒充该 fused step 的真实梯度。

## 4. 本机实验与独立复核

M4 Max/64GiB、macOS26.6.2、Python3.11.16、NumPy2.0.2，沿用此前已构建且绑定hash的隔离 shim。仅32×32线性模型，两个独立seed79/83，每臂8次SGD，lr=0.5、loss scale32。损失是 `0.5*sum((XW-target)^2)/rows`，因此梯度归一化不能误写成除以全部元素数。

所有6个进程正常结束、release返回、选定上游源码hash前后相同；图/模型数量为host2、device3、resident2。每种子各模式的初始 X/target/W 相同；第二种子反向模式顺序。未下载模型、安装依赖、改系统保护或开启异步私有入口。

| seed | host final/initial loss | device | resident | device/resident完整W |
| --- | ---: | ---: | ---: | --- |
| 79 | 0.139653 | 0.139599 | 0.139599 | 8代NPY内容hash全部相同 |
| 83 | 0.153773 | 0.153708 | 0.153708 | 8代NPY内容hash全部相同 |

总计48次optimizer update，32份真实完整梯度、48份完整参数、54次实际标量forward loss。device/resident同一FP16-state SGD配方的16代参数一致；**不宣称与host FP32 master bitwise相同**。

独立解析有界NPY，以 Python `math.fsum` 重建每一步完整32×32梯度、参数和loss，不导入NumPy/ANEForge、不重新跑设备。除采集时的2%梯度/参数门槛，还额外检查更新delta的相对误差，防止“W本身很大，完全没更新也能通过小相对误差”。实际完整梯度最大relative L2约0.000521，完整参数约0.000310；具体逐run值在JSON。

测试覆盖有限但过期的W、伪造为全零的梯度、输入hash变化、非有限尾元素、NPY类型/长度、bool时间、重复/缺失代际、设备mask或gate被改。损失下降本身不是验证器唯一依据。

## 5. 为什么本轮不发布resident训练提速倍率

resident 每步后额外读回W做完整独立验证，loss也单独执行。host/device 的梯度guard以及梯度NPY写盘都在 step span 内；resident 的审计readback/参数写盘在该span外。三臂 `step_api_median_ms_diagnostic_only` 实为被探针instrument的跨度，计时工作量不完全相同，不能直接相除生成“训练快N倍”。compile/load约75–97ms同样没有消失。旧runner快照保留当时较简略的scope字符串，本页明确补充实际边界，不重新标记旧时间。

要评价驻留的真实性能，应冻结数值配方后设计“相同K步+相同最终checkpoint验证”边界，并保持minibatch、RNG、loss scaling、有限性/失败恢复能力一致。本轮也未持有SoC实验排他锁或连续记录热/功耗，因此API跨度不作为无竞争稳态保证。本轮未做Adam、长时收敛、真实训练数据、能耗或完整LLM对照。

最早3个无v2后缀的进程只记录梯度/参数，没有实际scalar forward检查；保留为diagnostic predecessor，不回填成54次forward中的一部分。8个step也不应换称8个LLM训练batch。

## 6. 复现与继续探索

原创显式集成：aneforge_training_probe.py。需先按[既有ANEForge集成](../examples/README.md)准备固定源码隔离副本和shim，不自动下载/构建；独立监督进程限制150秒。例子：

源码、SoC/OS、graph、shape/dtype、lr/loss-scale/optimizer、alias生命周期改变时重验。后续Adam批次单独验证小图moments和同owner完整恢复，并保留default失败；同精度K步性能、overflow原子恢复、跨进程完整checkpoint与真实任务仍开放，不拿本轮线性SGD结果作它们的资格。
