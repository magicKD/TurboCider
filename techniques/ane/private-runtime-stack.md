# 私有 ANE 运行时：MIL、_ANEClient、e5rt 与模型引擎

[ANE 目录](README.md) · [现有隔离合同](private-ane.md) · [图搜索](graph-rewrite-search.md)

## 三个层次，不能统称“绕过 Core ML”

| 层次 | 代表 | 输入、构建与执行 | 值得借鉴的机制 |
| --- | --- | --- | --- |
| 最小 low-level 探针 | maderix/ANE | MIL 文本/weight blob → `_ANEInMemoryModelDescriptor` → compile/load/evaluate → IOSurface | 能力最小化、动态权重、调用固定税 |
| 图 DSL/编译与搜索 | ANEForge | graph → emitter/blob → 私有 Espresso/e5rt shim；另有 native/bridge 路线 | 合法变换、编译缓存、失败缩减、VJP |
| 模型运行时 | ANE-LM、ane.cpp/Field Guide、Orion | 以上机制 + 权重格式、融合、KV、CPU 算子、采样、session | 将局部吞吐转为模型级可用性 |

不走 `MLModel.predict` 仍可能复用系统 MIL/Espresso/ANE 编译与驱动。减少包装不等于拥有公开 ISA、绕过所有编译器限制，或所有算子都在 ANE。私有 framework 的无特权用户态调用，也不是发行兼容保证。

## maderix/ANE：先理解原语和参数传递

路线图的 [MrZaKaRiA/ane](https://github.com/MrZaKaRiA/ane) 是 [maderix/ANE](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29) 的fork，不是重定向地址；本次选到相同commit，因此不作为两个独立结果互相佐证。重点链路是 MIL→descriptor→compile/load→request→同步 evaluate→读 surface。

动态 matmul 将 activation 与权重沿 spatial 轴合并输入，再 slice/reshape/transpose；可绕开常量权重每次更新重编译，但增加 runtime 权重 IO。见 [动态参数训练](training-dynamic-weights.md)。低层 benchmark 的固定 compile 次数、SRAM 拐点、单输入限制都只能保留为特定版本观察。

## ANEForge：e5rt、内容寻址和图语义

固定 [ANEForge@896c4943](https://github.com/sbryngelson/ANEForge/tree/896c4943c5ebf811c0e815f8d1381fd1e157fa7d)的已读链是 `graph → _compile → MIL/blob → E5RT → ane_e5rt_dispatch.mm`。本库已有 [源码卡](../sources/repos/aneforge-rewrite-search.md)及 [13组推理实验](../review/aneforge-20261016.md)。

它的结构性能缓存不包含具体权重值，编译目录则绑定 MIL/blob；二者不能当同一资格。默认 Model 调用包含输入转 FP16/copy、同步 execute 和输出 FP32 新数组，`Program` 的借用 view 则复用内部内存；比较时必须冻结边界。

本次补读 [autograd.py](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py)发现训练与驻留 optimizer 已有实现，不能只按早期 README 把它当推理 wrapper；详见 [训练报告](training-autograd.md)。没有在本轮复跑训练或全部 native bridge，不将既有 e5rt 推理结果授予这些路线。

## ANE-LM：融合与常量图缓存

固定 [runtime.cpp](https://github.com/johnmai-dev/ANE-LM/blob/04b1af12aa30b3f280f511d286398469764b4326/core/ane_runtime.cpp)有单投影、fused-2/3 和完整 SwiGLU FFN 的 MIL 生成器。常量 Conv 避免逐 token 传全权重；融合减少 prediction/dispatch 和中间往返。

加载层通过 framework model identifier 和 marker 尝试持久 cache；cache load 失败才重新编译。matvec 入口将输入 float 转 FP16 写在每个 channel 的 spatial lane，再同步 evaluate 并读回输出。它不是本轮 GPU-resident 零拷贝实现；缓存、原 BF16→FP16、未初始化 lane/tail 和失效身份都需独立验收。

本轮只读该 runtime 的选定范围，未运行完整 Qwen 推理，也未审核所有模型层 CPU/ANE 分工。README 支持型号清单不是跨 OS 兼容测试。

## Field Guide / ane.cpp：为何模型引擎还要保留 CPU

[Field Guide@83e73eb9](https://github.com/skyfallsin/apple-neural-engine-field-guide/tree/83e73eb94051dc67431829d0f3d3eb67db0b8e9b)在 M3 Max 上记录 runtime conv、tile、attention、RMSNorm、融合与批处理的成功/失败，并把保留方案移至 ane.cpp。重点不是其某个 tok/s，而是：先 probe，再完整引擎确认；对不合算的 attention/norm 保留 CPU；通过 W-lane 合批提高 aggregate throughput。

其“W≥32、tile 使进程后续失败、runtime Conv eval 失败、约119µs+bytes/78GB/s”的模型都是所测 M3 Max/特定路径经验，不能改写成 M4/M5 硬件定律。多用户总吞吐与单流 token latency 也不能互换。完整负结果学习见 [negative-results](negative-results.md)。

## 采用前的统一门槛

有界源码/图/shape、private opt-in、独立 output/cache、逐次完成与数值检查、真实 stride/owner、timeout quarantine、明确 fallback 标签。不要关闭 SIP/AMFI、申请未知 entitlement、清系统缓存或执行上游 README 的安装/清理脚本来“得到更好成绩”。本轮未做这些变更。

## 本地学习入口与复查

- [systems.py](../labs/systems.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [search_contracts.py](../labs/search_contracts.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py startup-amortization`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
