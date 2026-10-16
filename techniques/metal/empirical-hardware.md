# 微基准与硬件特性：怎样使用社区逆向结果

[GPU 导航](README.md) · [TensorOps/M5](tensorops-and-m5.md) · [Profiling](profiling-playbook.md)

核查：2026-10-16。[metal-benchmarks 固定读本](https://github.com/philipturner/metal-benchmarks/blob/dc2adc640a1588246f4471d415aa6873cb6e3499/README.md)前 210 行包含 M1/M2 的架构、内存层级与指令/依赖微基准。本轮读了说明和结果，**没有运行上游全部 benchmark**。SHA 与范围见[来源清单](research-sources.json)。

## 值得学的是实验设计，不是复制规格数字

同一个 FP32 操作，在独立指令、依赖链、低 occupancy、高 ILP 下表现可能不同。微基准可以刻意改变每线程 live values、依赖长度、线程组与工作集，定位指令吞吐/延迟、cache 或寄存器约束。它们是观察和模型；社区推测的 cache/register 大小与官方保证要分开。

读本也明确保留过时/次优测法说明。其对未来 GPU 不太可能加入 tensor cores 的历史预测，不能覆盖后来 Apple 官方 M5 GPU Neural Accelerators 的事实。不要把一份旧读本的未来推断写成当前芯片规格；更不能将 M1/M2 的寄存器、指令吞吐、频率或缓存数字套给 M4/M5。

## 面向推理的五种最小探针

| 探针 | 控制变量 | 要否决的错误解释 |
| --- | --- | --- |
| 连续/strided copy 与 fused elementwise | 字节数、alignment、tail、pass 数 | “FP16 算术更快”其实只是少搬字节 |
| 依赖链 vs 多独立 accumulator | 相同操作数、不同 ILP/live values | 把低 occupancy 的 latency 当设备峰值 throughput |
| 工作集/重复次数扫描 | 数据跨 cache 层级、cold/warm 分开 | 将 cache 命中吞吐标成 DRAM 带宽 |
| GEMV/GEMM M 扫描 | 同一权重、dtype、N/K、consumer | 将 prefill 的 tile 推广到单 token decode |
| 矩阵原语/提交路径分别比较 | 同数学、不同 primitive；同 primitive、不同 batching | kernel 改进与 host launch 改进混在一个倍率里 |

所有实验先查设备 runtime limits 与 pipeline 的 `threadExecutionWidth`、`maxTotalThreadsPerThreadgroup`、实际 TG memory；不要只按家族代号硬编码。runtime 报告的可用/推荐内存是资源约束线索，不是允许长期占满系统内存的授权。

本机已有 [Metal 实验室](../labs/gpu.md)和[真实 affine tile 负结果](../review/gpu-affine-20261016.md)，本轮新增 [MLX 小组件](experiments-20261016.md)。它们覆盖的只是部分 workload；无 counter 时不能声称已量到 bank conflict、spill 或 NAX 利用率。要验证硬件归因，应增加可区分假设的控制，并让现有成熟 kernel 参与比较。

## 迁移到新 SoC/SDK 时重开什么

重新测支持的 API/primitive、dtype 与数值语义、shape dispatch、线程组资源、冷/热编译、实际权重/激活驻留。旧 tile、split-K 阈值与 GPU/ANE 分担比例只是候选；M5 的矩阵峰值提高后，elementwise、quantize/decode、memory 与提交开销可能成为新的占比大项。这是 Amdahl/roofline 分析方向，不是已测过的 M5 模型结论。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/metal/run.py](../labs/metal/run.py)、[labs/metal/kernels.metal](../labs/metal/kernels.metal)；case：`metal:vector`、`metal:gemm`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
