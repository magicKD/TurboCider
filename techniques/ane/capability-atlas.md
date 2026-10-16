# ANE 能力地图：硬件、图格式、编译器与实测分开

[ANE 目录](README.md) · [来源审计](source-audit.md) · [量化](w8a8-int8.md)

日期：2026-10-16。这里是可复查的能力边界，不猜未公开 SRAM、指令集或跨 SKU 性能。一次失败至少绑定 SoC、OS build、SDK、编译入口、shape、类型、权重常量/输入及数值域。

## 能力与版本矩阵

| 特性 | 可以确认的范围 | 不能推出的结论 |
| --- | --- | --- |
| FP16 浮点图 | 官方 ANE Transformer、Core ML SD 和大量运行时使用 FP16 | FP32/BF16 输入就按原精度在 ANE 计算；内部累加精度全部公开 |
| W8 存储、浮点激活 | `constexpr` 解码的权重压缩，非 M4 专属 | W8A8、2× 整数运算 |
| W8A8 整数加速 | Apple 以 **A17 Pro、M4**为例明确说明 INT8×INT8 吞吐较前代提高；Core ML 需要合适的激活/权重量化及算子 lowering；更新SKU仍需复查 | 整数能力首次出现的精确代际；旧芯片完全没有整数能力；老图/所有算子自动加速 |
| 低比特 LUT 表示 | ML Program 的 palettization 表示从 iOS16/macOS13 可用 | 运行一定保持压缩、一定 ANE 执行 |
| LUT 运行时收益 | 官方指出 iOS17/macOS14 起 ANE 上更可能按需解压、减少搬运 | 任意 LUT4 图都快 4×；硬件 INT4 MAC |
| grouped-channel LUT、量化 LUT | iOS18/macOS15 的新表示/配置；更多 LUT 有额外成本 | 最小 group 永远最优；比特数就是实际总 bits/weight |
| stateful KV | Core ML stateful ML Program 可省去显式逐次状态 IO；ANEMLL 有具体转换器 | state 永驻片上 SRAM；跨 session 可无锁复用 |
| 训练 | 可将反向和更新算式编为计算图；研究 runtime 已有实现 | 官方 MLUpdateTask、MLCompute training graph 自动在 ANE 训练 |
| Core AI | Apple 另有导出/运行入口，所读样本要求 OS27/Xcode27 | 本机 OS26.6.2 已经验证；Core AI 是 ANE 专属 |

主要依据：[官方量化性能](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)、[LUT 表示](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)、[LUT 性能/版本](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)、[stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)。这些是框架能力与官方建议，不替代目标模型实测。

## M4 Max 不等于更多 ANE 核心

Apple 的 [M4 Pro/Max 发布资料](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)列出 16-core Neural Engine。不要按 GPU 的 Pro/Max/Ultra 命名推算 ANE 核数或将 GPU 带宽全部算给 ANE。所读 anemll-bench 硬件 helper 将若干 Max 写成 32、Ultra 写成 64；那是代码里的型号映射，不是设备发现或官方规格，见 [profiling 审核](profiling.md)。

M5 GPU 的 Neural Accelerators/NAX 与独立 ANE 是两个设备方向；不能用 GPU TensorOps/MLX NAX 成绩更新本表的 ANE 实测列。[路线图区分](../hardware/apple-silicon.md)保留相同原则。

## BF16、FP16 和“支持”的三个含义

BF16 与 FP16 都是 16 bit，但前者指数范围大、后者有效尾数更多。将 BF16 参数和激活 cast 为 FP16 可能溢出；文件加载成功不等于原 BF16 数学成立。必须分别记录源精度、输入转换、图内类型、累加/归约、输出和 GPU join 精度。

同样，MIL 有某个 op、编译器接受它、该图在目标设备正确执行，是三个不同问题。本库已有单 SiLU 精度负结果，但不能由此声称“ANE 硬件没有 SiLU”。见 [图搜索](graph-rewrite-search.md)。

## 怎样扩展能力地图

每个探针应包含：最小合法图、编译与加载分项、每次完成检查、全输出 finite/误差、shape/stride、counter 缺失值和可重开条件。先采用几十个有独立数学 reference 的受控 case，再增加真实激活。不要用“119 次编译”“W≥32”“某个 SRAM 拐点”作为无版本的硬约束；这些是特定项目的经验，见 [负结果](negative-results.md)。

本轮新实验只证明 [指定公共图的接口性能与数值](experiments-20261016.md)：小图计划 CPU、大图计划 ANE，物理 trace 与整数 MAC 仍未取得。这个区分本身比把所有数字放进“ANE TOPS”表更有学习价值。

## 本地学习入口与复查

- [systems.py](../labs/systems.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [pmu.py](../labs/pmu.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py bucket-selection`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
