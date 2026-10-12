# Native LoRA FP32 SGMMA 组件筛选

日期：2026-10-08；设备：Apple M4 Pro；构建：`tc-runtime-build-v1-77863b3f55434af37ee12615039845b576878a6f310e39656bc2bc1cf308c2fb`。

本轮生成组件的数学与生命周期检查通过；真实 Qwen Image 2.1 Viggle r128 的 **layer 0 / step 5 完整 FFN** 在 AB、BA 两个记录顺序中分别减少 **7.26% / 3.59%** 耗时，Scalar v1 与 SGMMA v2 输出逐位相同。这是单一历史输入的完整 FFN 筛选通过。独立修正算子发生明显 AB/BA 性能反转，不能宣称稳定的 kernel 加速；本轮也不授予完整 DiT、图像 E2E、GPU/ANE LoRA 或 W2 资格。

## 实现与身份

候选使用 FP32 `simdgroup_float8x8` 执行 low 与 apply，继续读取实际 BF16 / FP32 因子，保留 FP32 low、effective scale、有序逐项 BF16 相加及 bias-last 边界。没有把 FP32 因子缩窄到 FP16。显式 `SgMmaR128` 对符合形状/步幅要求的 rank128 项选择候选，其余合法形状走已有 Scalar；候选能力/PSO 准备失败向上传播。

每个候选 low/apply threadgroup 的静态共享数组合计 **6,144 B**，Scalar 为 5,120 B；候选增加两个预准备 PSO，复用现有 low/output backing，**新增 native 全局 scratch 为 0 B**。这不表示 PSO 或 driver 开销为零，相关框架 guard 也没有认证。源码：[精度与容量](../../native/backends/metal_graph/kernels/native_lora.hpp#L13)、[选择条件](../../native/backends/metal_graph/kernels/native_lora.hpp#L87)、[low/apply](../../native/backends/metal_graph/kernels/native_lora.hpp#L346)、[prepare 与 dispatch](../../native/backends/metal_graph/native_lora.mm#L25)。

[运行前身份核验](../../outputs/native-lora-sgmma-20261008/pre-device-verification.json)记录 567 个构建源无摘要差异。生成探针 binary SHA256 为 `c83e6e3478cd89edd1adf5d9e1ce53466d868e6bcfd339821394dbc767fcebbe`，源码 SHA256 为 `b38166650f6a02e599f8b892948ed1fcef7fe631b75c8b4023f2287f8cce6f07`。真实 replay 源码 SHA256 为 `b8b73281269ea4b841fdf2461c7629345a4822d5018d6ae1f7f0cb024f557e0f`；其回执记录相同运行库身份，监督记录绑定实际 binary 路径，回执本身未提供该 binary 的内容摘要。没有将历史版本的结果转授本构建。

## 生成组件：数学通过，性能有顺序依赖

两个生产形状均为 M=1024、单 BF16 r128 项，分别 K4096→N12288 与 K12288→N4096；每形状 warm AB 一次、record AB/BA 各一次。两路线共用预生成的 BF16 base，**只测 correction，不包含 dense GEMM 或完整 FFN**。另有 M8/K128/N64 的两项有序 BF16/FP32 correctness-only fixture，各路线一次。本组共 14 次 native 提交，无模型加载、ANE 调用或图像。

下表使用包含 encode→submit→物理完成的 host wall；warmup 不计入排名。正值表示减少耗时，负值表示更慢。

| 组件 | 记录顺序 | Scalar ms | SGMMA ms | 耗时减少 |
|---|---|---:|---:|---:|
| gate/up correction | AB | 12.778 | 4.730 | 62.99% |
| gate/up correction | BA | 6.587 | 10.967 | −66.49% |
| down correction | AB | 16.091 | 5.086 | 68.39% |
| down correction | BA | 4.986 | 13.849 | −177.79% |

生产形状最大 low RRMS 为 `9.46e-7`，最大 BF16 output RRMS 为 `1.30e-5`，均通过预设 `1e-4 / 1e-3` 门；这些是相对 MLX oracle 的比较，不是把相同统计值当作两路线逐位一致证据。小 fixture 含 `1 + 2^-13` 的真实 FP32 因子（FP16/BF16 均无法精确表示），oracle 已校验 low 与最终 BF16 输出 34；两路线 low/output 均逐位相同于 oracle。它验证该固定算例，不能证明任意归约都逐位一致。

pair 内两 arm 之间没有插入 CPU 比较；pair 后的比较/完整性检查另记约 0.11–0.26 s。顺序反转真实存在，但没有频率/热状态遥测，不能把原因归结为 CPU 间隔、GPU 调频或缓存中的某一项。三个 phase 均 physical-complete、source/padding 不变且账本归零。

## 真实输入：完整 FFN 两个顺序均改善

输入来自既有 LoRA 轨迹 `75991-1226054446574458` 的 layer 0 / step 5，M1024、H4096、F12288。本地 pinned Viggle v0.2.1 r128、strength=1，每个 gate/up/down 一个 BF16 项。只物化该层的 gate/up/down、六个 factor、输入及一次 MLX oracle，没有准备整套模型或导出新模型。

每次执行含三个 dense 投影、gate/up 的有序 LoRA 修正、SiLU×up、down 修正，共 10 dispatch；SGMMA 路线其中 6 个 low/apply dispatch 选择候选。warm AB 一次、record AB/BA 各一次，共 6 次完整 FFN。

| 记录顺序 | Scalar 完整 FFN ms | SGMMA 完整 FFN ms | 耗时减少 |
|---|---:|---:|---:|
| AB | 54.993 | 51.003 | 7.26% |
| BA | 54.341 | 52.392 | 3.59% |

两个记录样本的算术平均为 54.667→51.697 ms，描述性差异约 5.43%；样本只有同层同输入的两个顺序，不是总体估计或置信区间。三个 pair（包括 warmup）两路线输出均 **0 BF16 位差**。六次结果对实际 MLX 完整 FFN oracle 的 RRMS 都是 `9.935683e-7`，4,194,304 个输出中 95 个 BF16 位差，max_abs=0.001953125，通过预设 RRMS≤0.005。没有放宽数值门。

host wall 覆盖 encode / submit / physical completion，排除准备、oracle、CPU 比较和来源 hashing。每 pair 两 arm 之间存在明确计时的 8 MiB 输出 CPU snapshot，约 0.261–0.279 ms；完整比较和检查在两 arm 之后，约 28.28–28.42 ms。Ticket GPU span 是第一 GPUStartTime 到最后 GPUEndTime，不能解释为 kernel busy time 或重叠证明。

## 准备、来源与生命周期

真实 replay：source 准备 297.93 ms（含 held-FD adapter 内容验证 246.68 ms）、一次 MLX oracle 170.03 ms、共享 dense PSO 132.12 ms、Scalar PSO 9.16 ms、SGMMA PSO 10.04 ms，均不计入 arm 时间。两路线依次准备，共用 OS/driver 编译缓存，不能由此推导全新安装的冷启动差异。

Base 由同一 held descriptor / canonical stat generation 证明来源，**没有重新 hash 完整 checkpoint**；adapter 同 held-FD 验证 pin，输入/complete metadata 及全部 produced source/oracle 内容有前后摘要，MLX 实际 Data owner 已核验。没有下载模型。六个 native scratch backing 合计 93,192,192 B（88.875 MiB），复用同一 MemoryExecutionContext/MemoryBridge。已有 MLX source 由同步后的 process baseline 覆盖，不冒充已登记 native claim；source payload=308 MiB，oracle 后 MLX active=331,350,016 B。收尾 storage count/bytes、reservation、pending release、unknown bytes 全为零；所有工作实际完成、source/padding 不变。整个请求和 framework guard 均未认证。

两个监督进程 exit0、无 timeout，elapsed 分别 2.263 s / 3.342 s。监督退出本身不授予物理完成；物理完成与零账本来自各 probe 的 Ticket / phase 检查。

严格 host 与 CPU 记录：算子 metadata 合同 192 项、source 合同 645 项；模型合同 13 接受/34 拒绝、public/private host；探针比较合同 7 个无效输入拒绝；请求 route 13 项通过。这些不替代本节的设备数值证据，也不扩大当前候选域。

## 验收边界与下一步

后续一组独立监督的 MLX→SGMMA 整图配对**现已完成**，详见[独立 E2E 报告](2026-10-08-native-lora-sgmma-e2e.md)及[原始回执](../../outputs/native-lora-sgmma-20261008/e2e-pair/receipt.json)。整图仍比 MLX 慢，**没有性能资格**；其数字不并入上文的组件收益，单输入完整 FFN 的收益也不转授整图。本组不支持宣称稳定生成算子加速、所有层/文本/LoRA 收益、E2E 图像质量、GPU/ANE LoRA、W1R/W2、App 或默认发布资格。独立 Splash 风格后端更深窗口的命令图/失败域设计仍需另外验收。

清理已由 root 在设备工作完成后结算：**244 个自有文件，29,332,442 B（约 27.97 MiB）**。范围是本轮 `build/native-lora-sgmma-runtime-20261008` 的对象/归档/模块缓存（238 文件）、`build/native-lora-sgmma-component-20261008` 与 `build/native-lora-sgmma-replay-host-20261008` 各一个对象，以及 `outputs/native-lora-sgmma-20261008/generated/{gate-or-up,down,fp32-ordered-trap}` 的四个生成 fixture。运行库、CLI、manifest、probe binary、原模型/历史 capture、原始 receipts/logs 及两张 E2E 图均保留；未触及系统 driver cache。[逐文件清理清单](../../outputs/native-lora-sgmma-20261008/owned-cleanup.json)保留原摘要，本文 JSON 另绑定清单 SHA256。本次文档更新自身没有删除文件或改写原始记录。

原始证据：[生成 NDJSON](../../outputs/native-lora-sgmma-20261008/generated/receipt.jsonl)、[真实 FFN](../../outputs/native-lora-sgmma-20261008/real-ffn.json)、[生成监督](../../outputs/native-lora-sgmma-20261008/generated-supervisor.json)、[真实监督](../../outputs/native-lora-sgmma-20261008/real-ffn-supervisor.json)、[host](../../outputs/native-lora-sgmma-20261008/probe-host.log)、[CPU](../../outputs/native-lora-sgmma-20261008/probe-cpu.log)。机器可读汇总与原回执摘要：[JSON](2026-10-08-native-lora-sgmma-screen.json)。
