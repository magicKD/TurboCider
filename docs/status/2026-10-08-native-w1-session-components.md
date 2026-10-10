# Native W1S：统一构建、完整 tiny 去噪与真实 FFN

2026-10-08，M4 Pro / 48 GiB。对应 [机器回执](2026-10-08-native-w1-session-components.json)，实现与后续独立后端规格见 [15](../design/qwen-image-gpu-ane-command-graph/15-native-w1-implementation.md)、[16](../design/qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)。

当前 private native 完整构建通过。构建身份为 `tc-runtime-build-v1-8b0ff447bb5ee33be46e6b5797f1833e60ca3a2895525f53365a3d04368624b5`；所有 native 构建源与 manifest 一致，使用进程内 CLT 选择、native/package-only，未额外构建 App。7 项路由、129 项通用合同（3 项跳过）及 5 项实验监督检查通过。监督器另外修正了超时后 exit 0 的竞态和结果验收失败却记录 returncode 0 的问题；这些检查不加载模型。

生成数据的完整 NativeDenoiser 对照为 H128/F1024/L2、target 32 / text 7、两步。GPU 与 hybrid 各一 run，共四步；后者首步仍 GPU、稳态完成 **2 FFN / 1 hybrid step**。最终 latent RRMS **0.2765%**，最大误差仅作诊断。所有输入、source/padding、物理完成、phase plan backing 数及同一账本归零检查通过。tiny GPU 稳态约 1.99 ms，hybrid 3.78 ms；只说明此小尺寸的完整路径及提交开销，不作为真实模型性能结论。

真实 FFN 仅读取本地 Qwen layer 0 的 BF16 gate/up/down（288 MiB）与已有强制 GPU 轨迹的 step 1 / 19 输入（合计 16 MiB）。M1024/H4096/F12288/A6144；每个输入每条路线 **一次 warmup、一次记录**，step 1 为 GPU→hybrid，step 19 为 hybrid→GPU，共八次 FFN。对照包含完整 GPU FFN 和完整 native W1S 的 W8/A8、driver submit、GPU 补集、restore/join/finite/物理完成，而非只量 ANE prediction。

| 捕获输入 | 完整 GPU | 完整 W1S | 耗时变化 | 输出 RRMS |
|---|---:|---:|---:|---:|
| layer 0 / step 1 | 50.590 ms | 30.935 ms | 减少 38.85% | 0.3388% |
| layer 0 / step 19 | 41.049 ms | 41.490 ms | 增加 1.07% | 0.2727% |

两个输入的记录各只有一个样本，不能拼成相同输入的重复统计或平均加速比。held-out step 19 没有净收益，因此**暂不增加完整 hybrid E2E**，不授 B1、默认采用或 App 资格。之前 GPU-only E2E、legacy hybrid 和 tiny FFN 属于各自构建，不作为本构建整图证明。

本组件 producer 只有 W8/A8 staging，记录跨度分别 9.15 / 19.89 ms；branch join 约 20.7 ms，consumer 约 0.7–0.8 ms。执行顺序允许 GPU/ANE 重叠，但 host span 不测量硬件 busy time，不能相加或据此证实 overlap。当前无法归因 producer 差异是频率、调度、cache 还是具体 kernel；没有相应证据。优先细分 staging、受 source/recipe 约束地复用 row scales，再比较有界预取和设备图，避免直接反复跑完整图。

冷准备另记：dense PSO 0.087 s、hybrid program/PSO 1.213 s；独立生成数据请求的准备时间也不与真实 layer 直接排名。自有新 Metal/Surface backing 的统一账本峰值 **164,677,644 B（157.05 MiB）**，9 个 Surface 共 89,161,728 B；全部归零。原始 MLX source/input 是明确保留的 baseline，没有合成账本 claim。源矩阵与两份捕获输入/文件均前后摘要一致；checkpoint 只使用 held descriptor + canonical stat generation，不声称完整文件内容哈希。

真实回放总进程时间 5.55 s，进程最大 footprint 597,672,896 B；该次 `/usr/bin/time` swaps=0 不认证系统全程 swap 或完整请求预算。compiler/driver、全进程内存、硬件 INT8 和整图性能仍未认证。

工具与原始证据位于 `outputs/native-w1-session-20261008`，测量二进制保留。结束后仅清理本轮明确归属的 object、未运行中间二进制和两个程序缓存；删除清单/摘要见 `owned-cleanup.json`，用户模型、历史实验与系统 driver cache 不变。没有下载模型或复制完整 checkpoint。
