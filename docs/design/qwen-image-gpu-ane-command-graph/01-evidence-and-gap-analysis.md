# 01 · 固定证据、实现差距与取舍

本章描述 2026-10-07 读取的代码与既有实验。设计建议见后续章节，不能把本章中的历史测量归因于尚未实现的 command graph。

## 1. 来源与版本

| 来源 | 固定版本 / 范围 | 用途 |
|---|---|---|
| Splash #260 | `80827e1c3593f7d10c266394e4d4381321b73ad0`，2026-10-06 合并的 PR head | 主要架构参考 |
| vpipe | `3e2682a0847c9e76dd37978f2be01f53195ae5e3` | 上游固定版本的原生执行、LoRA、权重准备、residency；本机旧 checkout 不等于此 pin，核验见 08 |
| TurboCider | HEAD `8d12b44cc764d079ee31c6092725e22fcc8fb9e0` **加当前未提交工作树** | 当前实现，不等同于该 commit 单独内容 |
| MLX | 本地 0.32.0 头文件与对应上游版本 | command encoder/Primitive 适配可行性 |
| 本机 FFN / E2E | [2026-10-07 报告](../../status/2026-10-07-qwen-real-ffn-screen.md)、[JSON](../../status/2026-10-07-qwen-real-ffn-screen.json) | 真实输入、完整路径与限制 |
| 固定 Core ML | [frozen 回执](../../status/2026-10-07-qwen-frozen-laptop.json) | 负面结果与不继续全模型导出的依据 |

源码定位统一列在 [08](08-source-map-and-decisions.md)。本地行号随改动可能变化，定位应以文件与符号为准。已有 Splash attribution 引用早期 `0ac3d5b…`；不能把后续 PR 的所有能力说成已经移植。

## 2. Splash 值得作为基线的部分

### 2.1 拥有实际提交顺序

`runtime/metal/CommandGraph.hpp` 表达 dispatch 与事件；`MetalBackend.mm` 负责编码与提交；`AneFfn::encode/commit` 先建立各层任务、再提交 GPU command。关键不是类名，而是模型 FFN、GPU 补集、量化、staging 和 ANE ready/done 在同一可分析的执行顺序中。

TurboCider 应复用这一职责划分，而非在现有阻塞回调外加一个“graph”包装。

### 2.2 动态权重与程序分离

编译程序表达相同形状的计算，层权重进入可复用 bank。它避免按每层/每种 LoRA 导出完整冻结模型；输入与权重格式转换仍然有成本。多 function 行数、预绑定 surface、行尺度缓存、空闲卸载，分别解决编译、请求绑定、staging 和生命周期问题。

### 2.3 设备依赖与请求验收分离

ready/done 事件让设备按依赖前进，完成状态用于判断本次执行是否可采用。出错时解除 wait 是故障排水动作，不是成功证明。

TurboCider 的预排队实现还必须证明：失败后放行的后续写入不能覆盖晚到 ANE 正在访问的 bank。03 对此增加独立的物理完成与隔离合同；不能仅照搬一个 timeout 常量或 signal 调用。

### 2.4 校准覆盖完整执行路径

Splash 采用少量深度、两种份额、提交固定开销校正、邻近通道份额稳健选择，以及逐行数 function 的收益检查。该思路值得保留，具体拟合假设不能直接套用。

其输入生成、LLM mixer stand-in、模型维度与硬件不是本地 Qwen DiT。TurboCider 已有真实 FFN 回放，应优先把真实数据和真实 attention 上下文加入这套方法。

## 3. vpipe 的参考边界

| 可参考项 | 应吸收的优势 | 不能直接假定 |
|---|---|---|
| command stream | 批量编码、资源绑定、提交窗口 | 同名接口能直接接进 MLX 调度器 |
| runtime LoRA scratch | 按最大行数和有效 rank 预分配，减少热点分配 | BF16 低秩 scratch 与当前 FP32 rank 合同等价 |
| LoRA 投影融合 | 只对适合的 M/N/K、rank 和 kernel 融合 | 所有 down/大 K 投影融合都更快 |
| Qwen ANE weight staging | 可复用准备器、转换器、worker | staging 时合并 BA 等于激活修正 |
| block residency | 循环扫描下固定 pinned 子集、可复用 refill | 增加内存缓存必然降低首请求时间 |
| Metal pipeline cache | 缓存 PSO/二进制产物，减少首次准备 | shader 编译时间是每次 GEMM 都支付 |

vpipe 的 Qwen ANE 路径在 staging 中把 LoRA 的 BA 合入权重后写 FP16 slot；当前 TurboCider 私有路径在激活上加入 gate/up 修正，并在完整 hidden 上做 down 修正。二者舍入、准备成本、strength 失效条件不同，设计为不同 recipe，不能混为一个 `runtime_lora=true`。

## 4. TurboCider 已有能力与缺口

| 领域 | 已有 | 缺口 / 后续设计 |
|---|---|---|
| GPU FFN | 完整 GPU 编译图，gate/up 合并投影 | 通道补集 gate/up 分开执行，不能假定 `mx::compile` 会恢复 GEMM 融合 |
| ANE 量化 | 动态 W8A8、旋转、GPU I/O、scale cache、双 bank | A8 staging 完成后 host 才继续提交；多层统一依赖尚未成立 |
| GPU/ANE 并行 | worker、async GPU head、shared event | 每层输入 eval、ANE finish、join 边界依旧存在 |
| Metal buffer | 引用 MLX buffer 并保留所有者 | 多队列依赖、资源 generation 与外部写入可见性需要统一合同 |
| LoRA | FP32 rank、pre-SiLU 修正、完整 down 修正 | Qwen 跨分支重复低秩投影、完整 hidden 拼接；多 adapter 舍入语义需区分 |
| 行数 | request rows、padding/tiling、变长请求 | 首步 1053 行套 1024 function 多一次调用；不同阶段策略未统一 |
| 失败 | 局部 whole-FFN GPU fallback、超时隔离 | 深预排队后需事务回放和跨层资源隔离，不能沿用局部回调假设 |
| 自动校准 | 多深度、两份额、完整候选 trial、进程内 bounded cache | 自动样本仍有随机输入；真实回放与持久性能计划未打通 |
| 内存 | memory plan、storage lease、completion、pressure | 新图不能绕过现有内存合同另建 arena 私账 |
| 条件缓存 | 文本、参考图、prefix/KV/snapshot 等按路径存在 | 不同驻留模式/worker 的可复用范围需统一；近似 DiT cache 独立 |
| App | 实验 worker、请求身份、退出/回执检查 | 每任务新进程；最佳 CLI 配置未等同于 App 配置 |
| H3/LTX | 自有 Metal、batch/continue、residency 经验 | 不是已经具备可供 Qwen 直接复用的统一 device graph |

相关代码：`native/backends/ane_ffn.cpp` 的 `run_channels`；`private/ane_w8_executor.mm` 的 `run`；`qwen21/runtime_ffn_graphs.hpp`；`apps/macos/RuntimeImageWorker.swift`。

## 5. 本机性能证据

### 5.1 已测请求边界

M4 Pro / 48 GiB / macOS 26.6；本地 Qwen Image 2.1 BF16；同 teapot prompt、seed 42、512²、无参考图。base 20 步，LoRA 为本地 Viggle r128、strength=1、6 步。component-staged、PE 与近似 DiT cache 关闭。

四个 fresh-process 请求，不等于系统编译缓存冷启动；每臂只有一个完整请求，不构成显著性或发布证明。其 runtime build 由机器回执单独记录，不能与早前 frozen build 混比。

| 请求 | GPU wall | 候选 wall | GPU denoise | 候选 denoise | 候选 setup |
|---|---:|---:|---:|---:|---:|
| base | 48.400 s | 43.044 s | 42.155 s | 35.268 s | 1.658 s |
| LoRA | 20.664 s | 21.292 s | 14.183 s | 13.637 s | 1.187 s |

base 采用 6144 ANE 通道、提前提交；LoRA 采用 4096 通道、不加同一提交栅栏。均启用 fixed async。两者失败、回退和 overflow retry 为 0。该结果不代表所有输入无需 overflow 防护。

### 5.2 组件结果解释

真实 BF16 输入来自 3 个深度 × 3 个步骤，回放包含 staging、A8、同步和 join：

| 候选 | 完整 FFN GPU → hybrid | 解释 |
|---|---|---|
| base 6144，无提前提交 | 41.87 → 39.84 ms | 有少量收益 |
| base 6144，提前提交 | 约 41.9 → 27.9 ms | 队列排序可显著影响重叠 |
| LoRA 4096 | 46.62 → 36.59 ms | FFN 有收益，未自动转化为请求收益 |
| base / LoRA，S1+旋转 | 未优于对应最佳旋转 | 不默认启用额外平滑 |

base 的提前提交收益已包含在最佳 E2E 中，新图不能再把这一次 1.502× 作为额外倍率乘上去。该 probe 没有完整 attention/residual，因此必须增加完整 block 对照。

### 5.3 首步与稳态

| 请求 | GPU 首步 | 候选首步 | GPU 后续步中位 | 候选后续步中位 |
|---|---:|---:|---:|---:|
| base | 2.249 s | 2.627 s | 2.097 s | 1.721 s |
| LoRA | 2.491 s | 2.979 s | 2.327 s | 2.101 s |

首步 1053 行（1024+29）需要两个 1024 调用，后续 1024 行只需一个。首步 GPU 与多 function 策略值得比较，但节省值只能在同配置完整请求中确认。

### 5.4 不能相加的统计

base 累计 stage 13.103 s，暴露 stage wait 仅 0.232 s；pre-FFN 19.992 s 包含 attention 等上游工作；post-join 4.471 s 还包含未完成的 GPU 工作。ANE wait 与 GPU 分支重叠。

因此：

- 不能把 wait 全当成量化 kernel 时间；
- 不能把 stage、ANE wait、GPU head、post-join 相加作为总耗时；
- 全缓存 W8 不能宣称节省 13 秒；
- 仍应检查 staging 带宽竞争，因为等待隐藏不等于资源占用消失。

## 6. 应保留的负面证据

### 6.1 全层 frozen FP16

另一匹配构建的 base 从 48.901 → 83.941 s，LoRA 从 20.763 → 30.947 s，去噪也略慢；setup 分别 31.421/8.870 s。约 9.1 GiB 的无效衍生物已在该实验后清理。

这是本机具体 frozen 候选的失败，不能推断所有 fixed/Core ML 都无价值；也不能继续以组件吞吐为理由默认导出 32 层。

### 6.2 静态 SmoothQuant W8A8

单层 base 1.937×，但未校准步骤的相对 RMSE 0.249；LoRA 1.286×，误差 0.112，超过该轮 0.03 门槛，拒绝全模型导出。这是该 per-tensor recipe 与校准样本覆盖的失败，不是对所有 SmoothQuant 的否定。

### 6.3 public runtime INT8

本机构建 SDK 15.2 无法编译需新 SDK 的 public runtime INT8 I/O 路径。该路线未得到匹配测量；回退不算 public INT8 性能数据。

### 6.4 ConvRot 来源

当前 dense BF16 不能仅改枚举就成为 Comfy ConvRot Q8 direct-code 来源；缺少正确代码平面、旋转 basis、尺度 ABI 时应拒绝。拒绝后 GPU 零误差不是 ConvRot 路线通过。

## 7. 设计优先级的依据

1. 先把相同的最佳 GPU 算子纳入更好的设备依赖，减少人为边界。
2. 六步 LoRA 优先处理初始化与重复低秩计算，二十步 base 更重视持续并行。
3. 首步单独处理，稳定 1024 行重复计划优先优化。
4. GPU/ANE 共用内存带宽；份额不是越大越好。
5. 新的 fusion/量化与 command scheduling 分臂测试，才能知道收益来源。
6. 现有内存/源身份/回执能力保留，避免性能重构同时丢失工程合同。
7. 暂不把 VAE、视觉编码器、QKV 全部一起拆到 ANE；它们是后续单独 profile 的候选，不能与 FFN rollout 捆绑。
