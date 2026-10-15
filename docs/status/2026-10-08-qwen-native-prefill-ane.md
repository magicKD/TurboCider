# Qwen Native encoder / 无 KV 编辑首步 ANE 进展

2026-10-08，M4 Pro / 48 GiB / macOS26.2。已参考 [Splash #260](https://github.com/incoai/splash/pull/260) 的通道并行、FP32 partial join、准备资源复用和提交后的源预取，完成 Native encoder 语言 FFN 与编辑首步 native FFN island。完整设计见 [23](../design/qwen-image-gpu-ane-command-graph/23-native-encoder-prefill-ane.md)。没有下载模型或导出完整模型。

## 实现

- 共享 `NativePrefillFfn`：变长有效行 → 固定 ANE bucket 补零 → chunk 并行 → FP32 join → BF16 裁剪。attention 不补零。使用原始 BF16 checkpoint，GPU/ANE 分别处理互补通道。
- Native encoder：视觉27层保持 Native GPU；语言36层可显式使用 C6144 W1。提交 GPU/ANE 后准备下一层源权重，最多当前+下一层；准备异常等真实物理完成后退役。
- Edit step0：5148真实行分成两个2688 bucket，32层共64 chunks。只改首步 FFN；attention/modulation/residual/Euler/VAE沿用 MLX。后续 step 关闭该 callback。
- Session 管理 prefill program/operator/scratch，增加 `Session::prewarm_native_prefill(absolute_cache)`；重复 idle prewarm 不重建。每次重置计数并重新 staging 当前权重，复用不包含 KV、conditioning 或 LoRA。
- 修正 Native ANE 精度/执行来源记录，保留真实 vision precision；移除 encoder ANE 不使用的 full-GPU FFN operator/workspace。默认仍为 GPU/MLX，显式实验路径不忽略 LoRA。

## 真输入组件筛选（r2 构建）

完整视觉编码后的 encoder layer0：M1066/H4096/F12288，bucket1152。
真实单图 edit step0 layer0：M5148，bucket2688，实际 VAE/conditioning，无 KV。
两个候选 C3072/C6144，各一次 warm 与 AB/BA 两顺序；误差对原始 MLX BF16 FFN。Native full GPU oracle在这两个输入上逐值一致。

| 场景 | C6144 Native GPU | 完整 hybrid | 两顺序平均省时 | FFN RRMS |
|---|---:|---:|---:|---:|
| encoder | 约43.7 ms | 约29.4 ms | 32.71% | 0.748% |
| edit首步 | 约205.6 ms | 约126.9 ms | 38.26% | 0.171% |

C3072省时约10.55%/15.29%，选择C6144。包含补零、W8/A8 staging、两分支、restore/join/裁剪；冷 program/PSO prepare另计，不能拿组件省时当请求收益。

## 完整 encoder（r3 构建）

一张1024 reference，512²目标；全部视觉27＋语言36层。每个新进程 arm用同一个 NativeEncoder对象编码两次，第二次 fresh source权重，未复用完整condition/KV。

| 资源状态 | Native GPU | Native GPU/ANE | 省时 |
|---|---:|---:|---:|
| 首次准备 | 4.7343 s | 4.6155 s | 2.51% |
| 准备复用 | 3.9392 s | 3.2868 s | 16.56% |

conditioning RRMS **4.510%（相对同构建 Native GPU）**，通过预声明8%筛选。这里不是相对 MLX 的完整conditioning误差，不能与上一轮2.79%直接相加。语言阶段复用时2.4067→1.7534s。每次36个ANE FFN、36 chunks、3096补零行；future source windows为48（视觉13＋语言35）。首冷总体收益未达到5%门槛，复用后通过。

r3为 `tc-runtime-build-v1-b1bfdab145c73016430effbc227d7125e45edc80f1418127210195653f0abc6c`。r4移除未使用准备资源并整理owner；未重复完整encoder，不能宣称16.56%已在r4重测。

## 编辑首步：保留冷启动负结果，再验证预热

冷启动 r3 单步 base：去噪 **12.0413→13.0791s（慢8.62%）**；request **21.1817→21.5448s（慢1.71%）**。
虽然完成32 FFNs/64 chunks，FFN内部总时间4.9608s，但 prepare **2.6481s**、input-ready等待5.3266s，抵消收益。图像pixel RRMS0.950%，只代表单步数值/执行诊断。

由此整理成Session owner并提供显式预热；在最终r4做一组同构建 fresh-process 单步对照，两个arm先同样load模型，hybrid额外idle prewarm，第一次生成仍无conditioning/KV命中：

| 指标 | GPU | 已预热 hybrid | 省时 |
|---|---:|---:|---:|
| 首步去噪 | 13.2520 s | 11.4997 s | **13.22%** |
| 生成 request wall | 22.2757 s | 20.8008 s | **6.62%** |
| 模型load＋prepare＋generate | 24.5391 s | 23.3926 s | **4.67%** |

本次已有系统/driver缓存下，额外prewarm wall约0.0794s（operator body统计0.0480s）；request prepare为0且`preparation_reused=true`。不能把历史2.65s减去0.079s单独当重构收益：两轮构建、缓存状态不同。完整cold总收益仍低于5%门槛。固定GPU→hybrid、各一次采样，系统/compiler caches未清，不宣称稳定平均加速。

最终32 FFNs/64 chunks全部真实完成；valid_rows164736、padded_rows7296。图像pixel RRMS0.950%，MAE0.899/255。Base一步不具有可用画质资格，未运行20步base或6步LoRA全图，也未做App UI验收。

## 当前瓶颈与下一步

已预热 edit仍有 input-ready **5.6524s**（包含等待attention/norm前图和当前权重，不能都归因CPU复制），producer **1.5164s**，并行branch **3.8858s**，consumer0.1826s。内部FFN总5.6199s与input-ready是不同窗口；encoder source-ready/branch窗口会重叠，不能把这些计数任意相加。

下一步优先：同层多chunk复用完成且source绑定的W8 bank，减少64次重复staging；再把权重staging提前到attention期间，建立真正双bank W2和冲突/取消证明。之后才扩展encoder专属校准、长prompt/其它reference-size分区、LoRA修正与App自动预热。当前是W1＋host source准备，尚非完整Splash Metal多层图；不证明ANE硬件INT8计算placement。

## 验证与清理

最终构建 `tc-runtime-build-v1-92e9f729b0b148f96db11e4c004c87d4435b3d8ca774331dce9bfcb388a6a0f4`，581个生产源文件前后匹配。
CLI/lib编译成功；API合同129项通过（3跳过），stage合同102检查，result metadata通过，generated短行/多块/变权重/别名拒绝/源准备异常后真实完成通过；非private分支syntax通过。

最初周期正弦生成样本超过8%门槛，保留失败回执，未放宽门槛；确定性随机样本的三种尾块RRMS约2.34–2.39%。测试请求中的视频/音频默认项、空LoRA策略和省略的零计数字段触发过harness校验，修正后只续跑尚未执行的部分，没有重复已完成GPU整图。

本轮按真实组件＋完整/资源状态验证两组推进；冷编辑负结果后仅补一组单步资源预热对照，无通道大扫描或长时间多步生图。清理owned对象/archive/module cache、无效中间构建、C3072及tiny程序：约 **75.85 MiB**。只保留两种有效C6144程序，共约 **12.02 MiB**，原始模型未动；系统driver/compiler cache未手动清理。

入口：`outputs/qwen21-native-prefill-ane-20261008`。最终可执行：`build/qwen21-native-prefill-ane-r4-20261008`；配套probe：`build/qwen21-native-prefill-probe-r6-20261008`。回执、源/二进制hash及全部限定在[状态JSON](2026-10-08-qwen-native-prefill-ane.json)。
