# Native register W1：首次完整 Base 请求净收益

2026-10-08，M4 Pro / 48 GiB。独立 native Metal DiT 的 GPU/ANE 路线完成一组同构建、同 seed 的 Qwen Image 2.1 Base 整图对照：**去噪省时 13.60%，完整请求省时 9.97%**，最终图片肉眼接近。机器记录见 [JSON](2026-10-08-native-register-w1-base-e2e.json)，完整设计见 [独立后端规格](../design/qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)。

这次只跑一组真实 FFN 筛选和一组配对整图，没有下载模型、复制完整 checkpoint 或导出全模型。当前成功路径值得继续开发；每条路线仅一个整图样本，不能授稳定性能、默认选择或产品发布资格。

## 实现与构建

当前构建身份为 `tc-runtime-build-v1-4fd1e980da701fa564cafc1f77702fcba49542d251cb5324ce42907c48324c23`。native/private 完整构建通过，562 项构建源在记录和清理时均与 manifest 摘要一致。使用进程内 CLT 选择和 native/package-only 构建；本次没有重新构建或验收 App。

`PreparedAneStages` 新增显式 register W staging：H128／H512、BF16 dense source、Sylvester basis，每个 block 32 个 SIMD lanes；两份特化 library 在 prepare 增加四个 PSO。复用原 register 算术，按 native entry 重排 buffer／parameter slots，保留 source offset/pitch、舍入、status flags 和 alias 合同。**A8／restore 保持 generic，默认 W staging 也仍为 generic**。没有新增 buffer、Surface 或 phase allocation。

完整 native DiT／Euler 由自有 Metal backend 执行，GPU complement 与一次性 private ANE invocation 通过已有 handoff／物理完成协议连接。模型加载、文本 encoder 和 VAE 仍通过 MLX 在阶段边界运行；本轮不是全 App 脱离 MLX。控制面仍为 **W1S 同步逐层主机准入**，不是 Splash 式多层 W2 设备事件图。

实验请求使用 `qwen21_denoiser_backend="native-metal-graph"`、`execution="gpu_ane"`、`qwen21_w8a8=true`、明确的绝对 owned ANE cache 路径和 `qwen21_native_w8_staging="register"`。该字段只允许 native private W1 域；类型、枚举、schema 1／2 位置和域外显式 generic 均校验。实际执行结果独立报告 staging、完成计数和设备跨度，不以请求字段代替执行证明。

## 有界组件筛选

不加载模型的检查覆盖 412 项 stage source 检查、82 项 layout 检查、严格 `.mm` 编译、native register 域合同、8 项请求路由测试、129 项通用合同（3 项跳过）和 5 项 E2E 监督测试。

生成数据的 GPU staging 测试只有四个 case：有限值、Inf、NaN、scale overflow。generic/register 的 codes、scales、padding 和 flags **逐字节一致**，source 保持不变；8 次配对 GPU submission、64 个 tiny dispatch，无 ANE request、无模型加载。回执见 [byte equivalence](../../outputs/native-w1-register-20261008/stage-byte-equivalence.log)。

生成数据的完整 NativeDenoiser 为 H128/F1024/L2、target 32 / text 7，两条路线各运行两步。hybrid 实际 register、完成 2 个 FFN／1 个稳态 step，最终 latent RRMS **0.2765%**。source/padding、物理完成、phase backing 计划与同一账本释放检查通过。tiny 稳态 GPU 约 2.06 ms、hybrid 3.70 ms，仅说明小尺寸的提交成本和完整路径成立，不用于模型性能排名。

真实 FFN 仅读本地 layer 0 BF16 gate/up/down（288 MiB）及已有 step 19 输入（8 MiB）。M1024/H4096/F12288/A6144，固定同一个输入；一次 warmup AB 后各记录一次 AB／BA，共六次完整 FFN，包含 W8/A8、submit、GPU 补集、restore/join、finite 和物理完成。

| 相同 layer 0 / step 19 输入 | 完整 GPU | 完整 register W1S | 省时 | RRMS |
|---|---:|---:|---:|---:|
| 记录 AB | 53.965 ms | 26.010 ms | 51.80% | 0.27265% |
| 记录 BA | 41.039 ms | 35.377 ms | 13.80% | 0.27265% |

两次都有净收益，但差异明显，不拼成稳定加速倍率。producer 实际 GPU Ticket 跨度为 4.041／11.588 ms，GPU complement 为 20.480／21.862 ms；host producer 为 4.270／12.266 ms。波动在 GPU 跨度内也存在，没有频率、cache 或调度的归因证据。当前 producer device span 是首 command buffer 开始至末 buffer 结束的区间，可能包含间隔，不能称 kernel busy time。

真实回放 cold prepare 单列：dense PSO 0.088 s、hybrid program/PSO 1.234 s、source 物化 0.021 s。source/input 前后摘要一致，held descriptor／canonical stat generation 成立，全部物理完成，owned ledger 归零。完整回执见 [real FFN](../../outputs/native-w1-register-20261008/real-ffn-receipt.json)。

## 完整请求对照

同一 build，512×512、20 steps、seed 42、24 text tokens，prompt 为 `A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.`，DiT cache off，顺序 native GPU→native hybrid，各一次新进程。shared OS／driver cache 未重置，因此不构成严格全冷／随机顺序统计，也不构成 generic/register 单项消融。

| 当前完整 native 请求 | GPU | GPU/ANE register | 本次变化 |
|---|---:|---:|---:|
| 去噪 | 41.617 s | 35.957 s | 省时 13.60%，1.157× |
| 完整请求 | 48.602 s | 43.758 s | 省时 9.97%，1.111× |
| Native 阶段准备 | 0.459 s | 1.329 s | 增加 0.871 s，已计入完整请求 |
| 文本编码 | 2.792 s | 2.767 s | 仍为 MLX GPU |
| VAE decode | 0.996 s | 1.005 s | 仍为 MLX GPU |

两条路线均实际完成 20 步、导出 PNG、退出 0。hybrid 首步 GPU、随后 **19 个 hybrid step／608 个成功 FFN**，实际 backend 为 `native_metal_graph+private_ane_w1+mlx_encoder_vae`，staging 为 register；608 个 FFN 均有双侧正值 GPU Ticket 跨度。所有 phase 最终 succeeded、无 quarantine／taint，native storage、reserved、pending、unknown 和 site active 计数归零。

本次最终 RGB 相对 L2 **0.868%**，平均绝对差 **0.527/255**，PSNR **46.00 dB**。肉眼检查两张图的主体、茶壶轮廓／盖子／把手／壶嘴、反光和桌面构图接近，没有发现明显对象缺陷。最大像素差仅作诊断，不作为逐像素 8% 门槛；RGB 诊断也不是 latent 或全量质量认证。见 [图像检查](../../outputs/native-w1-register-20261008/e2e-pair/image-comparison.json)。

| Native GPU | Native GPU/ANE register |
|---|---|
| [GPU 图片](../../outputs/native-w1-register-20261008/e2e-pair/native.png) | [Hybrid 图片](../../outputs/native-w1-register-20261008/e2e-pair/hybrid.png) |

完整 [监督回执](../../outputs/native-w1-register-20261008/e2e-pair/receipt.json) 保留 request、计划、执行、原始阶段内存 trace 和前后观测；测量二进制、源摘要与 tool SHA 记录在状态 JSON。前一构建的组件与 MLX 对照仍绑定各自构建，不能混为此次同构建三方排名。

## 内存与剩余瓶颈

hybrid 新增 capacity upper **122,798,080 B（117.109 MiB）**，同一 phase ledger 的实际 owned 峰值从 215.306 增至 332.339 MiB（增加 117.033 MiB）。其中 9 个 Surface 共 89,161,728 B，两个 FP32 partial 共 33,554,432 B。没有为 32×20 次 FFN 保存多份完整权重 bank。

DiT phase 采样进程峰值由 13.646 增至 13.767 GiB；`/usr/bin/time` 整请求 peak footprint 分别为 15,415,979,360／15,416,192,352 B，最高阶段可能在 encoder／loader，不能按这两个近似相等的总峰值否认 DiT 的局部增加。各观测范围均保留，MLX allocator 峰值不替代全进程峰值，1 GiB framework/driver guard 及完整请求预算仍未认证。

本次系统 swapin／swapout 计数未增、swap 用量保持 3305.75 MiB，但已有 swap、压缩和 pageout 活动；不宣称零换页或无内存压力。`pmset` 前后没有记录 thermal/performance warning，不等于连续硬件 tracing。四份模型／tokenizer 来源 stat 前后不变，没有重新哈希完整大 checkpoint。

整图 hybrid host producer 合计 19.263 s、对应 GPU 区间 15.755 s；production producer 还包含 attention/norm，**不能全部叫 W staging**。complement GPU 区间 12.457 s，host branch join 12.571 s；后者不是 ANE 独立计算耗时。host total 32.465 s 是 FFN 控制器累计，不能和 device spans 相加推算硬件 overlap。收益还受 attention、重复 W 编码、逐层主机准入以及冷 program/PSO 成本限制。

下一项先独立实现受 source/recipe 约束的 compact row-scale 复用，把缓存 backing 与恢复到 pitched bank 的拷贝计入同一 phase。32 层有效载荷约 1 MiB，按层合并的上界约 1.5 MiB；直接缓存完整 W8 codes 约 2.25 GiB，不作为本机默认方案。随后比较有界 prepared-session 和下一层 staging。当前测量不承诺这些候选的收益。

Base 成功路径之后，再接入 native GPU LoRA 全目标、pre-SiLU/down 语义和 hybrid ABI，并单独验收；编辑／encoder、Z-image、W1R/W2 设备图以及 App 高级配置继续分别推进。硬件 INT8 arithmetic、真实 ANE placement/overlap、异常／取消／late callback 设备路径尚无本次验证结论。

## 清理

实验结束后删除本轮明确归属的 **248 个文件／33,710,258 B（32.15 MiB）**：object/static archives、module cache 和三个 owned ANE program cache。保留测量 binary、runtime manifest、library/CLI、图片及回执，并校验摘要未变。清理明细见 [owned cleanup](../../outputs/native-w1-register-20261008/owned-cleanup.json)。用户模型、历史实验和系统 driver cache 不变；本轮不再运行额外模型实验。
