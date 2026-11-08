# Qwen Image 2.1 DiT W8A8 与离线缓存实测

实测日期：2026-10-16。基于已合并 PR #12 后的主干 `0ed94404c202ac3f3a3b5a5da33bf3a56453ccef`。

在 **Apple M5 Pro、24 GiB、无 LoRA、512×512、40 步、流式加载与分阶段释放、GPU＋ANE 开启**的同批次狐狸样本中，DiT W8A8＋离线缓存将现有 W4 组合的整次请求中位数从 **29.821 秒降至 24.009 秒，减少 5.812 秒（19.49%）**。文本编码器和 VAE 保持原实现及原精度。

最终组合是 **DiT W8A8＋离线权重缓存＋Q/K norm RoPE 融合＋DBCache fast＋原有流式 ANE**。W8A8 替换原来的 attention / GPU FFN suffix W4，不会同时叠加两种权重量化。该组合相对本轮原始流式 ANE 中位数减少 **46.336 秒（65.87%）**；原始路线波动为 55.811–87.436 秒，比例仅描述这批实测。

## 实际改变了什么

GPU 上可量化的 DiT Linear 包括首步的输入、时间、调制、attention Q/K/V/输出、FFN 与最终输出投影，以及后续 decode 的 attention 和 GPU FFN suffix。权重按输出通道采用 INT8，激活按 token 动态量化为 INT8，再使用 M5 Metal INT8 TensorOps 矩阵乘。累加结果反量化为 FP16，按原接口恢复 BF16 或 FP16；norm、RoPE、softmax、attention 中动态 QK/AV 矩阵乘及非线性仍走原路径。已有 6144 通道 ANE prefix 及其冻结 W8A8 图保持原样。

量化公式为 `scale_w = absmax(FP32(weight), K) / 127`，全零通道使用 scale 1；`q_w = INT8(clip(round(weight / scale_w), -128, 127))`。激活在 FP16 输入上逐 token 求 absmax，再量化为 INT8。N/K 补齐到 128，单行输入也补齐到 32 行，确保小投影实际执行 A8 GEMM，避免落入上游的浮点激活 MV 分支。没有校准集、percentile clipping、LoRA 合并或文本/VAE A8。

内核来自 [Mininglamp-AI/cider](https://github.com/Mininglamp-AI/cider/tree/4d91fcee9439f7aea17ae6e965271d9536c604a0)，固定到 `4d91fcee9439f7aea17ae6e965271d9536c604a0`。仅接入 per-channel primitive 和所需 Metal 源码；内核嵌入原生库，客户端运行不依赖源码目录、Python Cider 包或实验环境变量。MIT 许可随源码、App 和 CLI 分发。

## 为什么有效

原 attention / suffix W4 路线缓存了 decode 权重，但首步仍执行 BF16 投影，并在每个新请求中重新打包压缩权重。本方案将首步 Linear 也接入 W8A8，同时把最终 INT8 权重、FP32 scale 和零 bias 落盘。命中时直接读取这三个已打包数组，跳过源权重的求值与量化；decode attention 和 suffix 在请求内复用，block 首步临时权重继续逐层释放。

本轮现有 W4 的 denoise 中位数为 18.583 秒，W8A8 缓存为 13.674 秒。每次命中读取 392 个条目，未再写入；首次量化与保存累计 13.520 秒。这里实测的是内核、首步覆盖和离线缓存的组合收益，不能把全部增益归给 INT8 算力，也不能把历史单项百分比相加。文本和 VAE 阶段的小幅计时差异属于同机测量波动。

## 三组完整请求对比

| 配置 | 第 1 / 2 / 3 次整次请求秒数 | 中位数 秒 | MLX 峰值中位数 GiB |
| --- | --- | --- | --- |
| 原始流式 ANE | 55.811 / 87.436 / 70.345 | **70.345** | 5.665 |
| 现有 W4 组合，新开关关闭 | 26.518 / 37.124 / 29.821 | **29.821** | 5.429 |
| DiT W8A8＋离线缓存，新开关开启 | 24.993 / 24.009 / 23.450 | **24.009** | 5.429 |

原始组关闭 staged fast defaults、W4 权重缓存、Q/K 融合与 DBCache，仍保留相同的流式加载、分阶段释放和 ANE 分区。现有组为 attention W4＋suffix W4＋Q/K 融合＋DBCache fast；新组只切换 `qwen21_dit_w8a8=true`，DBCache fast 和 Q/K 融合保持开启。

三轮顺序分别为「现有、新、原始」「新、原始、现有」「原始、现有、新」。每个单元均为新进程的第一次生成请求，缓存命中组复用磁盘量化产物；不清空操作系统文件缓存。计时使用 `request_wall_including_export`，覆盖加载、源模型 SHA 校验、ANE 准备、文本编码、DiT、VAE 和 PNG 导出，排除进程启动及 engine 创建。桌面后台活动正常运行，没有并行编译或其他本任务的推理。每组三张 PNG 各自逐字节相同。

环境为 macOS 26.4.1（25E253）、MLX 0.32；物理内存 25,769,803,776 字节。原始路线的高波动也体现在进程读取约 226 GB 和系统压缩计数变化中；进程 I/O 计数不等于物理 SSD 流量，系统 VM 计数不能归因到单个进程。表中总增益不代表所有提示词或后台负载下的固定比例。

## 首次生成和持久缓存

首次启用、缓存为空的整次请求为 **37.766 秒**，其中量化与写入累计 **13.520 秒**；MLX 峰值 **5.910 GiB**。这是一次首次缓存生成，已有 Core ML 分区已预编译；不包括模型下载或 Core ML 分区导出。它比本轮现有 W4 中位数慢，适合在接受一次准备成本后重复使用。

共 **392 个 safetensors 数据文件、11,694,466,156 字节（10.89132 GiB）**：首步/非 block Linear 200 个、decode attention 128 个、GPU suffix 64 个。缓存位于用户的 `Library/Caches/TurboCider/qwen21-dit-w8a8/<recipe>/<DiT SHA256>/`。关闭开关保留磁盘文件，后续开启可复用；原始 BF16 模型仍须保留。

配方 ID 为 `dit-pc-w8a8-absmax127-nk128-m32-v1`。目录和每个条目的 metadata 同时绑定经过 HybridSession 完整校验的 DiT SHA、配方、投影名称、源 dtype、原形状及 padding。读取验证数组数量、dtype、shape、正有限 scale 和零 bias；损坏、错模型或错配方明确报错。每条目加锁，临时文件写完后原子发布；取消不发布半个条目，完整条目可留待重试。条目未做逐次全载荷内容哈希；这里的校验不是对任意权重位翻转的完整检测。

## 图像偏差

RGB RMSE 以 0–255 像素值计算，相对同 seed 的原始流式 ANE 输出。它反映与该输出的像素差异，不等同于感知质量或模型准确率。

| 样本 | seed | 新 W8A8 RGB RMSE | 实际检查 |
| --- | --- | --- | --- |
| 狐狸与雪地 | 42 | 6.676470 | 三轮及首次缓存生成逐字节相同 |
| BLUE SKY 白底蓝字 | 13 | 4.229184 | 两个单词拼写正确 |
| 红色方块、蓝杯、黄球 | 17 | 4.506536 | 颜色和从左到右顺序正确 |

狐狸样本的现有 W4 组合 RGB RMSE 为 **22.600062**，新组合为 **6.676470**。本轮狐狸原始基准重新生成；文字和物体沿用 2026-10-15 同模型、同参数的原始流式 ANE 基准。三个新输出均与先前 DiT-only W8A8 探索输出一致；没有接入文本编码器或 VAE 的有损实验。

提示词分别为：

- `A red fox standing in fresh snow, soft morning light.`
- `A clean white poster with the words BLUE SKY in large blue letters, centered composition, no other text.`
- `A red cube, a blue ceramic cup and a yellow ball arranged left to right on a wooden table, studio lighting.`

## 适用条件与客户端对比

开关默认关闭，且保留到草稿、请求和历史记录。客户端和原生请求同时限制为 **Apple M5 Pro、精确 24 GiB、无 LoRA、512×512 文生图、20–40 步、MLX、`component_staged`、冻结 W8A8 ANE 开启**，并关闭参考图、提示词增强、设备配置及冲突实验。流式文本/DiT 与分阶段释放仍然启用。其他设备（包括普通 M5、M5 Max 或不同内存）、常驻模式、纯 GPU、Runtime ANE 及 LoRA 请求不能启用该选项。

客户端打开「推理加速与准备」，启用「额外启用 ANE」，选择匹配的已编译分区，并开启「Qwen 固定量化分区（实验）」。使用 512×512、40 步、「分阶段加载（推荐）」和固定 seed，关闭 LoRA、参考图和提示词增强；然后切换 **「DiT W8A8＋离线缓存」**。对比本报告的最终组合时，前后均选择 DiT 缓存「快速」。新开关不自动改变 DBCache 档位；不满足条件时无法新开启，已保存的开关仍可关闭。

原生 v1 参数为 `qwen21_dit_w8a8`，v2 位于 `execution.qwen21_dit_w8a8`。可用现有 `tools/native/benchmark_native.py`，相同 request / prompt / seed 下分别设置 false、true，使用 `--runs 1 --record-vm`，各运行三个独立进程。缓存首次准备应单列。客户端关闭开关时省略该扩展字段，保持已有 Runtime worker 协议兼容。

## 真实执行与生命周期验证

40 步 fast 请求均实际执行 40 步、DBCache 命中 25 步，Core ML runtime 调用 **648** 次，另有 **32** 次构造 warmup，运行失败 0。20 步 fast 为 **19.242 秒、8 次缓存步、416 次 runtime 调用**；40 步关闭 DBCache 为 **31.471 秒、0 次缓存步、1248 次 runtime 调用**。两种边界配置均命中 392 个权重条目，attention/suffix 均覆盖 32 层。Core ML 使用 `cpuAndNeuralEngine`，manifest 与调用数已验证；这些回执没有测量芯片内每个算子的物理落点。

同一引擎依次执行「文本第 4 层取消 → W8A8 重试 → 关闭开关 → W8A8 第 12 步取消 → 再次重试」。两次取消返回状态 2 且没有导出图片；三次完成输出分别与对应的正常请求逐字节相同。最大闲置 MLX active 为 **166,510 字节**，engine 释放后为 **2,670 字节**。每次 denoise 的 6 GiB wired limit 均恢复到测试调用者设置的 256 MiB，结束后恢复原调用者限制。

本次共完成 **18 次真实模型生成和 2 次取消**：14 次基准/质量/边界生成、3 次同引擎完成请求，以及 1 次打包 App 内引擎的 20 步生成。打包引擎命中 392 个缓存条目、执行 416 次 runtime 调用，PNG 与未打包 20 步结果一致；该独立 CLI 进程总耗时为 32.663 秒，包含启动和 engine 创建，未混入上述请求计时中位数。

自动检查包括：107 个 Qwen 原生测试（7 个可选条件跳过）、49 个 Qwen 请求契约测试、8 个运行时构建身份测试、2 个 C++ 边界测试、Qwen 客户端工作流测试及 prompt parser/sampling fixture。新增 GPU fixture 验证 M=1/37/129 的实际 A8 结果、padding、compiled/eager 一致、零输入、缓存命中、错 SHA/配方/scale、截断文件与取消后恢复。正式 App 构建和本地打包签名校验通过。

## 复核定位

- ANE manifest SHA256：`4fba06ff0da1c432f317c7713946f888917b24b44751a3f48868827ba1f52fc0`；32 层、1024 行、hidden 4096、prefix 6144、`int8_pc` / `sq_v1_both`。
- DiT 模型 SHA256：`89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc`。
- 实测未打包动态库 SHA256：`80eacdfc7eb769666f62e36cf35bf1a9157d846ec0c081ce65f8ed24a450730e`。
- Runtime build ID：`tc-runtime-build-v1-aad807667658cf9af29b4c19ddec0fb7220c9b4990e99e719b450ccc1553ee9f`；实测后逐项核对 native/build 输入与 manifest 一致。
- 狐狸原始 / W4 / W8A8 PNG SHA256：`c9f90f1821b8044ef5a2d73b55a431ed30cf6b467d7c0559dc251071ba659d6c` / `e3b89b59f2f61cb03c5a45a58ade5fe853390d509be2ab7e0f283636f3aa7380` / `a613428414c843a889d60831548b4298300c4e4dae01b29600cc7ac409ca2913`。
- 实现：[请求资格限制](../../../native/models/qwen21/memory_policy.hpp)、[缓存与 INT8 适配](../../../native/models/qwen21/cider_w8a8.cpp)、[DiT 路由](../../../native/models/qwen21/transformer.cpp)、[客户端开关](../../../apps/macos/AccelerationView.swift)。
- 既有四项报告保留：[GPU suffix W4](01-gpu-suffix-w4.md)、[attention W4](02-attention-w4.md)、[Q/K norm RoPE](03-qk-norm-rope.md)、[DBCache](04-dbcache.md)。这些是各自测量条件下的历史记录。
