# Native LoRA SGMMA：单组整图验证

2026-10-08。**当前构建的独立 native DiT 后端成功完成 Viggle r128 的六步整图，实际执行配方、adapter 来源、原生阶段释放和图像检查通过；本次完整请求比同构建 MLX 路线慢 14.44%，尚未实现整图净加速。** 保留显式实验选择，不据此改变默认路线或授予自动选择资格。

原始证据：[整图监督器回执](../../outputs/native-lora-sgmma-20261008/e2e-pair/receipt.json)、[执行日志](../../outputs/native-lora-sgmma-20261008/e2e-supervisor.log)、[图像比较](../../outputs/native-lora-sgmma-20261008/e2e-pair/image-comparison.json)。完整数值、来源身份与 10 项小型证据文件 SHA256 记录在[机器汇总](2026-10-08-native-lora-sgmma-e2e.json)。没有修改原始测量回执。

## 构建与范围

两路线使用同一 CLI／library，实际构建为：

```text
tc-runtime-build-v1-77863b3f55434af37ee12615039845b576878a6f310e39656bc2bc1cf308c2fb
```

顺序为 **MLX → native SGMMA**，每路线一个新进程、一个样本，无重试、扫描或额外 warmup。共同设置为本地 Qwen Image 2.1 BF16、Viggle v0.2.1 r128、strength=1、512×512、6 步、seed=42，无参考图；prompt 为 `A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.`。采用 `component_staged`、纯 GPU、`compile_gpu=false`、DiT cache off；`allow_approximation=true` 仅许可既有六步蒸馏日程，不启用近似 DiT 缓存或 W8A8。

所有模型和 adapter 已在本地，没有下载、checkpoint 副本或新模型导出。进程彼此独立，但共享 OS／文件／driver 缓存，未重置这些缓存，不能称两路独立冷启动对照。监督器先通过 [18 项 mock CPU 测试](../../outputs/native-lora-sgmma-20261008/supervisor-cpu.log)，测试没有运行设备或加载模型。

## 实际执行与来源

native 请求显式选择 `qwen21_native_lora_kernel=sgmma`。验收读取实际 `native_denoiser_lora`，不是只检查 request／plan：

- `runtime_backend=native_metal_graph+mlx_encoder_vae`，`gpu_graph=native_full_dit_euler_runtime_lora`，实际完成 6 步。完整 DiT 由独立 native graph 执行，文本编码和 VAE 仍采用 MLX。
- 实际配方为 `runtime-lora-fp32-sgmma8x8-aligned-r128-scalar-tail-ordered-bf16-bias-last-v2`。它包含按形状回退 scalar 的策略；不能据这个 bundle 标识声称每一个 projection 都走了 SGMMA。
- 一份 adapter、195 个 resolved bindings、227 个 logical projections／ordered terms、max rank128、strength1；实际 LoRA scratch 为 **52,047,872 B**。该数值不是全部 DiT 或整进程内存。
- `source_kind=verified-held-fd`、`source_generation=3`、`completed=true`。adapter SHA256 为 `bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`；actual adapter revision 为 `9cefbea6f41fa5f4cebcda4983133cf6f45d23723b2ee6b8c7e78a9718ce5d75`；verified artifact digest 为 `d2ba4f7b04f3025b1c04b7a13825267f5f581f7c5616c36e23ba688b95378c80`。

监督器核对 DiT、文本编码器、VAE、tokenizer 和 adapter 五份原始文件的前后 stat 一致，build artifact 身份也未变化。监督器没有重复哈希大型模型。MLX 路线使用原有运行时 adapter pin 校验并实际应用 227 个 projections，不能把 native 的 held-fd／phase 证明移用于 MLX。

## 时间与内存

| 口径 | MLX | Native SGMMA | Native 相对 MLX |
|---|---:|---:|---:|
| 实际 `request_wall` | 20.959514250 s | 23.986534208 s | 慢 14.4422% |
| 实际 `denoise` | 14.325801417 s | 16.795572375 s | 慢 17.2400% |
| 监督器进程 wall | 21.658253792 s | 24.070941167 s | 包含启动／退出 |
| 外部 `time -l` real | 21.64 s | 24.06 s | 进程整体口径 |
| `time -l` peak footprint | 18,313,032,904 B | 15,415,881,032 B | 本样本 native 较低 |
| `time -l` maximum RSS | 12,499,763,200 B | 13,619,920,896 B | 本样本 native 较高 |

native phase prepare 为 **0.626528625 s**，包含于 request wall、位于 denoise 计时之外，不能再次加到 request wall。两路 denoise 的收尾边界也不完全相同，因此判断完整请求收益以 request wall 为主。`time -l`、监督器 wall、模型内部计时不是可叠加的阶段。

native 阶段 `execution_state=succeeded`、`tainted=false`、`worker_quarantined=false`；物理完成后 tracked native storage count／bytes、reserved bytes、pending release bytes、unknown bytes 均为零。**这里归零的是受跟踪的原生分配**，不是进程内存；源权重和输入处在 retained baseline，账本保留的 baseline 为 15,094,213,840 B。native tracked peak 超出 baseline 的部分为 277,816,320 B；1 GiB framework guard 仍是未认证 allowance，`full_request_certified=false`。

两路 process swaps=0，`vm.swapusage` 前后均为 used 3297.75M；native phase 自身 `swap_observation_available=false`。`pmset` 没有记录 thermal／performance warning，这些观测不证明整个系统没有压力，也不足以判断 GPU 频率或解释内核耗时。footprint、RSS、MLX allocator 和 native ledger 的覆盖不同，不互相替代。

## 图像与组件证据的边界

两张导出图均为 512×512 RGB。相对 MLX 图的 RGB relative L2 为 **1.034864%**，PSNR **44.64296 dB**，MAE **0.66840／255**、最大单通道差 **63／255**。主线程已目视检查：同一蓝色陶瓷茶壶、木桌、壶嘴／把手／壶盖及柔和日光，构图与反光细节接近。这是一个 prompt／seed 的最终图像诊断，不代表像素逐位一致、中间张量等价或一般质量资格；原监督器的 `image_quality_verified=false` 保持不变。

此前同构建的[真实 layer0／step5 完整 FFN 对照](../../outputs/native-lora-sgmma-20261008/real-ffn.json)已通过六次调用的有限值、来源、padding、物理完成及同账本释放检查；scalar／SGMMA 输出逐位相同。正式 AB 为 54.993458 → 51.002875 ms，BA 为 54.341333 → 52.391917 ms，SGMMA 分别快约 7.26%／3.59%。这些是该层完整 FFN 的有限组件收益；本组整图结果说明它们尚未转化为相对 MLX 的净收益。

**当前构建没有 scalar 整图对照。** 不能拿旧 `6d914…` 构建的 23／25 秒级历史结果与本轮相减，宣称因果上的 scalar→SGMMA 整图加速；也不能从当前 MLX→SGMMA 差值推断 SGMMA 比 scalar 慢。尚需从已有时间边界、图组织与低秩路径阅读中定位剩余成本，本轮不继续追加设备实验。

该结果不覆盖 ANE、GPU／ANE 物理并行、private/fixed/public ANE 的优劣、W2 深 graph、编辑、其他 adapter／rank／strength、Z-image 或 App 验收，不授予性能、内存容量或自动路线资格。

## 清理

设备工作完成后，主线程完成[本轮 owned 清理](../../outputs/native-lora-sgmma-20261008/owned-cleanup.json)：删除 **244 个文件、29,332,442 B（约 27.97 MiB）**，仅涉及新建 `.o`／`.a`、module cache 和显式 generated tensor fixtures。测量 binary、runtime CLI／library／manifest、原始模型／captures、raw receipts／日志及两张 E2E 图像均保留；未删除原模型或系统缓存。清理回执 SHA256 为 `b14ca812f398ae95f6cc290d0a392bbaac3683aa81e409ea13a8cc764cfa1a05`。
