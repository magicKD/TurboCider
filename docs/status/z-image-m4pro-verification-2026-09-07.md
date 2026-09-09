# Z-Image-Turbo：M4 Pro 48GB / 512×512 验证

2026-09-07，在 `dev-verify` 的 `59d9c437` 基础上完成。完整数值见 [验证 JSON](../design/validation/z-image-m4pro-512-2026-09-07.json)；请求、原始日志、图片、模型映射及可复跑脚本位于本地 `outputs/z-image-m4pro-512-20260907/`。

本机基础 GPU、指定的 `NSFW_master_ZIT_000017532.safetensors` LoRA、三种基础 GPU/ANE 分区和 LoRA 绑定分区均实际生成了 512×512 PNG。最快的基础混合方案为 `hybrid_a8192`：暖请求 **15.768 秒**，相对基础 GPU **1.339×**。本次只验证固定提示词/种子的显式实验路线，没有把它加入 M4 Pro 自动加速策略。

## 环境与模型

- Apple M4 Pro，20 核 GPU、14 核 CPU、48 GiB 统一内存，macOS 26.6 (25G72)。
- TurboCider 原生 C++ / MLX 0.32.0 / Metal；coremltools 8.3.0 用于离线导出。Python ctypes 只驱动 C ABI 计时，神经网络推理运行于原生库。
- DiT 和 VAE 使用指定 `comfy_z_image_turbo/models/` 中的权重；Qwen3 文本编码器与 tokenizer 复用指定 `FLUX.2-klein-4B/`。
- 没有下载模型，也没有生成预合并的完整 LoRA checkpoint。`outputs/.../model/` 通过目录符号链接和同卷文件硬链接映射原权重，硬链接不复制权重数据。
- LoRA 大小 621,346,936 bytes，SHA-256 为 `36320cddae030103ff72b43c2878cb6659079bb7b742b9e90a94b5f9d6b6d33d`。180 组 rank-128 A/B 投影全部通过形状匹配，并由 native runtime 实际应用；strength=1.0，role=transformer，融合方式为 `in_memory_delta`。

## 测量条件与结果

统一使用 `A cinematic red fox walking through fresh snow, soft morning light.`，512×512、seed=42、9 步、resident、dynamic text。编码后 21 tokens；ANE bucket=1056，即 1024 个图像 token + 最多 32 个提示词 token。

每个配置在独立进程中创建一个驻留会话，先记录首次请求，再记录 3 次相同请求的暖运行；表中暖耗时为中位数。端到端计时覆盖整个 `tc_engine_generate` 调用及 PNG 导出，不含进程/engine 构造和离线分区导出/编译。首次请求包含文本编码、模型加载和需要的 Core ML 初始化，但没有清空系统文件及 Core ML 缓存，不能解释为整机冷启动。暖运行复用提示词 conditioning；更换提示词会重新进行文本编码。

最终序列运行时，所有后台导出已经结束，模型推理严格串行，未启用 tensor dump。早期不带 `clean_` 的报告受导出/构建干扰，保留用于诊断，不参与最终加速比。

| 配置 | 进程首次 / 秒 | 暖端到端 / 秒 | 暖去噪 / 秒 | 对对应 GPU 加速比 |
|---|---:|---:|---:|---:|
| 基础 GPU | 23.486 | 21.109 | 20.475 | 1.000× |
| 基础 GPU+ANE 40% | 27.857 | 18.768 | 18.157 | 1.125× |
| 基础 GPU+ANE 60% | 24.919 | 16.612 | 16.005 | 1.271× |
| 基础 GPU+ANE 80% | 39.694 | 15.768 | 15.160 | 1.339× |
| LoRA GPU | 26.790 | 23.457 | 22.553 | 1.000× |
| LoRA GPU+ANE 60% | 39.747 | 16.951 | 16.035 | 1.384× |
| 基础 GPU 末尾复测 | 24.983 | 22.394 | 21.782 | 1.000× |
| 基础 GPU+ANE 80% 末尾复测 | 25.674 | 15.993 | 15.365 | 1.400× |

末尾 GPU 复测相对开头的变化为 +6.09%。基础分区的加速比以开头 GPU 为分母，末尾 80% 分区复测单独以末尾 GPU 为分母；LoRA 混合以 LoRA GPU 为分母。每次请求均完成 9 步，不通过减少步数或尺寸获得加速。所有配置的四次重复输出是否逐文件确定性一致，均记录在验证 JSON 中。

## 并行执行与近似误差

每个 block 先完成 attention，随后将 FFN 的 intermediate 通道分成两部分：GPU 异步计算后缀，同时 Core ML 计算前缀，最后相加并进入 residual。这里实际重叠的是 FFN 两个分支；同一 block 的 attention 是其依赖前置步骤。32 个 block × 9 步，每张图片 **288 次 Core ML 调用**；所有混合配置累计 output copy 为 0。

4096/6144/8192 分别占 10240 个 FFN 通道的 40%/60%/80%。ANE 使用 INT8 per-channel 权重及 FP16 IO，GPU 使用 BF16；activation scale=8、output scale=32。LoRA 混合必须使用绑定该 adapter SHA、路径、role 和 strength 的独立 manifest，运行结果确认 `lora_identity_verified=true`，不能复用基础 manifest。

抽查基础 4096、8192 和 LoRA 6144 的 block 0/2/31：Core ML compute plan 将卷积、SiLU、乘法及 split 全部优先分配到 Neural Engine。运行时使用 `cpuAndNeuralEngine`，并记录实际预测调用；本次没有硬件 residency trace，因此保留运行时的 `observed_ane_residency=unknown`，不声称逐算子驻留已通过硬件计数器验证。

图片人工查看正常，shape 均为 512×512×3，runtime 检查 latent 与像素有限。量化后存在可见细节变化；以下是相同 seed 的 RGB 像素比较，LoRA 混合以 LoRA GPU 为参考，其余以基础 GPU 为参考：

| 混合方案 | MAE / 8-bit | Pearson correlation | PSNR / dB |
|---|---:|---:|---:|
| hybrid_a4096 | 4.574 | 0.969321 | 24.90 |
| hybrid_a6144 | 4.011 | 0.975728 | 25.91 |
| hybrid_a8192 | 3.174 | 0.987425 | 28.77 |
| lora_hybrid_a6144 | 7.517 | 0.954564 | 23.38 |

这些数据不代表与 GPU 逐元素一致，也不能替代多提示词/多种子的质量评估。本次未重新运行 ComfyUI 或外部原始框架，因此不作与它们的数值一致性或速度比较。MLX 峰值约 14.94 GiB，仅覆盖 MLX allocator，不包括 Core ML、OS 缓存；验证 JSON 另外保留进程峰值 RSS，仍不包含系统服务进程。

## 构建修复与验证边界

原分支调用了带 `force_fused` 的新版 MLX SDPA 签名，但项目锁定并安装的 MLX 0.32.0 没有此参数，导致 C++ 构建失败。`native/backends/mlx.cpp` 增加编译期签名适配：旧版使用默认 kernel selection，新版保留显式 `force_fused`。新版签名的分支没有在本机另一套 MLX 上实测；本次全部数字对应 0.32.0。

本机 Xcode framework 存在加载兼容问题，使用已有 CLT 与 SDK 完成构建：

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
tools/native/build.sh
```

原生库、CLI、Swift App 及集成测试程序均构建成功；本次出图验证走公共 C ABI/CLI 引擎，没有把它表述成 App UI 操作验收。Metal self-test 通过；Z-Image checkpoint 测试 5 项、Core ML LoRA 测试 4 项、inventory 测试 2 项通过。`make test` 其余已执行检查通过，但 LTX vendored `ltx_transformer_io.c` 与 `ltx_shaders.metal` 和相邻 `ltx-mac` 不一致的两个断言失败；这些文件未在本次修改。

## 本机复跑

从仓库根目录使用已经准备好的请求：

```sh
build/native/turbocider generate \
  outputs/z-image-m4pro-512-20260907/model \
  outputs/z-image-m4pro-512-20260907/clean_hybrid_a8192_0.json

.venv/bin/python outputs/z-image-m4pro-512-20260907/bench.py clean_hybrid_a8192
.venv/bin/python outputs/z-image-m4pro-512-20260907/bench.py clean_lora_hybrid_a6144
```

CLI 的单次 `generate` 对应进程首次运行，连续暖运行使用 bench 或复用会话的 `batch`。复跑会更新这些实验输出，若需保留，请先复制请求并改输出路径。请求中包含当前机器的绝对 manifest 路径；迁移目录后需要重新生成请求/编译索引。当前 bucket 只适用于该尺寸且编码后不超过 32 tokens；更长提示词需要导出更大的 bucket。显式混合使用 `allow_approximation=true`。
