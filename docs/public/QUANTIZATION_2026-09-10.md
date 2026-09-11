# Z-Image / Qwen3 本机量化与文本后端评测

日期：2026-09-10。机器：Mac Studio，Apple M4 Max（16 核 CPU），64 GiB
统一内存；MLX 0.32.2。以下是本次实测，不是其他 Mac 或其他模型的性能承诺。

## 已接入的模型管理

- Z-Image 下载默认使用 ModelScope 的 `Comfy-Org/z_image_turbo`，独立选择
  BF16、INT8 ConvRot、NVFP4。不同组合建立独立安装，VAE/tokenizer 可去重共享。
- 原生加载、目录安装和安装检查均识别 INT8-only / NVFP4-only 目录。
- Qwen3-4B 可选 BF16 或本地 MLX affine Q4/Q8 组件。Q4/Q8 通过
  `tools/convert/qwen3_affine.py` 离线生成，保留 BF16 embedding，group size 32。
  **目前不是 App 内一键转换，也不是直接下载任意第三方 Q4/Q8 权重。**
- Comfy 源不含 tokenizer，下载配方从 `Tongyi-MAI/Z-Image-Turbo` 补充两个
  tokenizer JSON；实际 ModelScope helper 预览通过，共五个所需文件。
- root 下的共享 `text_encoder/` 优先于 Comfy BF16 text 文件，避免所选 Q4/Q8
  被静默忽略。共享组件检查拒绝不符合本转换约定的 quantization 配置。
- NVFP4 标记为实验性 GPU W4A16；此版本的 LoRA / ANE 请求会明确拒绝。

完整操作说明见 [模型库](MODEL_LIBRARY.md)。磁盘大小不等于运行内存。

## 图像模型：更小不代表更快

固定提示词 `A cinematic red fox walking through fresh snow, soft morning light.`，
seed 42，512×512，9 步，GPU，无 LoRA/ANE。每组合独立进程，1 次冷请求及
3 次同提示词暖请求，报告暖请求中位数。三个暖请求命中 TurboCider prompt cache，
**不能用其 text_encode≈0 计算编码器加速比**。PNG 导出计入外部时间。

| DiT / Qwen | DiT 文件 GB | 冷请求 s | 暖出图 s | MLX 峰值 GiB | 相对全 BF16 的图像像素 cosine |
|---|---:|---:|---:|---:|---:|
| BF16 / BF16 | 12.31 | 11.117 | 8.472 | 14.928 | 1.000000 |
| INT8 ConvRot / BF16 | 6.20 | 11.231 | 9.344 | 9.217 | 0.999763 |
| NVFP4 W4A16 / BF16 | 4.51 | 11.673 | 9.405 | 7.690 | 0.984688 |
| BF16 / Q4 | 12.31 | 9.803 | 8.473 | 14.928 | 0.991683 |
| BF16 / Q8 | 12.31 | 9.641 | 8.478 | 14.928 | 0.999331 |
| INT8 ConvRot / Q4 | 6.20 | 11.171 | 9.344 | 9.217 | 0.988801 |

冷请求含权重准备、磁盘/OS 缓存影响，顺序执行但未清空系统文件缓存，不能把这些
冷时间差直接归因于量化。MLX 峰值来自 allocator，不含完整 OS/Core ML/file-cache
内存；也不是最低物理 RAM 要求。进程 RSS 原始值另有记录，但 macOS Metal 共享/
映射内存的 RSS 记账可能明显小于 GPU allocator，不能把两者混用或相加。

Z-Image 会在编码后释放 text encoder，因此 Q4/Q8 降低编码阶段内存，不一定降低
由 BF16 DiT 主导的整图峰值。本样例六种组合都生成了可辨识、符合提示词的狐狸。
INT8 与 BF16 很接近；Q8 文本变化较小；Q4 文本和 NVFP4 改变了姿态、毛发与背景。
像素 cosine 是偏离基线的指标，**不是审美、文字遵循度或模型质量评分**。
这里没有多提示词/多种子统计，也没有中文排字、人物或视频的量化质量验证。

## MLX 是否原生支持 NVFP4？

**本机 MLX 0.32.2 的 NVFP4 算子可以在 M4 Max 上实际执行。**
实测 `quantize` / `dequantize` / `quantized_matmul(mode="nvfp4")`，并非只检查 API。
128×2048 乘 2048×2048 的小型 GPU 运算也成功；小算子计时不代表整图性能。

原始 Comfy checkpoint 不能直接当 MLX 的 packed 参数使用。此次接入处理了：

1. Comfy U8 中高 nibble 在前，转换成 MLX 低 nibble 在前的 U32 打包。
2. FP8 E4M3 block scales 从 cuBLAS 分块布局重排成 MLX 的 row-major group-16。
3. `weight_scale_2` 按 Comfy 的全局乘数使用。它与 MLX `global_scale` 参数的
   数值约定不同，不能把同一个数直接传过去，否则可能发生约 2688 倍的幅值错误。
4. GPU 使用 W4A16，**不量化激活**，不使用 Comfy 的 `input_scale` 做 W4A4。

这是无损布局转换，不是 BF16 重量化到另一个“4 bit”。三个真实层的独立 FP32
解码与 MLX 解码逐元素相等，第一层 C++ 适配输出也逐元素相等。完整 checkpoint
共重排 180 层；实际 QKV 的 packed matmul 对 FP32 解码矩阵参考 relative L2 为
0.002362（包含 BF16 运算舍入）。另外，小型合成测试验证了 nibble/scale 重排，
并验证拒绝负数/NaN global scale 和损坏的 scale 尺寸。

原 NVFP4 文件 SHA-256 经核验：
`a553c889dbcb910de4c98293237573219a37007c1074a3f04576646a088bd5c8`。
未测试其 NVIDIA W4A4 路径，也不声称 M4 Max 具备 NVIDIA FP4 Tensor Core。

## Qwen3-4B：纯编码测试

相同原始权重 SHA-256：
`6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a`。
相同确定性 token IDs，32 / 128 tokens，无 padding；取第 35 个 block 后、最终
RMSNorm 前的隐藏状态。权重已加载，排除分词、磁盘加载、prompt cache 和出图。
各配置 1 次预热 + 3 次测量；输出显式求值后计时结束。

| TurboCider / MLX | 文件 GB | 32 tokens ms | 128 tokens ms | 128-token MLX 峰值 GiB | 128-token 进程峰值 RSS GiB |
|---|---:|---:|---:|---:|---:|
| BF16 | 8.04 | 113.61 | 150.19 | 7.909 | 7.520 |
| affine Q4 / group 32 | 3.05 | 35.05 | 97.98 | 2.901 | 2.868 |
| affine Q8 / group 32 | 4.87 | 35.66 | 99.75 | 4.593 | 4.560 |

| 编码误差，相对 native BF16 | 32-token cosine | 128-token cosine | 128-token relative L2 |
|---|---:|---:|---:|
| Q4 | 0.997959 | 0.996316 | 0.097992 |
| Q8 | 0.999980 | 0.999972 | 0.007991 |

因此，本机 Q8 在这个测试中保真更好，Q4 进一步省内存；两者暖编码速度接近。
这些确定性 IDs 用于数值/速度测试，不是自然语言质量测试。真实提示词质量见前面的
出图样例，不能从随机 IDs 的 cosine 推导用户偏好。

## llama.cpp：确实做了独立对照

只读使用本地 `references/llama.cpp`，源码版本
`64e9bceb2c3a856efed96feda784a50947049feb`，构建产物留在 TurboCider 的
`.deps/llama-bench-build`，没有修改参考仓库或把 llama.cpp 打入 App。

从同一 Comfy BF16 Qwen 权重转换 GGUF；Q4_1 / Q8_0 保留 BF16 embedding。
GPU offload 99 layers，Flash Attention 开启，context 512，batch/ubatch 512，
8 CPU threads。每次清空 KV，使用 public eval callback 取 `l_out-34`，在该 GPU
graph segment 截止，而非取普通最终归一化 embedding；包含同步及 CPU readback。
它是验证用编码适配器，不是生产集成或官方 llama-bench 的 token-generation 测试。

| llama.cpp | 32 tokens ms | 128 tokens ms | 128-token 峰值 RSS GiB |
|---|---:|---:|---:|
| BF16 | 37.76 | 78.37 | 7.761 |
| GGUF Q4_1 | 32.97 | 77.48 | 3.108 |
| GGUF Q8_0 | 35.81 | 79.40 | 4.589 |

BF16 对 native BF16 的隐藏状态 cosine 是 0.9999989 / 0.9999984，128-token
relative L2 为 0.001790。数值对齐支持此测试取到了相应层，但两端精度/缓存和
graph 实现并非逐位相同。GGUF Q4_1/Q8_0 与 MLX affine 的量化公式/scale 精度也不
完全相同，Q4/Q8 横向比较是两套完整配置的比较，不是仅替换执行器的同权重实验。

结论：本机本测试 **llama.cpp BF16 纯编码更快**（32 tokens 约 3.01×，128 tokens
约 1.92×）；量化后短提示词速度接近，128 tokens llama.cpp 仍较快。进程 RSS
没有显示一个后端在全部精度下一律更省：native Q4 较低，Q8 接近。
没有把 llama.cpp 接入生产出图链路，因此**没有“仅替换成 llama.cpp 后最终图像
质量相同”的实测结论**。不建议只凭这张表更换整个产品后端。

## sd.cpp 与 llama.cpp 不是一回事

参考树的 `src/conditioning/conditioner.hpp` 使用自己的 `LLM::LLMRunner`；
实现位于 `src/model/te/llm.hpp`，基于 ggml，并非调用 llama.cpp 的
`llama_get_embeddings`。Z-Image 选择 `{35}` 层；FLUX Klein 选择 `{9,18,27}`
（这里按 1 起算）。因此不能把 sd.cpp 文本时间贴成“llama.cpp 性能”。

额外跑了本地固定 Unsloth sd.cpp binary：
`master-813-bfbef5b-u13b9d92`，source commit
`13b9d92b5e9a1563536c9c980e700470f9ab6702`。不是声称测了最新 sd.cpp。
相同 BF16 DiT、Qwen、VAE，512² / 9 steps / seed 42，Metal，1 冷 + 2 暖请求。

| 指标 | TurboCider | 固定 sd.cpp binary |
|---|---:|---:|
| 冷请求（不含 server/model setup）s | 10.819 | 44.255 |
| 暖请求外部中位数 s | 8.474 | 29.177 |
| 暖 denoise s | 8.225 | 19.360 |
| 暖 VAE decode s | 0.230 | 9.075 |

此比较测的是完整配置与执行策略。原生暖请求命中 prompt cache，而 sd.cpp 日志
中两个暖请求 conditioning 为 0.95 / 0.15 s，因此不能用这两列证明 native 的
Qwen 数学运算更快。sd.cpp 的外部计时额外包含 localhost HTTP/base64 路径。
sd.cpp 日志报告参数分配约 20317.52 MiB（含 text encoder 8414.50 MiB），
不是运行峰值；该轮没有独立采集它的全进程 peak RSS，不能与 MLX peak 作等口径排名。

两个引擎的狐狸图像构图不同，但都符合此提示词。同一个 seed 不保证两个引擎的
初始噪声完全相同；未做 noise-/conditioning-identical 的整管线对齐，不作审美排名。

## 生视频与其他 text encoder 的适用边界

| 当前组件 | 不能直接用通用 embedding 替换的原因 |
|---|---|
| Z-Image Qwen3-4B | 第 35 block、逐 token、未做最终 RMSNorm；模板与 padding 必须一致 |
| FLUX Klein Qwen3 | 多层隐藏状态拼接；4B 与 9B hidden width 不同 |
| Wan UMT5 | 双向 encoder、相对位置 bias、mask/词表和 dtype 约定不同 |
| LTX Gemma | 48 层及多层特征拼接投影，输入投影维度 188160；还含视频 connector |
| H3 文本组件 | 当前代码为 50 层、hidden 5120 的特定契约，不能套用 Qwen3-4B 表格 |

本次没有重复评测这些视频组件的量化速度和最终视频质量。Unsloth 是这里参考
binary / GGUF 工具链的来源之一，不是另一个已测试的 Apple GPU text-encoder
执行后端。选择后端必须先对齐全部隐藏状态、位置、mask、精度和投影契约。

## 建议与验证记录

64 GiB 这台机器优先 BF16 DiT；Qwen Q8 可作为保真/内存折中。内存紧张时可尝试
INT8 DiT + Q4 text，先从 512² 开始；NVFP4 压得更小，但样例偏离更大且没有更快。
尚未实测 16/24/32 GiB 设备、1024² 量化对照或长提示词，不给最低 RAM 保证。

已运行：原生/Swift 构建、`make test`、`make test-library`、显式 GPU NVFP4
合成正反例、三真实层布局校验、六组图像配置、六组 native / 六组 llama.cpp
编码配置、固定 sd.cpp 整图对照。默认回归会跳过未 opt-in 的 NVFP4 GPU 测试；
该测试另外用 `TURBOCIDER_TEST_GPU=1` 实际通过。部分现有 fixture/numpy 测试
按仓库原规则跳过，不能把跳过视为通过。没有完成 App 人工视觉验收。

本机原始证据（未加入 Git 的大文件目录）：`outputs/quantization-20260910/`。

- `summary.json`：五组 native 图像及六组 native 编码原始样本。
- `llama-summary.json`：六组 llama.cpp 编码、RSS、隐藏状态误差。
- `quality.json` / `comparison.png`：六种组合的图像差异和并排图。
- `nvfp4-final/report.json`：最终构建的 NVFP4 独立出图实验，正确标注 W4A16。
  较早的 `nvfp4-text-bf16/` 记录保留，最终复测图片相同，表格使用最终计时。
- `nvfp4-native-verified.json`：独立解码、MLX、C++ 的逐元素检查。
- `sd-cpp-reference/report.json` / `stable-diffusion-cpp.log`：固定参考引擎实验。

复现工具位于 `tools/validation/z_image_quantization.py`、
`llama_qwen_benchmark.py`、`benchmark_nvfp4.py`、`nvfp4_layout.py`；入口均提供
`--help`。C++ probe 位于 `tools/native/qwen3_quant_probe.cpp`、`nvfp4_probe.cpp`
和 `tools/validation/llama_qwen_conditioning.cpp`，需分别链接本地 MLX/native
或 llama/ggml 库。模型下载、转换、benchmark 不属于 App 的运行依赖。

例如在已有 native 构建和本次参考构建的工作区：

```sh
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
LLAMA_SOURCE="$PWD/../references/llama.cpp" \
LLAMA_BUILD="$PWD/.deps/llama-bench-build" \
  bash tools/validation/build_quantization_probes.sh
```

仅测 native 时省略 `LLAMA_SOURCE` / `LLAMA_BUILD`。新增 NVFP4、Qwen Q4/Q8、
llama 对照 GGUF 权重均留在 `models/`，原 BF16/INT8 文件未覆盖；未自动清理这些
约 27 GiB 的本地实验权重。

格式核实来源：ModelScope 仓库 API；本机 MLX 0.32.2 API/头文件；ComfyUI
`comfy/ops.py`、`comfy/quant_ops.py`；Comfy-Org/comfy-kitchen
`backends/eager/quantization.py`、`float_utils.py` 的公开参考实现。
