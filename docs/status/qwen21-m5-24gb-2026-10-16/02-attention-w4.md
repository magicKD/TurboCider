# Qwen Image 2.1 attention W4 请求内缓存

历史单项实验日期：2026-10-02。最新主干整合验证：2026-10-15 至 2026-10-16。发布日期：2026-10-16。

在 Apple M5 Pro、24 GiB、512×512、40 步、无 LoRA 的狐狸样本中，历史 attention W4 单项将整次请求中位数从 **55.517 秒降至 38.553 秒，减少 30.56%**。其作用包括把 decode attention 权重量化后留在请求内，并在后续步骤复用，因此收益包含重复权重读取的减少，不能全部归为低比特矩阵乘本身。

当前 PR 将它与 GPU FFN suffix W4、QK norm RoPE 融合及显式选择的 DBCache fast 组合，最新主干对照为 **53.828677625 → 25.784862292 秒，减少 52.098%**。这是四项组合的整图结果，不能作为 attention 单项收益。两轮实验始终保留 layer streaming、`component_staged` 和 ANE 配置；W4 会改变输出画面，需要接受近似计算。

## 缓存内容与有效机制

每个 DiT block 有 Q、K、V、输出四个无 bias 的 `4096×4096` attention 投影，共 32 层。首个采样步骤仍使用 BF16 权重，建立原有 prefix KV。首次 decode 才把这四个矩阵打包为 affine W4 或 W8，group size 为 64；求值完成后，缓存独立的 packed weight、scale 和 bias，后续 decode 复用。

这里缓存的是投影权重。已有 prefix K/V 的表示和 attention 运算保持原接口。输入、scale、bias、量化矩阵乘输出均为 BF16；“保持同精度接口”表示数据类型不变，不表示数值或图像无损。四个 projection 各有三个数组，每层共 12 个：

| 每个投影的存储 | W4 | W8 |
| --- | --- | --- |
| packed weight | `uint32[4096,512]` | `uint32[4096,1024]` |
| scale | `bfloat16[4096,64]` | `bfloat16[4096,64]` |
| bias | `bfloat16[4096,64]` | `bfloat16[4096,64]` |
| 每投影字节数 | 9,437,184 | 17,825,792 |
| 每层四投影字节数 | 37,748,736 | 71,303,168 |
| 32 层总字节数 | **1,207,959,552** | **2,281,701,376** |
| 32 层总 GiB | **1.125** | **2.125** |

总权重元素数 `N = 32 × 4 × 4096² = 2,147,483,648`，缓存字节公式为 `N × (bits/8 + 4/64)`。后项来自每 64 个权重的两个 BF16 scale/bias。四次历史量化整图实测的缓存层数和字节数与此一致。

原路线每步需要逐层读取这些 BF16 投影权重；缓存路线只在首次 decode 打包时读取一次，后续 attention 不再求值这些源矩阵。历史进程读计数中位数由 **227.878 GB** 降至 W8 的 **55.522 GB**、W4 的 **55.516 GB**。这是进程 I/O 计数，不等于物理 SSD 流量，也不能据此分解每种机制的耗时贡献。保留压缩权重会增加请求内工作集，历史单项 W4 的 MLX 峰值相应增加约 1.125 GiB。

实现位于 [staged_attention_quantization.hpp](../../../native/models/qwen21/staged_attention_quantization.hpp) 和 [Transformer](../../../native/models/qwen21/transformer.cpp)。打包后检查数组已求值、无 primitive/input/sibling 依赖并具有紧凑 backing；编译 block 显式接收 12 个数组，不把原 BF16 源矩阵捕获为跨步常量。首步 prefix KV 和已有 dynamic QKV、BlockVariants 路径继续使用原有实现。

## 历史单项狐狸对照

套件 `attention-quantization`，seed 42，提示为 `A red fox standing in fresh snow, soft morning light.`。执行顺序为 BF16 → W8 → W4 → W4 → W8 → BF16，每格是一个新进程的首请求，文件缓存未清空。时间为 `request_wall_including_export`，包含权重验证、加载、首次打包、去噪和导出，排除进程启动与 engine 构造。

本组只改变 attention 缓存：ANE W8A8 FFN 前缀 `[0,6144)`、GPU BF16 FFN suffix 保持不变，未叠加 suffix 量化、QK 融合、DBCache 或 LoRA。GB 使用十进制，GiB 使用二进制。

| Run ID | 配置 | 整次秒数 | 去噪秒数 | 进程读量 GB | MLX 峰值 GiB | 进程最大 footprint GiB | 首次打包秒数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00-baseline | BF16 | 58.873152 | 47.068639 | 229.411 | 5.665 | 6.533 | — |
| 01-attn8 | W8 | 40.032384 | 29.571976 | 55.526 | 7.790 | 8.625 | 0.540444 |
| 02-attn4 | W4 | 38.792800 | 28.572322 | 55.520 | 6.790 | 7.626 | 0.450486 |
| 03-attn4 | W4 | 38.313222 | 28.005085 | 55.513 | 6.790 | 7.626 | 0.452746 |
| 04-attn8 | W8 | 39.250661 | 28.971548 | 55.518 | 7.790 | 8.626 | 0.467837 |
| 05-baseline | BF16 | 52.160508 | 41.935680 | 226.344 | 5.665 | 6.524 | — |

| 中位数或质量指标 | BF16 | attention W8 | attention W4 |
| --- | ---: | ---: | ---: |
| 整次秒数 | 55.516830 | 39.641523 | 38.553011 |
| 去噪秒数 | 44.502160 | 29.271762 | 28.288703 |
| 整次耗时减少 | — | 28.60% | 30.56% |
| 进程读量 GB | 227.878 | 55.522 | 55.516 |
| MLX 峰值 GiB | 5.665 | 7.790 | 6.790 |
| 进程最大 footprint GiB | 6.528 | 8.625 | 7.626 |
| RGB RMSE，0–255 | 0 | 11.9384 | 19.5882 |
| 每次 Core ML runtime 调用／失败 | 1248 / 0 | 1248 / 0 | 1248 / 0 |

MLX 峰值只覆盖 MLX allocator，footprint 覆盖进程生命周期，两者不能相加。请求结束后的 MLX active 为 164,692–164,812 B。32 次 Core ML warmup 单独计数，不计入表中的 1248 次 runtime 调用；配置为 `cpuAndNeuralEngine`，物理 ANE 驻留率没有采样。

三种配置各自两次 PNG 逐字节一致。目视均保留红狐狸低头朝右的雪地场景；W8 改变后腿站位与腹部轮廓，W4 的后躯和尾部轮廓变化更明显。W4 比 W8 少 1 GiB 缓存，但整次中位数仅再少约 1.089 秒，不能按位宽推算等比例加速。首尾基线相差 6.713 秒，温度与后台负载未控制，每配置只有两个样本，因此上述比例是这一组的观测值。

## 历史独立文字样本

`heldout-text` 使用 seed 13，提示为 `A clean white poster with the exact words "BLUE SKY" in large dark-blue capital letters, centered, no other text, flat graphic design.`。其余请求参数与狐狸组一致。此共享套件还有其他候选，下面仅列首尾基线和 attention 两项；不能与狐狸组混合统计。

| Run ID | 配置 | 整次秒数 | 去噪秒数 | 进程读量 GB | MLX 峰值 GiB | 最大 footprint GiB | RGB RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00-baseline | BF16 | 49.780378 | 39.676446 | 226.340 | 5.674 | 6.533 | 0 |
| 01-attn8 | W8 | 37.743646 | 27.616234 | 55.517 | 7.799 | 8.633 | 50.8715 |
| 02-attn4 | W4 | 38.263606 | 28.231965 | 55.514 | 6.799 | 7.634 | 69.6516 |
| 06-baseline | BF16 | 54.567661 | 43.996954 | 226.341 | 5.674 | 6.565 | 0 |

基线整次中位数为 52.174020 秒，去噪中位数 41.836700 秒；W8、W4 各只有一次，不能估计重复运行方差。两者分别花 0.465731、0.443630 秒首次打包，仍有 32 层缓存、1248 次 runtime 调用和零失败。

三种输出均正确写出 `BLUE SKY`，没有多余文字。基线为斜体衬线字，候选改为直立无衬线，W4 字号更大且位置、行距不同。提示未指定字体，因此像素差异不能直接解释成语义质量下降。W4 在这个样本上比 W8 慢 0.520 秒；这个英文短语也不能代表中文、长文本或所有种子。

## 当前 PR 的组合结果

整合代码为 `242befe`，最新主干基线为 `2a4c2b5b03b8dba4c53f0f8e8f37d0dfa1e6a441`。`formal` 套件仍用狐狸 seed 42、512×512、40 步和相同 ANE manifest，顺序为 A–B–B–A–A–B。A 是主干的原 staged W8A8 路线；B 是 attention W4、suffix W4、QK norm RoPE 加显式 `qwen21_dit_cache=fast`。每次都是新进程首请求，无文件缓存清空。

| Run ID | 路线 | 整次秒数 | 去噪秒数 |
| --- | --- | ---: | ---: |
| 1792079228692-upstream-fox | A | 54.713867 | 44.067167 |
| 1792079283508-fast-fox | B | 26.400967 | 14.828719 |
| 1792079310343-fast-fox | B | 25.351826 | 15.043620 |
| 1792079335789-upstream-fox | A | 53.828678 | 43.608358 |
| 1792079389774-upstream-fox | A | 53.724513 | 43.501677 |
| 1792079443624-fast-fox | B | 25.784862 | 15.320972 |

| 中位数 | A 主干 | B 四项组合 |
| --- | ---: | ---: |
| 整次秒数 | 53.828677625 | 25.784862292 |
| 去噪秒数 | 43.608358209 | 15.043620208 |
| 进程读量 GB | 226.783 | 55.519 |
| MLX 峰值 GiB | 5.665 | 5.429 |
| 进程最大 footprint GiB | 6.555 | 4.777 |
| Core ML runtime 调用／失败 | 1248 / 0 | 648 / 0 |

B 的 attention 缓存仍恰为 32 层、1,207,959,552 B，三次首次打包为 0.512167、0.519316、0.489396 秒；DBCache 每次命中 25 步。B 的更低总峰值同时受到 suffix W4 和跳层路径影响，不能归因于 attention 缓存单项。狐狸相对 A 的 RGB RMSE 为 22.6001；组合的文字、三物体样本分别为 10.5378、8.1072，属于组合质量记录，不能当作 attention 独立误差。

相关实现分别见 [GPU suffix W4](01-gpu-suffix-w4.md)、[QK norm RoPE](03-qk-norm-rope.md)、[DBCache](04-dbcache.md)。

## 准入与可复现参数

[统一策略](../../../native/models/qwen21/memory_policy.hpp) 仅为 Apple M5 Pro、物理内存恰为 24 GiB、MLX denoiser、512×512、20–40 步、无输入图及无 LoRA 的 staged W8A8 请求启用默认组合。请求必须允许近似、采用已验证的 6144 通道 ANE 前缀，并且没有冲突的诊断或其他 residency 模式。其他设备不会继承这组默认值。这里的 layer streaming 是该 `component_staged` 路线内部的逐层装载；不需要打开通用 `streaming_offload` 开关。

当前合格请求默认 attention W4、suffix W4、QK 融合开启；DBCache 仍由请求显式选择。历史实验版本的 attention 默认值为 0，复测单项时应显式关闭当前组合默认值，避免把组合收益混入单项。

将下列请求保存为 `request-attention.json`，把 `ane_manifest` 替换为本地已验证 manifest。历史 manifest SHA-256 为 `4fba06ff0da1c432f317c7713946f888917b24b44751a3f48868827ba1f52fc0`。模型目录由 `MODEL_ROOT` 提供，需包含原 BF16 checkpoint 与配套组件。

```json
{
  "model": "qwen-image-2.1",
  "operation": "image.generate",
  "prompt": "A red fox standing in fresh snow, soft morning light.",
  "width": 512,
  "height": 512,
  "steps": 40,
  "seed": 42,
  "frames": 1,
  "audio": false,
  "dynamic_text": true,
  "execution": "gpu_ane",
  "residency": "component_staged",
  "qwen21_denoiser_backend": "mlx",
  "qwen21_w8a8": true,
  "allow_approximation": true,
  "qwen21_dit_cache": "off",
  "ane_manifest": "assets/ane/manifest.json",
  "loras": [],
  "output": "results/attention-w4.png"
}
```

在没有其他 Qwen 诊断环境变量的 shell 中，使用仓库依赖环境和已编译库：

```sh
TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS=0 \
TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS=0 \
TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=0 \
TURBOCIDER_QWEN21_STAGED_ATTN_BITS=4 \
python3 tools/native/benchmark_native.py \
  --library build/native/libturbocider.dylib \
  --model "$MODEL_ROOT" --model-id qwen-image-2.1 \
  --request request-attention.json --output results/attention-w4 \
  --runs 1 --record-vm
```

只改变 `STAGED_ATTN_BITS` 为 `0`、`8`、`4` 即可进行当前代码上的单项对照；每格另起进程，输出目录分别设置。历史六次数字属于 10 月 2 日版本，当前复测应单独成组。要复测 PR 组合，应移除上述四个环境覆盖，让合格请求采用默认值，并在 JSON 中显式设置 `qwen21_dit_cache` 为 `fast`。结果须核对 `qwen21_staged_attention_quantization` 中的 `bits`、`cached_layers`、`cache_bytes`、`pack_seconds`，以及 Core ML 调用数和失败数。

## 正确性与取消重试

[真实 MLX fixture](../../../tests/native/qwen21_staged_attention_test.cpp) 覆盖 W4/W8 的 12 数组形状、dtype、字节数、四个源缓冲区实际释放、重复 eager/compiled 投影与独立 MLX 参考一致、BF16 接口和坏参数拒绝，并验证与 QK norm RoPE 的组合。[测试入口](../../../tests/native/test_qwen21_staged_attention.py) 已在整合验证中通过。

[Transformer](../../../native/models/qwen21/transformer.cpp) 只允许逐层释放、标准几何、decode split FFN 且没有 dynamic weights、external/fused QKV、LoRA 或步间 FFN reuse 的 attention 缓存。位宽或融合设置改变时，setter 清空请求缓存和编译变体；异常也清空部分构建状态。[Pipeline](../../../native/models/qwen21/pipeline.cpp) 每步从同一源租约重建权重映射，DBCache 跳过的层仍未求值，下一步清空映射；去噪结束后先销毁请求 Transformer，再进入 VAE。

历史 `lifecycle-followthrough` 中，`cancel-attn4` 在 denoise completed=2 返回取消且没有 PNG；同一 engine 的 `retry-attn4` 成功，PNG 与两次正式 W4 图逐字节一致。取消／重试后 MLX active 分别为 **164,896 / 165,334 B**，去噪期间 wired limit 为 6 GiB，返回后均恢复调用者的 256 MiB，套件结束恢复原值 0。

整合 `lifecycle-combined` 又完成同一 engine 的 10 项顺序验证：文字第 4 层取消重试、去噪第 2 步取消重试、第 12 步取消重试、关闭默认组合、重新启用组合，以及 max-consecutive=8 诊断的第 12 步取消重试。全部取消均无输出，重试 PNG 与各自正式参考一致；各调用后 active 为 0–169,252 B，wired limit 均恢复到调用者的 256 MiB。取消请求本身不返回完整运行回执，因此缓存层数和调用预算由随后的成功重试核验。这些生命周期请求不计入性能中位数。

## 原始证据标识

下表保留 suite、run ID 和报告 SHA-256 前 16 位。核心数值已嵌入正文；原始 report、identity 和 PNG 未上传至此 PR，表中标识用于对应保留的原始记录，不作为仓库文件链接。历史数值与当前组合数值来自不同二进制，应分别比较。

| Suite | Run ID | report SHA-256 前 16 位 |
| --- | --- | --- |
| attention-quantization | 00-baseline | `c346502d9e371160` |
| attention-quantization | 01-attn8 | `87813a15ecaaac49` |
| attention-quantization | 02-attn4 | `7df551526fa5fd4c` |
| attention-quantization | 03-attn4 | `7399b2e1e8b8aab2` |
| attention-quantization | 04-attn8 | `16ad110bbcaf7e26` |
| attention-quantization | 05-baseline | `dc218569242e7ac6` |
| heldout-text | 00-baseline | `2a498ffd9fcbfb2c` |
| heldout-text | 01-attn8 | `2b5bdd6b9dba99a3` |
| heldout-text | 02-attn4 | `e58d564ecb25c67f` |
| heldout-text | 06-baseline | `ee417b0b5270c18b` |
| formal | 1792079228692-upstream-fox | `438a9d5d86424de7` |
| formal | 1792079283508-fast-fox | `50795defd252ef98` |
| formal | 1792079310343-fast-fox | `6b764c13ed5fb878` |
| formal | 1792079335789-upstream-fox | `68b13d3d46597b3a` |
| formal | 1792079389774-upstream-fox | `ad6edca4b8b22e5a` |
| formal | 1792079443624-fast-fox | `65f42f183f99216a` |
| lifecycle-followthrough | cancel-attn4 与 retry-attn4 所在报告 | `5bf396583636a59e` |
| lifecycle-combined | 10 项顺序验证报告 | `97dd6119be61fa6a` |

历史库 SHA-256：`37af44bb3190c19b462df25d8f0b0ed85c5025461dd4015fcecdf8a49c5688eb`。历史 `attention-quantization/identity.json`：`ba1ab49155577cb8f87f7632f8f419a2d9b048c9c0dd141b0314b3440b366cc7`；`heldout-text/identity.json`：`4c86889d1461cb68be4aac6130f16703fdf6f5aa1d73b876701894b8fb68d732`。

整合基线库 SHA-256：`394f27d1e4b653867846b8ae4484e8f66710071a68c574a18d96805847ca0f53`；候选库 SHA-256：`36305cc4999e6dca2ec0ecaa3c81263d24a49592602b2b6514fadc0704c43d53`。当前结果字段和精度标签由 [结果桥接](../../../native/platform/apple/results.mm) 与统一策略共同输出。
