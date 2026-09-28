# 冻结 base Core ML 与 runtime LoRA：当前路线（2026-09-28）

本页是 Qwen-Image-2.1 和 Z-Image Turbo 的运行入口与实验结论索引；
各次测试的提示词、精度、调用数及限制分别见
[Qwen 记录](qwen21-runtime-lora-fused-2026-09-28.md)和
[Z-Image 记录](z-image-runtime-lora-fused-2026-09-28.md)。以下速度均为
M4 Max、**512×512、已加载的 resident 热请求**，包含 VAE/PNG；
不能当作冷启动、别的 LoRA、设备或画质的保证。

## 保留什么、如何选择

| 请求与路线 | 入口 | 当前结论 |
| --- | --- | --- |
| 无 LoRA，原生最快的已测 base 路线 | 保留现有 `execution=auto` 及模型专用显式 manifest；不要为了共图强制使用 `lora_fused` | 不改变此前 base 的加速选择；Z-Image 另有显式 1024-image-row/5120-channel W8A8 路线，已测约 1.31× GPU。 |
| runtime LoRA、严格 GPU 对照 | `execution=gpu`、`lora_strategy=inference_time` | 两模型的正确性/观感基准；Qwen 仍以此为默认。 |
| runtime LoRA、完整可复用 base Core ML FFN | `execution=gpu_ane`、`hybrid_mlp_mode=lora_fused`、匹配的 **base-only** `ane_manifest`、`allow_approximation=true` | GPU 运行 LoRA，Core ML 只保存冻结 base 权重。同一个图可供 base 和不同 LoRA 复用；Z-Image 4096-channel 完整路线实测约 1.20×，可选的直接 FP16 修正路线约 1.217× GPU；Qwen 6144-channel 的整图优势尚不稳固。 |
| Qwen 仅导出 base gate/up | `hybrid_mlp_mode=lora_gate_up`、匹配的专用 manifest | 完整计算 LoRA，但 GPU 负责 SiLU/down；比原有融合 base 图慢，只保留显式研究选项。 |
| 仅 GPU 后缀计算 LoRA | `hybrid_mlp_mode=lora_suffix`、匹配的 base manifest | **故意不完整**：漏掉 Core ML 前缀的 LoRA gate/up/down 效果；不得作为正确的 LoRA 加速成绩或自动默认。 |
| Z-Image 将 LoRA 合入模型 | `hybrid_mlp_mode=lora_merged`、匹配的 adapter-bound manifest | 只用于独立对照；换 LoRA 要换图，不满足 runtime 共图要求。 |

`lora_fused` 的名称沿用已有 CLI 契约：它融合的是 **base FFN 算子**，
不是 LoRA 权重。GPU 先计算该次请求的 gate/up LoRA 差值，作为
Core ML 的第二个 FP16 **激活输入**，在 SiLU 前相加；图输出 base down
和 pre-down hidden，再由 GPU 计算该 hidden 上的 LoRA down。没有 LoRA
时差值是零，GPU 不计算 LoRA down。注意力及非 FFN LoRA 仍由 GPU 执行。
若只让 Core ML 输出已完成的 base FFN，之后无法准确补回 SiLU 前的
LoRA；旧 `lora_suffix` 因而不能替代此路线。manifest 校验 base checkpoint
的 SHA 与几何，并要求它不绑定适配器身份；显式选错图应报错而非
悄悄使用不相容的近似。

对 Z-Image，选择固定 1056-row、4096-channel、W8/FP16 激活的
`--runtime-lora-fused` 图，512px、8 步，resident；额外的
`TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16=1` **仅为可选实验**，令 GPU
LoRA gate/up 差值从 FP32 直接舍入至 FP16，不先经过 BF16。它不跳过
投影，但有小幅数值差异（单狐狸样本对原 hybrid 的 RGB RMSE 0.0110），
同二进制配对约快 0.6%，两次最终版热请求与 GPU 的中位比约
**1.217×**；未满足自动推广到其他 LoRA 的质量与跨提示词证据。

对 Qwen，当前已验证的是 Viggle v0.2.1 r256、6 步、512px：
正确的共图 GPU/Core ML 与 GPU 四次热态中位分别为 **7.715 s** 和
**7.811 s**，名义 1.012×，小于运行波动，不作为默认加速结论；
三张诊断性缩到 512px 的参考图编辑分别约 **12.50 s** 与 **12.34 s**
（两次热请求中位），无可靠整图优势。原尺寸参考编辑尚不能按这
组结果推广。GPU 上可选 FP16 低秩计算和 Metal Q/K norm-RoPE 优化
仍是显式选项；不能把基础模型的提升或仅 GPU 后缀的更快数据
记为完整 LoRA 的收益。

## 保留的代码与已撤回的实验

- `native/backends/coreml.mm` 负责冻结图验证、共享输出和动态激活
  接口；`native/backends/mlx.cpp` 负责带适配器的任意矩阵行/列切片。
- `native/models/qwen21/hybrid.cpp` 和 `native/models/z_image/z_image.cpp`
  仅在匹配的显式图上组装 LoRA gate/up、GPU 补集和 down-LoRA；
  `native/models/*_module.cpp` 在 plan 阶段把未经验证的组合挡住。
- `apps/cli/main.mm` 的 `--hybrid-mode MODE --ane-manifest MANIFEST`
  是一次请求的选择，不改写 JSON，也不按文件名猜适配器或图。
- 真实模型的 base → LoRA A → 临时派生 B → base 共图切换由
  `tools/validation/runtime_lora_shared_graph_switch.py` 检查；
  B **只是合成文件**，不能代表另一个训练 LoRA 的画质。
- 不保留 Qwen 的延后 GPU 后缀、双 GPU stream、直接 FP16 LoRA
  差值、只将 3/5/7 层回退全 GPU 等慢速实验开关。延后 GPU 后缀
  的同图交错测试从 **8.018 s** 退化到 **9.880 s** 热态中位；
  负结果在 Qwen 记录中，隔离的对照工具
  `tools/validation/hybrid_runtime_schedule_screen.py` 仍可复用。

切换 manifest、适配器或模式后，先运行 `plan` 和项目测试，再核对
实际绑定投影、Core ML 每请求调用数、PNG 与肉眼任务完成情况；
对不同 adapter 的训练步数、质量和速度重新验证。Core ML 的
CPU/Neural Engine **调度策略不证明**每个算子物理运行在 ANE 上。

## 整理后回归

最终源码重编译后，`make test-qwen21`、CLI ANE 覆盖测试、`make test`
与 `git diff --check` 均通过；`make test` 中依赖未提供的 Wan 夹具
或测试专用构建的用例按既有门禁跳过。共图切换脚本再次跑通
Z-Image 与 Qwen：Core ML 累计
调用数分别是 256/512/768/1024 与 160/320/480/640；适配器请求
分别绑定 238 与 227 个投影，首尾 base PNG 均逐字节相同。
这证明本次把 Core ML warmup 的零修正张量移出逐层循环后没有改变
这两条已测路径的输出；**没有重新进行速度测试**，不能把整理
本身计作加速。测试临时适配器与图片由验证脚本退出时清理，既存
编译产物与研究结果未在本次整理中批量删除。
