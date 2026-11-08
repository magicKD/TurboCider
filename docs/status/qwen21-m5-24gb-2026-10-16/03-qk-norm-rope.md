# Qwen-Image-2.1 QK 归一化与 RoPE 融合性能报告

发布日期：2026-10-16。历史单项实验：2026-10-02；当前整合与验证：2026-10-15 至 2026-10-16。当前实现提交 `242befe983d5b17439daecb5ef2ab44de8fa5a8f`，整合对照基线 `2a4c2b5b03b8dba4c53f0f8e8f37d0dfa1e6a441`。

在 Apple M5 Pro、24 GiB、无 LoRA 的 Qwen-Image-2.1 流式 GPU + ANE 路径上，将 Q/K 的 RMSNorm、RoPE 和注意力布局转换融合到一个 Metal kernel，历史狐狸样本的整次耗时中位数由 **56.617 秒降至 52.797 秒**，下降 **6.75%**；去噪由 **45.671 秒降至 42.341 秒**，下降 **7.29%**。另一独立题材单次复核的整次耗时下降 **3.81%**。两组输出均有像素差异，不能将该优化称为无损或逐位等价。

PR #12 将它纳入精确硬件与请求条件下的默认优化，同时保留显式关闭用于消融。2026-10-15 至 16 的 **52.098%** 整次降时来自 QK 融合、GPU FFN suffix W4、attention W4 与 DBCache 的组合，不能归因于 QK 单项。

## 融合实现与有效原因

[QK Metal kernel](../../../native/models/qwen21/metal/qk_norm_rope.hpp) 直接读取投影输出的 token-major Q/K，每个 SIMD group 处理一个 128 通道 head，以 FP32 累加平方和并计算 RMSNorm，再执行 RoPE，直接写出注意力需要的 head-major 布局 `[1, 32, tokens, 128]`。[Transformer 调用点](../../../native/models/qwen21/transformer.cpp) 以该结果替代原先分别归一化、旋转和转换布局的计算路径；Q/K/V 投影和后续注意力仍按原有职责执行。

融合减少独立算子调度以及中间数组读写，且避免在这些步骤之间重复处理布局。内核在 RMSNorm 后显式转换回激活 dtype，再读回 FP32 做旋转，保留了归一化与 RoPE 之间的 BF16 舍入。然而 SIMD 归约次序与融合算术路径不同，保留这次舍入并不保证结果逐位相同，因此计划用 `qwen21_metal_qk_norm_rope` 标记算法近似。

历史单项实验中，W8A8 Core ML FFN 前缀、BF16 GPU 后缀、实际采样步数和 ANE manifest 均保持一致。MLX 峰值及读量也基本相近，结果与减少 GPU 端归一化和旋转开销的机制一致；没有算子级 profiler 将全部整次收益精确分摊到上述成本。

## 历史单项实验条件

主套件 `hybrid-kernel-memory` 使用原始 BF16 checkpoint、512×512、40 步、seed 42，提示词为 `A red fox standing in fresh snow, soft morning light.`。所有请求均无 LoRA、无参考图，使用 `execution=gpu_ane`、`residency=component_staged`、`qwen21_w8a8=true`、`allow_approximation=true`。文本编码器及 DiT 逐层释放，VAE 分阶段加载；这里的流式指该 layer-streamed 路径，不是公共动态 streaming 配置。

ANE 分区覆盖 32 层、1024 rows、FFN 前 6144/12288 通道，使用 W8A8 与 FP16 输入输出。两组均保留已有紧凑 BF16 GPU suffix 缓存及去噪阶段 6 GiB wired 限额；未启用本次整合的 W4 缓存或 DBCache。

完整运行顺序为 baseline → qknorm → cache1g → cache1g → qknorm → baseline。下表只纳入其中 baseline 和 qknorm，cache1g 是另一项实验。每个 cell 为独立进程中的首次请求，`prompt_cache_hit=false`；计时为 `request_wall_including_export`，包含 PNG 导出。文件缓存未清空，不等于冷盘测试，也没有同进程第二次请求样本。温度和后台负载未完全控制。

| Suite cell | 配置 | 整次耗时 s | 去噪耗时 s |
| --- | --- | ---: | ---: |
| `00-baseline` | QK 融合关闭 | 57.503465 | 45.945504 |
| `01-qknorm` | 仅 QK 融合开启 | 52.711476 | 42.297391 |
| `04-qknorm` | 仅 QK 融合开启 | 52.882621 | 42.385110 |
| `05-baseline` | QK 融合关闭 | 55.730775 | 45.395579 |
| baseline 中位数 | 2 次 | **56.617120** | **45.670542** |
| qknorm 中位数 | 2 次 | **52.797049** | **42.341251** |

按 `1 − 候选中位数 / 基线中位数` 计算，整次耗时下降 6.7472%，去噪下降 7.2898%；整次速度比为 1.0724×。两个样本不足以估计稳定的统计区间。

| 资源指标中位数 | baseline | QK 融合 |
| --- | ---: | ---: |
| MLX allocator 峰值 bytes | 6,083,208,421 | 6,083,208,470 |
| 进程 lifetime max physical footprint bytes | 7,006,627,580 | 6,894,601,896 |
| 进程 lifetime max RSS bytes | 1,538,891,776 | 1,569,349,632 |
| 进程读量 GB | 227.883 | 226.348 |
| Core ML runtime 调用数 | 每次 1248 | 每次 1248 |

GB 为十进制。MLX allocator、RSS、physical footprint 的统计范围不同，均不能替代整机总内存。进程读量是计数器增量，不是 SSD 介质实测吞吐；本组数据不支持把 QK 收益解释为明显减少权重读取或内存占用。

## 独立题材与画质限制

`hybrid-heldout` 使用 seed 17，提示词为 `A red apple on the left, a blue ceramic cup in the center, and a yellow lemon on the right, arranged on a wooden table, soft studio lighting.`，仍为 M5 Pro / 24 GiB、无 LoRA、512×512、40 步和同类流式 GPU + ANE 配置。

此套件使用加入 staged DBCache 准入后的另一份历史库，只与其自身 baseline 比较，不和主套件合并。顺序为 baseline → qknorm → db025；下表只比较前两项，各为一个新进程请求，没有尾部 baseline 或重复样本。

| Suite cell | 整次耗时 s | 去噪耗时 s | Runtime 调用及失败 |
| --- | ---: | ---: | --- |
| `00-baseline` | 53.493062 | 43.405975 | 1248 / 0 |
| `01-qknorm` | 51.453216 | 41.297502 | 1248 / 0 |

该次整次耗时下降 3.8133%，去噪下降 4.8576%。这是独立题材和另一构建的探索性复核，不能据此承诺稳定加速区间。

| 同套件候选相对 baseline 的差异 | 狐狸 | 苹果杯子柠檬 |
| --- | ---: | ---: |
| RGB MAE，0–255 标度 | 0.55067 | 0.63349 |
| RGB RMSE，0–255 标度 | 1.54032 | 1.84550 |
| RGB Pearson correlation | 0.9995201 | 0.9996253 |
| 最大 RGB 绝对差 | 43 | 40 |
| 最大 alpha 绝对差 | 1 | 1 |

主套件两个 baseline PNG 彼此逐字节一致，两个 qknorm PNG 也彼此一致，但两组之间不同。历史目视记录中，狐狸主体、低头姿态和雪地构图保留，毛发与局部雪纹变化；heldout 的三种物体位置、颜色和木桌构图保留，高光及纹理小幅变化，基线已有的左侧苹果裁切仍存在。两题材未记录明显新增破碎伪影，但像素误差与有限目视样本都不足以证明普遍画质、文字准确性或指令遵循质量。

所有上述历史样本均记录 40 个实际去噪步骤、plan LoRA 数为 0、`layer-streamed` 运行选择、`mlx_cpp_metal+coreml` backend、`cpuAndNeuralEngine` 计算策略、1248 次 runtime 调用与 0 次失败。MLComputePlan 显示 32 层的 384 个非常量操作倾向 Neural Engine；运行报告仍是 `observed_ane_residency=unknown`，这些证据不等于直接测得物理 ANE 利用率。

## 当前默认准入与消融

[内存与优化策略](../../../native/models/qwen21/memory_policy.hpp) 中的 `qualified_staged_optimizations` 限定 **Apple M5 Pro 且物理内存精确为 24 GiB**、MLX 后端、无 LoRA、512×512 文生图、20–40 步、`component_staged`、显式 GPU + ANE、原有 W8A8 FFN 分区及允许近似。还要求无参考图、提示词增强、encoder ANE、tensor dump、设备预算或公共动态 streaming、native 路线、resident prefix、tiled prefill、ref-local attention、profiling、FFN reuse、fused QKV 等干扰项。启用 DBCache 时必须保留 front 8 blocks、warmup 8 steps；关闭 DBCache 也能命中默认优化。

`staged_optimization_options` 在上述条件成立且 `TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS` 未设置或为 `1` 时，默认启用 suffix W4、attention W4 与 QK 融合。非合格请求不会自动得到这三项默认优化。原有显式 QK 环境开关的兼容范围仍保留，不能据此推导成所有设备默认开启。

将 `TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS=0` 只会关闭新默认值，不会关闭请求中的 DBCache，也不会覆盖显式的位宽和 QK 环境变量。显式 `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=0` 始终受到尊重。`TURBOCIDER_QWEN21_METAL_QK_ROPE=1` 是另一条配对 RoPE 路径，与 QK norm/RoPE 融合互斥。

以下在同一当前版本上测量 QK 单项：使用无其他 Qwen 诊断环境变量的终端，将下面内容保存为 `qk-ablation.json`。保持同一模型、ANE manifest、提示词、种子与步数，DBCache 显式关闭。

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
  "execution": "gpu_ane",
  "residency": "component_staged",
  "dynamic_text": true,
  "loras": [],
  "qwen21_w8a8": true,
  "hybrid_mlp_mode": "base_fused",
  "allow_approximation": true,
  "qwen21_dit_cache": "off"
}
```

在仓库根目录设置自己的模型、匹配的已编译 manifest 和同一版 native 库位置，选择尚未存在的输出目录。下面每次调用都新建进程，只生成一次；顺序为关闭 → 开启 → 开启 → 关闭。

```sh
QWEN_MODEL_DIR='models/Qwen-Image-2.1-BF16'
QWEN_ANE_MANIFEST='compiled/manifest.json'
QWEN_LIBRARY='build/native/libturbocider.dylib'
QWEN_OUTPUT_ROOT='qk-ablation-new-run'
QWEN_CELL=0
for QWEN_QK in 0 1 1 0; do
  env TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS=0 \
      TURBOCIDER_QWEN21_STAGED_SUFFIX_BITS=0 \
      TURBOCIDER_QWEN21_STAGED_ATTN_BITS=0 \
      TURBOCIDER_QWEN21_METAL_QK_ROPE=0 \
      TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE="$QWEN_QK" \
      .venv/bin/python tools/native/benchmark_native.py \
      --library "$QWEN_LIBRARY" --model "$QWEN_MODEL_DIR" \
      --model-id qwen-image-2.1 --request qk-ablation.json \
      --ane-manifest "$QWEN_ANE_MANIFEST" --runs 1 --record-vm \
      --output "$QWEN_OUTPUT_ROOT/$QWEN_CELL-qk$QWEN_QK" || break
  QWEN_CELL=$((QWEN_CELL + 1))
done
```

[基准工具](../../../tools/native/benchmark_native.py) 生成的计划中，开启组应比关闭组多出 `qwen21_metal_qk_norm_rope`，两组都不应含 `qwen21_staged_suffix_w4`、`qwen21_staged_attention_w4` 或 `qwen21_decode_dit_cache`。还应检查实际步骤、流式选择、Core ML 调用与失败数。该命令是当前版本的独立消融方法，不是上文历史库的重新运行记录。

## 当前组合实测的归因边界

当前整合套件 `qwen21-m5-24gb-integration-20261015` 的 formal 狐狸样本，沿用 M5 Pro / 24 GiB、无 LoRA、512×512、40 步、seed 42、流式 GPU + ANE。上游基线关闭上述四项，fast 同时开启 QK 融合、[suffix W4](01-gpu-suffix-w4.md)、[attention W4](02-attention-w4.md) 与 [DBCache fast](04-dbcache.md)。每次均为新进程首次完整生成，包含 PNG 导出，文件缓存未清空。

| Formal run ID | 配置 | 整次耗时 s |
| --- | --- | ---: |
| `1792079228692-upstream-fox` | 上游基线 | 54.713867 |
| `1792079283508-fast-fox` | 四项组合 | 26.400967 |
| `1792079310343-fast-fox` | 四项组合 | 25.351826 |
| `1792079335789-upstream-fox` | 上游基线 | 53.828678 |
| `1792079389774-upstream-fox` | 上游基线 | 53.724513 |
| `1792079443624-fast-fox` | 四项组合 | 25.784862 |
| 上游中位数 | 3 次 | **53.828678** |
| 组合中位数 | 3 次 | **25.784862** |

组合整次耗时下降 **52.0983%**。该批次没有只切换 QK 的同构建重复对照，不能把这一降幅或历史单项百分比相加后作为 QK 的独立贡献。组合狐狸图相对上游基线的 RGB MAE 为 8.11171/255、RMSE 为 22.60006/255，变化明显大于历史 QK 单项；组合质量需按组合输出单独判断。

## 验证与可追溯证据

[CPU 策略测试](../../../tests/native/runtime_policy_test.cpp) 覆盖精确机型与内存边界、请求准入、显式 QK 关闭、默认关闭及位宽覆盖；[C ABI contract 测试](../../../tests/native/test_contract.py) 检查计划标签、默认开关与不兼容组合。当前整合留存的相关 C ABI 日志记录 48 项测试通过，日志 SHA-256 为 `b88fb9ab847b49e8d9dd24fcc9d963be7c12da366014c45806e69e6c190d1a3c`。

[QK 数值测试](../../../tests/native/qwen21_metal_qk_test.cpp) 检查 BF16 输出形状及多个 token 长度，相对 L2 阈值为 0.01；这是数值容差测试，不是逐位一致性保证，也不扩大本报告的默认硬件或请求范围。历史与当前正式性能数据来自实际完整生成，未以该数值 fixture 代替端到端计时。

原始 JSON、运行身份、日志和 PNG 在本地保留，未随本报告上传；以下 suite/run ID 和 SHA-256 可用于核对归档。报告中的数字直接来自完成的 run，未将 smoke、生命周期检查或其他候选合入中位数。

| 证据 | SHA-256 |
| --- | --- |
| `hybrid-kernel-memory` 历史库 | `781cc79ed418ce3cbecbc3d403c582d82f45a7e8037b6cbb3085b20282a5038a` |
| `hybrid-kernel-memory/identity.json` | `2b07595c84f36783da678c5a41ec9cea7d2491e301189d16c124a3eb3219e074` |
| `hybrid-kernel-memory/00-baseline/report.json` | `d8f24dc689ad513ba542a68843c42abbb2f6a3d4ca7b3edc81d08d28f91f29a2` |
| `hybrid-kernel-memory/01-qknorm/report.json` | `7e74f014fd875c30c268420658f69fdb20fcc9392083b0133a169c1e1e08e4aa` |
| `hybrid-kernel-memory/04-qknorm/report.json` | `8907ad56c8af9afc5e06b250bff17e03fbfa30300f197febcb3c82dc33fe7e70` |
| `hybrid-kernel-memory/05-baseline/report.json` | `5f7d76b5df0cc1296a12735177b06b6d7ff326bc3a77953e77bad29e56c0dd7e` |
| `hybrid-heldout` 历史库 | `be9f8a3aee67dd74805ddb744bb9bc38cf9d33e103daf8d6a29cc3a24cf73f25` |
| `hybrid-heldout/identity.json` | `e1603dc2d97186a46df0f63bd873361d85d91f440557b8ccdf59dbf099a0cac3` |
| `hybrid-heldout/00-baseline/report.json` | `d420bde65b4e2b4ba508f381104c18541ce305ccf9100152a5a10b50c31c0895` |
| `hybrid-heldout/01-qknorm/report.json` | `c15e1a777273c05566bfd825e5eb3d561a3206e5352f5b1ffd116ae47fcdda35` |
| 当前 formal 上游库 | `394f27d1e4b653867846b8ae4484e8f66710071a68c574a18d96805847ca0f53` |
| 当前 formal fast 库 | `36305cc4999e6dca2ec0ecaa3c81263d24a49592602b2b6514fadc0704c43d53` |
| 当前 `1792079335789-upstream-fox/report.json` | `68b13d3d46597b3aa0af26b834baa0fe3ed328d2653b31a26ad75e90d11067d3` |
| 当前 `1792079443624-fast-fox/report.json` | `65f42f183f99216a1f7d693799a624dbf7a55c596b7f0713fdcbd57a03f24005` |
| 当前整合 `summary.json` | `93a76ee9f601a4a47f7c6f20176e971f5427fff0142c9ffb95cbb4f9490007bf` |
