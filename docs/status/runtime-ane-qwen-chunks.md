# Qwen 长序列：减少 prediction 次数与重新平衡 rows

2026-09-29。接续[当前 1024² 三路对照](runtime-ane-qwen-1024-v2.md)。
组件筛选与 c1792 整模型交错对照均已完成：1024² base 热请求相对 c320
为 **1.078×**。保留为该工作负载的显式 optional，不作为通用默认。
不修改产品 native 后端、不合并 LoRA、不改变 SiLU 或默认路由。

## 为什么继续测 chunk

c320 v2 在 1024² base 的热请求中每次有 4,864–4,896 次 prediction。
设计稿要求区分 host chunk 与图内 K/N tile，不能直接套用 VPIPE 的默认
chunk。本轮先保持 GPU/ANE 总行数不变，测更少的大 chunk；再单独调整
分区，检查 ANE prediction 改善能否变成并行 FFN 墙钟收益。

## 组件条件与验证

- M4 Max 64 GB；Qwen 真实 BF16 checkpoint 的 block0 FFN 权重。
- **合成** 4,096-row BF16 输入（探针固定 seed41、normal×0.5），不是
  真实 1024² activation 捕获。现有 teapot 捕获只有 1,024 rows，未复制或
  reshape 冒充长序列。此阶段不生成图片，也没有训练 LoRA 质量资格。
- SwiGLU v2/LoRA-input 图，H4096/F12288，K1024/N512，exp SiLU；base
  传零修正。每层权重通过 runtime slots 传入，图内无模型/adapter 权重。
- 每个 trial 独立进程，排除两次 warmup，十次 GPU/并行交错测量；
  每配置两个正反向 trial，共 **20 个实测样本取池化中位数**。
- `gpu_seconds` 是探针独立编译的完整 FFN，不是整模型 GPU 基线；
  并行窗口含输入转换、输出处理、收集与拼接。含 staging 列中转换是
  暴露成本，此探针没有可隐藏 staging 的 attention。
- 每 trial 先检查竞争推理；所有配置同样开启 100 ms 独立进程采样。
  导出/构建/测试与计时串行。首次图加载可能触发设备缓存准备，加载时间
  单独记录，不进入下表热组件时间；这不是冷启动性能比较。

独立探针 SHA：
`5b04fde9590a214b18dc1fdcbdce0727e100e25d01c9d1ed85a50a60dcb8999f`。
产品库仍为 `e55b1a06cfd5f1c239b773907baa5b1fc07bdf0a614da7cdd84754a4ed5bdb0c`。

探针新增逐样本输入/输出/worker 时间、prediction/retry 数、headroom、
交错顺序与几何，保留原中位数字段；worker 与 GPU 重叠，不能把各窗口
全部相加。`make build-runtime-ane-probe test-runtime-ane` 通过 4 host +
9 Core ML 图/集成检查，包括小型 v2 不同 chunk/相同总分区与 CPU reference。
66 项加速契约也通过。没有重跑完整 `make test`，没有重建产品运行库。

## A：固定总分区，prediction 更快不等于并行更快

顺序 c320 → c640 → c640 → c320；ANE 固定 1,280 rows、GPU 2,816 rows。
时间为毫秒：

| 配置 | 完整 GPU FFN | prediction | worker 总窗口 | 并行 FFN | 并行含 staging |
| --- | ---: | ---: | ---: | ---: | ---: |
| c320 × 4 | 82.806 | 53.412 | 56.267 | 57.741 | 60.291 |
| c640 × 2 | 82.793 | 41.038 | 43.723 | 57.707 | 60.409 |

prediction 时间下降约 23.2%，但完整并行窗口几乎不变，不能宣布整模型
加速。显式 GPU head 完成后的 join 接近零，提示瓶颈已在 GPU head/收集
一侧；这不是测得的 GPU kernel 硬件时间，也不是物理 ANE placement 证明。
两配置 relative L2 均为 0.001833，cosine 0.999998，无 overflow retry。

## B：将 ANE 分区增至 1,536 rows

顺序 c320 → c768 → c1536 → c1536 → c768 → c320。候选的 GPU 均为
2,560 rows；c768×2 与 c1536×1 才是同总分区的直接比较。相对 c320 同时
改变了 chunk 和分区，不把全部改善归因于 dispatch。

| 配置 | ANE rows | prediction（ms） | 并行 FFN（ms） | 并行含 staging（ms） |
| --- | ---: | ---: | ---: | ---: |
| c320 × 4 | 1280 | 53.467 | 57.757 | 60.252 |
| c768 × 2 | 1536 | 46.778 | 52.389 | 55.172 |
| c1536 × 1 | 1536 | 40.687 | 52.414 | 55.151 |

重新平衡后的并行窗口下降约 9.3%。单次大图继续减少 prediction 时间，
但完整 FFN 仍近似持平；没有证明 c1536 比 c768 端到端更好。候选 relative
L2 均为 0.002005，cosine 0.999998，无 retry。

## C：缩小两分支差距，单 chunk 1,792 rows

顺序 c320 → c1536 → c1792 → c1792 → c1536 → c320。c1792 的 GPU 为
2,304 rows；下表仍是这组自己的配对数据，不跨组拼接分母。

| 配置 | prediction（ms） | worker（ms） | 并行 FFN（ms） | 含 staging（ms） | GPU 完成后的 join（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| c320 × 4 | 53.430 | 56.318 | 57.762 | 60.282 | <0.001 |
| c1536 × 1 | 40.681 | 43.864 | 52.444 | 54.957 | <0.001 |
| c1792 × 1 | 46.547 | 50.219 | 51.491 | 54.020 | 3.350 |

c1792 相对 c320 的并行 FFN 耗时减少约 **10.9%**，含 staging 约 **10.4%**。
虽然 c1792 的 prediction 比 c1536 慢，完整并行窗口略好，说明应优化
分支平衡而不是只追求最低 prediction 时间。relative L2 0.002160、cosine
0.999998，无 retry；候选多处理一部分 ANE rows，误差范围不完全相同。
尚未将单层约 10.9% 写成整请求收益。

## 证据、内存与限制

三组 summary complete，共 16 个 trial、160 个计时样本。所有 probe
返回 0，实际形状自检、GPU/小图 CPU 数值检查和失效恢复门禁保持原状。
小图 CPU reference 来自测试套件，大形状组件只与 GPU reference 比。
重新读取原始 stdout，逐样本中位数与报告一致；16 份独立采样流重验，
原始文件/报告/采样工具 SHA 和关联记录匹配。

采样数 A/B/C 为 150 / 247 / 212，共 609，最大间隙不超过 110.047 ms；
全组无新增系统 swap-in/out 或 compression。组件进程 footprint 峰值
约 1.59–2.10 GB，但这里仅持有该层所需权重，不代表完整模型或外部
Core ML 服务/driver/wired 内存。`cpuAndNeuralEngine` 不证明 ANE residency。

本地原始证据，不随源码分发：

```text
outputs/runtime-ane/qwen-c320-c640-matched-rows/
outputs/runtime-ane/qwen-chunks-rebalance-1536/
outputs/runtime-ane/qwen-chunks-rebalance-1792/
```

新图只改变 `--rows`；其余固定为：

```sh
.venv/bin/python3 -B tools/coreml/export_runtime_ane.py \
  --kind swiglu --rows 1792 --hidden 4096 --width 12288 \
  --tile-k 1024 --tile-n 512 --lora-inputs \
  --output outputs/runtime-ane/new-qwen-c1792-v2

build/native/ane-runtime-probe MANIFEST 10 4096 ne \
  models/Comfy-Org-Qwen-Image-2.1/diffusion_models/qwen_image_2.1_bf16.safetensors \
  1 simd
```

输出目录须不存在。c320 对应最后的 chunks 参数为 4；c640/c768 为 2；
c1536/c1792 为 1。不能把 c1792 直接当作 512² 的快路径：短于 chunk 的
输入由现有 scheduler 走完整 GPU。对 LoRA/编辑的影响需单独实测。

## 整模型对照：已完成

c1792 → c320 → c320 → c1792 的交错对照，每 trial 独立 resident
batch，一冷两热，共完成 12 次 Qwen base 1024²/40 请求。使用现有
`runtime_ane_model_screen.py --routes runtime --chunks auto --sample-memory`，
同一 e55 库、狐狸雪景/seed42、无 LoRA；不固定组件实验的分区，检验真实
调度是否仍能获得收益。各 trial 预检竞争推理，期间没有构建或测试。

证据目录：`outputs/runtime-ane/qwen-base-1024-c1792-abba/`。
主 summary 与四份子 summary 均 complete；重新验证 12 份原始结果、
请求尺寸/步数/seed、manifest/运行库身份、计时与累计调用。

| Trial | Chunk | 冷请求（s） | 热请求 1（s） | 热请求 2（s） |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1792 | 163.945407 | 152.437024 | 153.601948 |
| 1 | 320 | 169.247137 | 166.737294 | 164.884904 |
| 2 | 320 | 168.797994 | 164.467731 | 165.114878 |
| 3 | 1792 | 157.106893 | 152.497107 | 153.814630 |

每配置全部四个热请求池化：c320 中位数 **164.999891 s**，c1792
**153.049528 s**；加速 **1.078082×**，耗时减少 **7.24265%**。
这是整请求（含 VAE/PNG）收益，不是把组件 10.9% 直接当端到端结果。
该组没有 GPU/冻结图 trial；不能使用上一组的 187.800/163.665 s 作为
本组匹配分母，也不据此宣布 c1792 已通过对冻结图的同组胜出验收。

两配置每 trial 冷/热1/热2 的 hybrid blocks 均为 1,184 / 1,248 / 1,216，
完整 GPU probes 为 96 / 32 / 64，请求总 block 数均为 1,280。
c320 的 predictions 为 4,448 / 4,896 / 4,864，c1792 降至
1,184 / 1,248 / 1,216，每个 hybrid block 一次 prediction。
全部无失败、fallback 或 overflow retry，headroom=1。收益伴随 chunk
大小及实际分区变化，不能只归因于更少 dispatch。

四份独立采样流、报告、工具 SHA 和 correlation 重新核对通过，共
**19,336 样本**，最大间隙 **115.172125 ms**。trial2 有 65,536 bytes
系统 swap-in，其余为零；全组无新增 swap-out/compression。c320 的
进程树峰值 footprint 为 31.909–31.953 GB，c1792 为 32.057–32.125 GB
（十进制）。范围含加载/冷/热/退出，不覆盖全部外部服务/driver。
c1792 估算 runtime bytes 为 1,214,251,008、slots 为 463,470,592，
与实测进程 footprint 不是同一口径。

首个 c1792 图加载为 7.372821 s，热请求复用后 hybrid_setup 约 68 μs；
不能忽略初次加载或缓存影响而宣称普遍冷启动优势。
已查看首个候选冷/末次热图，并在全组结束后打开反向 c320/c1792 的
末次热图：狐狸主体、姿态、构图、松枝雪景和毛发观感接近，未见明显
质量退化。只有一个 prompt/seed，不代表广泛质量或编辑验收。
同配置正反向的对应请求 PNG 哈希一致，但冷/热1/热2 之间并不相同；
肉眼接近不要求逐像素相同，也不删除数值/状态隔离门禁。

### 作为 optional 使用

保留上述 `--rows 1792 --tile-k 1024 --tile-n 512 --lora-inputs` 独立图，
在既有 CLI 用 `--hybrid-mode runtime --ane-manifest MANIFEST` 显式选择，
chunks 仍使用 `auto`，正式计时不启用 profile。无需新增开关或重编译产品。
c320、冻结 base 图、普通 GPU 均保留，默认不变；c1792 只新增 1024²
base 实测依据，不宣称通吃 512² 或不同参考图数。

LoRA 验证必须实际执行 ANE：c1792 对 512² 无参考图的短 FFN 会由
scheduler 直接走 GPU。当时共图工具只接受生成，不能用路由标签证明
大 chunk 的 LoRA 运算。后续已补齐三图长 prefill 编辑和工具/CLI 门禁，
检查真实 prediction、完整修正和返回 base 隔离，见[后续记录](runtime-ane-qwen-edit-chunks.md)。
该组三图 auto 为 0.977×，尚未超过 GPU；不扩展到其他未验收场景。

此前文档/便携性检查及随后探针整理的实际验证见
[收尾记录](acceleration-cleanup.md)。上述组件性能来自整理前的探针 SHA；
整理只统一统计数据来源，不能重标为新的组件性能 after。产品运行库不变，
未 stage/commit、未清除任何旧图或原始证据。
