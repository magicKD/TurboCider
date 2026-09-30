# Runtime ANE：完整 GPU block 探测与整理收尾

2026-09-29，接续 [unsplit 回退](runtime-ane-unsplit-2026-09-28.md)。
本轮只保留和验证已有优化，不增加新的推理路线，不改变默认选择。

## 结论

Qwen 六步 runtime-weight LoRA 热中位 **8.777 s**，普通 GPU **8.159 s**，
GPU/runtime **0.930×**，仍慢约 **7.6%**。上一构建为 9.530 s；跨构建筛选
提示开销下降，但不是交错 ABBA 因果对照。最后两个 runtime 请求全部走 GPU，
不能将此结果称作持续 ANE 并行的收益，亦不推广为默认。

保留原有更快的冻结 base 图、完整 `lora_fused` 和显式 optional 的 `runtime`。
LoRA 始终独立：不合入 checkpoint、Core ML artifact 或 runtime base 权重槽。
冻结图与 runtime-weight 图是不同接口，不能互换 manifest。

## 保留的实现

| 调度决定 | Qwen 行为 | 用途 |
| --- | --- | --- |
| `Hybrid` | stage 与 attention 重叠，GPU head + Core ML tail，残差完成后计时 | token-row 混合候选 |
| `GpuProbe` | 完整编译 GPU block，无 staging/FFN 桥接，完成后计时 | auto 的真实 GPU 对照 |
| `Gpu` | 原完整 GPU block，保持 lazy、不加计时同步 | 无收益层回退 |
| `SplitProbe` | 保留拆图边界的全行 GPU FFN | 显式 `CHUNKS=0` 消融 |

两种受测 Qwen 路线在计时前先求值输入，计时覆盖 attention、FFN、residual，
混合路线还包括 staging；不把前面排队的工作计入当前 block。
full/split 编译函数各自缓存，不复用错误的捕获状态。
Z 仍沿用 stage-only 接口，不能把 Qwen 的完整块探测改进记到 Z 上。

`ane_scheduler.hpp` 只做调度；`ane_ffn.{hpp,cpp}` 管理张量生命周期、
stage/run/observe 配对和失败回退；`ane_runtime.*` 管理 Core ML 图及权重槽。
模型层只组装 callback，不复制 executor。未完成、错层、重复 observe 会被拒绝。
保留分段诊断，但 `TURBOCIDER_RUNTIME_ANE_PROFILE` 默认关闭。

新增累计计数 `full_gpu_probe_blocks_session_total` 和
`full_gpu_probe_seconds_session_total`；probe 是 unsplit GPU 的子集，后者又是
GPU block 的子集，不是额外执行一遍模型。Qwen 混合样本的调度时窗已经从 FFN
扩到完整 block，旧构建的分段计时不能直接拿来相减解释新构建耗时。

## 实验依据

原生库 SHA256：
`f2872ffb2337d1b438fdf8e7e0833c3a36181a16282d371d0a28b2b52afd8c7b`。

M4 Max 64 GB；Qwen-Image-2.1 BF16；Viggle v0.2.1 r256，6 步；512²；
狐狸提示词、seed42；resident；v2 c288/tile1024、chunks auto。
GPU → runtime，每路一冷三热，热整请求包含 VAE/PNG，排除加载和首个请求。
运行器先检查竞争负载，繁忙时等待；这只是启发式检查，不保证设备独占。

| 路线 | 冷请求 | 三次热请求 | 热中位 |
| --- | ---: | --- | ---: |
| GPU | 13.499576 s | 8.186042 / 8.108328 / 8.158859 s | 8.158859 s |
| runtime v2 | 11.808535 s | 9.078417 / 8.705703 / 8.777301 s | 8.777301 s |

runtime 每请求计数（由累计值作差；每次 32 层 × 6 步 = 192 blocks）：

| 请求 | hybrid | unsplit GPU | 其中完整 GPU probe |
| --- | ---: | ---: | ---: |
| 冷 | 98 | 94 | 64 |
| 热 1 | 36 | 156 | 0 |
| 热 2 | 0 | 192 | 32 |
| 热 3 | 0 | 192 | 32 |

累计 134 次 Core ML prediction，无失败回退、无 headroom 重试；每请求均绑定
227 个 LoRA 投影。auto 判定当前混合候选不划算，随后关闭混合并保留周期性探测。
即使全回 GPU，逐块探测同步、运行时管理和请求准备仍有成本；目前没有足够的
分项证据把剩余 0.618 s 全归因到某一种开销。

本地原始依据：`outputs/runtime-ane/qwen-runtime-lora-v2-full-probe-screen/`，
包括 `summary.json`、每路 `stdout.jsonl`、请求 JSON、进程快照和 PNG。
这些产物不随源码分发。最后一对 PNG 肉眼主体、构图和色调接近，局部细节有差异；
但最后 runtime 请求无 ANE prediction，不是持续混合路径的广泛质量验证。

两个 trial 前后 swap usage 均为 958.38 MiB，swapins 均为 2742；只能说明
快照间未观察到新增 swap-in。MLX peak 不含 Core ML/OS，不是完整内存认证。
`cpuAndNeuralEngine` 仅是策略，实际物理 ANE residency 仍 unknown。

## 复现与 optional 用法

先准备对应模型、adapter 和 v2 manifest。下面路径均相对仓库根目录，
`path/to/...` 需换成自己的 artifact；GPU 请求模板不写入机器专用 manifest。

```sh
# plan 不加载权重，不替代实际生成检查。
build/native/turbocider plan examples/requests/qwen21-viggle-runtime-lora-512.json \
  --hybrid-mode runtime --ane-manifest path/to/qwen-runtime-v2/manifest.json

# 同一个 v2 base-only 图可以给 base 与 runtime LoRA 使用。
build/native/turbocider batch models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-base-512.json \
  examples/requests/qwen21-viggle-runtime-lora-512.json \
  --hybrid-mode runtime --ane-manifest path/to/qwen-runtime-v2/manifest.json

# 独立的性能筛选；每次用新输出目录，避免覆盖依据。
.venv/bin/python3 tools/validation/runtime_ane_model_screen.py \
  --model models/Comfy-Org-Qwen-Image-2.1 --model-id qwen-image-2.1 \
  --runtime-manifest path/to/qwen-runtime-v2/manifest.json \
  --lora models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors \
  --output outputs/runtime-ane/new-qwen-lora-abba \
  --routes gpu,runtime,runtime,gpu --steps 6 --warm-repeats 3 --chunks auto
```

省略两个 ANE CLI 参数就是模板中的 GPU 路线。`CHUNKS=0` 不是这个 GPU 基准。
离线导出器与组件 probe 保留在 `tools/coreml/`、`tools/native/`，不会变成生成
依赖；整图对照与共图切换集中在 `tools/validation/`。

## 验证及边界

- 本构建已通过 host 4 项、报告/preflight 17 项、CLI 11 项，以及显式
  Core ML 7 项微图/集成。Qwen full/split/probe 合成切换 relative L2 为 0；
  LoRA 微图最差 relative L2 0.00533939，Q4/Q8 为 0.00658136。
- 调度微测用只有一个合法 chunk 的形状检查 on/off，避免行数平衡器合法改变
  chunk 后误判控制器；未改生产策略，也未放宽数值容差。
- 本轮整理补跑 `make test-qwen21` 和完整 `make test`，均成功退出；随后
  单独复跑包含新增路径守卫的 layout 7 项，也通过。专用 audit/test-hook、
  GPU opt-in 与缺失夹具用例按既有门禁跳过，不能当成已覆盖。
- 本构建后来补跑了 Qwen base 三路（见下节），Z base 和真实共图切换尚未
  在本构建重跑。此前 `fb79…` 的 Z base 对照和两模型状态切换仍作为历史依据，
  不能标为本构建成绩。
- Qwen runtime-weight 1024²/编辑、多训练 LoRA 质量、真实 Q4 checkpoint、
  完整内存/overlap/ANE residency 和不可变 artifact lease 仍未验收。

整理不等于整个性能研究目标完成。已知负结果和诊断工具保留，但不自动启用；
没有删除模型、Core ML 缓存或原始实验依据，没有改动既有暂存区。

## 接续补测：本构建 Qwen base

同一 `f287…` 原生库，512²、40 步、无 LoRA，GPU → runtime → frozen，
每路一冷两热。沿用上述狐狸提示词、seed42；runtime c288/auto；冻结图
仍为原有 6144-channel W8A8，没有更换 GPU 基线。

| 路线 | 两次热请求 | 热中位 | GPU/该路线 |
| --- | --- | ---: | ---: |
| GPU | 42.747952 / 42.778913 s | 42.763432 s | 1.000× |
| runtime | 39.653162 / 39.731179 s | 39.692170 s | 1.077× |
| frozen | 30.203308 / 30.206585 s | 30.204947 s | 1.416× |

证据：`outputs/runtime-ane/qwen-base-full-probe-regression/`。
runtime 累计 3648 hybrid、192 完整 GPU probe，无失败或溢出重试；
对应上一 `fb79…` base runtime 38.988 s，本轮稍慢。每块完整计时所加的
输入/残差同步是接续优化的候选，不能仅凭两个跨构建样本断言全部差距来自它。
已查看最后 GPU/runtime PNG：主体、构图与色调接近，局部毛发/枝叶细节有差异。
三路前后 swap usage 958.38 MiB、swapins 2742 不变，仍非完整内存压力资格。

## 接续诊断：LoRA 文件校验成本

直接调用同一 `tc::sha256_file`（函数未改），对 1,359,147,904 字节的
Viggle adapter 做五次独立计时。空闲复测首轮 0.567216 s，其后
0.563764 / 0.561019 / 0.560767 / 0.561478 s，热中位 **0.561249 s**。
五次 SHA 均匹配已绑定 adapter；原始依据为上方 LoRA screen 目录内的
`sha256-cost-idle.jsonl`。先前与 host 编译重叠的探针另留 `sha256-cost.jsonl`，
不取代此次空闲数据。

原运行记录中，runtime 最后两个请求扣除 denoise、VAE、text/image encode、
hybrid setup 后约余 0.578 / 0.577 s；GPU 对应约余 0.035 s。
结合代码在每次 runtime 请求全量哈希，说明身份校验足以解释大部分非 denoise
差距，而不是把这段时间算成 ANE FFN。它不是同一请求内的逐段 trace，不能把
0.561 s 与整请求差距精确一一相减。
没有移除 SHA，也没有把不可信可变文件仅凭 size/mtime 缓存：同大小、同 mtime
替换仍必须失效。后续如优化，须先具备适配器快照/不可变身份的可靠边界。
