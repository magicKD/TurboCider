# Qwen-Image-2.1 加速路径索引（2026-09-27）

验收优先级：先检查文生图的主体、构图和可辨细节，图像编辑还须检查
参考主体及明确的编辑指令确实得到保留；通过肉眼观感门槛后，优先
选择端到端更快的路径，不以逐像素、RMSE 或相关系数接近为硬性
质量门槛。对小字、身份特征、贴纸位置等任务关键细节应逐项检查；
同输入、同缓存状态的速度收益必须重复确认，遇到质量退化或退速
保留 GPU/原尺寸回退。以下诊断数据不自动构成 CLI 默认推广依据。

## 入口与运行代码

| 责任 | 代码 | 要点 |
| --- | --- | --- |
| 请求准入、显式诊断开关 | `native/models/qwen21_module.cpp` | `plan` 阶段拒绝未验证的参考尺寸、LoRA、精度组合 |
| 诊断开关与序列 tile 覆盖解析 | `native/models/qwen21/diagnostic_options.hpp` | 统一 plan／执行时的开关解析（仅字面值 `1` 启用）及 0/8/16/20/24/32 层映射 |
| 原生 Session 与跨请求缓存 | `native/models/qwen21/pipeline.cpp` | 条件编码缓存、resident prefix-KV 身份、首步/后续步路由与调用计数 |
| DiT 注意力和 FFN 调度 | `native/models/qwen21/transformer.cpp` | 前缀 K/V、最后一层 target-only、后半层 reference-local、step-FFN 缓存 |
| GPU/ANE 分片和序列 tile loop | `native/models/qwen21/hybrid.cpp` | 依 manifest 把 FFN 前 4096 或 6144 channels 交给 Core ML W8A8；最快受测 6144/BF16 GPU 后缀；首步 1024-row 重复调用 |
| 请求报告与近似标签 | `native/platform/apple/results.mm` | `plan` 和实际调用返回所用近似；Core ML 策略不代表已证明 ANE 驻留 |
| 运行与比对 | `tools/native/qwen21_session_probe.mm`、`tools/validation/qwen21_compare_sessions.py` | 准备/缓存/取消/切换、每请求计数和同输入 hit-only 比对 |

正常 CLI `generate` 是一次请求，不会得到跨请求 prefix hit；必须在
同一 `resident` Session 中重复相同有序参考图和 prompt。默认仍是 BF16
GPU；`execution=gpu_ane` 需要显式 manifest 和近似允许。GPU FFN
W8A16 后缀已在两图全尺寸五步样例中比 BF16 略慢（17.19/17.20 s
对 16.98/16.93 s），目前最快测试路径使用 BF16 GPU 后缀。

矩形 768×512／512×768 输出另有显式诊断：单层 decode 输入的 1536
行用原 1024-row W8A8 图执行两次，首步仍为 GPU；两种方向的 base
20 步单样本 warm 请求相对 BF16 GPU 分别约 1.150×／1.146×。
新增同门禁下的 DBCache 叠加，横向 40 步文生图 warm 请求
GPU `65.763 → 31.902 s`、hybrid `56.164 → 26.972 s`；
纵向 40 步文生图 GPU `65.680 → 31.951 s`、hybrid
`56.768 → 27.500 s`；
横向 20 步两图原尺寸编辑 GPU `54.259 → 38.191 s`、hybrid
`50.627 → 35.621 s`，单次同输入且主体可辨。三图矩形
resident hybrid 在 64 GiB 机器上先被保守内存门禁拒绝，
不能外推 warm 表现；同样开 DBCache 的 component-staged
冷请求可运行但 GPU `58.403 s`、hybrid `63.935 s`，没有净加速。
质量、调用数与限制见[矩形复用实验](qwen21-rectangular-w8a8-2026-09-27.md)。

Viggle 六步 runtime LoRA + base W8A8 前缀的 GPU 后缀 LoRA 实验
只在文生图单样本得到 1.027×，一图编辑约 1.004×，且未包含
ANE 前 6144 通道的非线性 LoRA 校正；不能默认为完整学生加速。
两边同样加上 FP16 rank matmul 和融合 Q/K 后，文生图两次
约 1.22×；一图编辑两次分别 1.119×／0.996×，不稳定。
双／三图编辑各一次分别为 0.992×／0.998×，没有加速。
新增显式后 16 层参考图局部注意力后，双／三图原尺寸参考图编辑
在同 prompt/seed 两次 warm 请求中的 hybrid/GPU 中位速度比约
1.053×／1.028×，肉眼仍保留两壶及三图样本的龙贴纸；提升太小，
不是稳定的 1.3×，且没有不同种子／编辑任务的质量门禁，仍不作为
默认 CLI 路线。质量以主体、指令和参考细节的肉眼保留为主要条件，
数值像素差只用于排错；若基线编辑本身失败，不因候选接近基线就
宣称成功。
单独开启 512px 参考缩图后，六步 LoRA 双／三图编辑的同输入
resident warm 请求各重复两次，GPU/hybrid 配对倍速中位分别为
1.139×／1.070×；两壶、居中的龙贴纸及白边肉眼保留，但细节与
原尺寸参考图不同，且仍缺 ANE 前缀 LoRA 校正。它是另一种显式
速度／观感取舍，不能用原尺寸的 GPU 时间计算严格加速比；仍缺
交错重复，默认 CLI 不自动打开。新增三图 seed 29 静物单次约
1.156×，另一个“贴纸贴在蓝壶正面”的编辑指令两路观感均合格，
但 hybrid/GPU 仅 **0.983×**，出现退速；因此不能按图数
盲目选择 hybrid。
见 [LoRA/base ANE 诊断](qwen21-viggle-base-ane-reuse-2026-09-27.md)。

## 按参考图尺寸区分

| 场景（512² 输出） | 首请求／warm miss | resident prefix hit | 质量与结论 |
| --- | --- | --- | --- |
| 1–3 张参考图，明确缩为 512px | 后 16 层首步 FFN 用同一 W8A8 模型按 1024 行循环；三图约 7.706 s，对同输入 GPU 10.114 s | 三图严格补齐 tile 尾部约 4.887 s；单独 target-only 命中约 4.673 s | 两壶／贴纸可辨，但输入缩图和数步 FFN 缓存均有损；见 [序列循环](qwen21-tiled-prefix-kv-edit-2026-09-27.md)与[命中快捷路径](qwen21-prefix-hit-target-only-2026-09-27.md) |
| 3 张原尺寸 1024px 参考图 | BF16 GPU 首步；后半层局部注意力 + 最后一层 target-only 的 hybrid ~22.80 s，对相同输入准确 GPU ~25.22 s | 同路线 3 次匹配 hit 的 GPU/hybrid 墙钟中位 8.732/7.890 s；不可与 miss 互比 | 不缩图，轻微改变贴纸／壶表面；见 [首请求](qwen21-fullref-last16-local-2026-09-27.md)与[稳态命中](qwen21-fullref-prefix-steady-2026-09-27.md) |
| 512² 文生图，5 步 | GPU 纯路径与 6144-channel W8A8 路径均可显式选择 | resident 缓存只适用于满足身份匹配的重复请求 | 按 [公开运行说明](../public/QWEN_IMAGE_21.md)核对 prompt/seed；不能把三参考图收益外推 |

这里的 `1024px` 是**每张参考图的处理尺寸**，不是输出尺寸。
以上时间均为 M4 Max 上的已准备且条件编码命中的 Session 请求，
包括 VAE/PNG，不包含模型加载；不构成通用加速保证，也没有证明
Core ML 的实际硬件执行位置。40 步时首步的摊销比例和收益不同。

20–40 步 base 可选显式 DBCache 诊断，支持 GPU 或既有 W8A8
hybrid 的相同请求内 decode 残差缓存。40 步、三张缩为 512px 的
参考图，阈值 0.25／连续跳过上限 8 的单次 warm 请求 GPU
`52.305 → 27.131 s`（1.928×）、hybrid `39.497 → 21.375 s`
（1.848×）；后者比同条件 GPU 候选快约 1.269×。主体与贴纸
肉眼仍可辨，但局部把手／阴影差异更明显；保留为实验选项，未通过
多种子和重复测试。阈值 0.08 在先前单样本跳过零步，不应将实验
倍速误写为默认收益。详见 [DBCache 实验](qwen21-dbcache-2026-09-27.md)。
单张 **1024px 原尺寸**参考图的 40 步编辑也完成独立配对：
阈值 0.25、连续上限 8 时，GPU `55.204 → 29.174 s`、hybrid
`42.443 → 23.417 s`，均保留壶形但未纠正基线本身未达成的哑光要求。
三张 **1024px 原尺寸**参考图的 40 步编辑，单次 warm hybrid
`71.692 → 43.902 s`（DBCache 1.633×）；再叠加后 16 层首步
reference-local 注意力为 `41.650 s`（相对无缓存 1.721×）。
三主体肉眼仍在，但贴纸脸型、白边和阴影变化较明显；首步长序列
是瓶颈。相同 prompt/参考图的第二个 seed 29 配对得到
`71.409 → 41.708 s`（1.712×），两组均符合主体／构图的肉眼
门槛；仍缺不同编辑任务与交错重复，保持显式诊断。
另补两张 **1024px 原尺寸**参考图的 40 步 base 四路同输入对照：
GPU 无缓存／DBCache `68.414／38.224 s`，hybrid 无缓存／DBCache
`57.308／32.797 s`；缓存 hybrid 相对缓存 GPU `1.165×`。
两个缓存方案都跳过 28 步中间 24 层，两壶构图、壶嘴与把手肉眼
保留。四路以相反顺序各重复一次独立进程，缓存 hybrid 相对缓存
GPU 第二轮约 `1.171×`；仍是单提示单 seed 的 resident warm
请求，不是冷启动或多任务质量结论。

## 撤回的诊断，不要作为 CLI 推荐配置

- 原尺寸图长序列 FFN 全层或首八层 tile loop：五步虽加速，但
  贴纸边缘／脸明显劣化；首八层 40 步还退速。见
  [全层](qwen21-fullref-tiled-prefill-screen-2026-09-26.md)、
  [首八层](qwen21-fullref-first8-tiled-prefill-screen-2026-09-27.md)。
- 原尺寸 prefix hit 的 1024 目标行额外送 W8A8：两次 hit 缩短约
  4–6%，但贴纸眼睛及轮廓明显重影，已从源码移除。见
  [质量否决](qwen21-fullref-prefix-hit-w8a8-rejected-2026-09-27.md)。
- 独立 LayerNorm fusion、first-step batched `img_in`、K/V 连续化等
  没有可确认的端到端增益，实验代码已撤回，记录保留在同目录。

修改路由后运行 `make test-qwen21`，然后在目标工作负载上分别检查
`plan` 标签、同输入张量、PNG 人工观感、实际预测调用数和
取消／GPU 切换／卸载；只对相同缓存状态的 warm 请求报告速度。
本次整理将 tile 层数解析合并到 `diagnostic_options.hpp`，并修复了
LoRA 合约测试仍固定要求旧 FP32 单一路径的问题：现在检查可选
FP16 低秩矩阵乘与 FP32 累加的真实实现。撤回候选后重建原生库和
Session 探针，`make test-qwen21`、`make test`、hybrid merge test 与
`git diff --check` 通过；真实三原尺寸参考图 prefix hit 的 PNG 与
历史 hybrid 基线逐字节相同，128 次 Core ML 预测、零错误，取消后
重试、GPU 路由切换和卸载均完成。没有把本次代码整理记为新速度收益。

后续的 DBCache 整理将 resident prefix-KV 身份失效集中到
`Session::clear_prefix_cache()`，并将 W8A8 请求级 Core ML 调用数
从 Session 和探针的两份内联算式归入可独立测试的
`expected_w8a8_calls()`。相应原生库、探针重新构建并通过
`make test-qwen21`；这些是维护性调整，不应写成新加速收益。

LoRA 后缀另增加临时 safetensors 数值测试：真实绑定融合 gate/up 的
两个独立适配器，比较全矩阵与跨 gate/up 边界的行切片、输入列分割，
并覆盖 FP16 低秩选项；同时拒绝 runtime LoRA 意外进入 base-only
整层 GPU 回退。该测试属于投影正确性，不等于完整 LoRA 学生网络
与 base ANE 前缀等价。
