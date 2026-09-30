# Runtime ANE：Z-Image 完整 block 调度

接续 [Q4 兼容修复](runtime-ane-q4.md)及
[Qwen 的周期采样](runtime-ane-periodic-sampling-2026-09-29.md)。
本轮将 Z-Image 的 legacy stage-only 接入改为完整 block 计划，沿用同一个
`RowScheduler`，不另建模型专用调度器，也不改变普通 GPU/冻结图默认。

## 实现

- 在选择 Z block 实现之前调用 `plan_block()`；disabled/GPU probe 直接走
  原有 `z_block()`，保留完整 BF16 编译图或原生 GGUF/LoRA 投影。
  GPU-only 不再经过 split FFN 回调、权重 staging 和每块计时 fence。
- 测量窗口是输入已求值之后至最终 residual 输出，不将前一个 lazy block
  算进本块；GPU 和 hybrid 的 on/off 决策覆盖相同计算范围。
- 稳定混合块沿用 `HybridUntimed`，仅在初始化、分区改变、周期复查时完整
  计时。FFN 的返回输出仍已求值且由调用者持有，最终 residual 可保持 lazy。
- 完整 LoRA gate/up-before-SiLU、corrected-hidden down-LoRA、失败整段 GPU
  重算、取消/drain 和 adapter 切换隔离均保持原有契约；不合并 LoRA 权重。
- 固定 chunks 与 `chunks=0` 仍显式调用 stage/run；0 保留 split GPU 消融，
  不是普通 GPU 性能基准。短序列没有合法 ANE chunk 时直接走完整 GPU。

这对应设计中的 whole-block on/off、periodic reprobe、attention-hidden staging；
参考 VPIPE 的 block plan/probe 与 persistent worker，但不照搬其 LoRA
staged-weight 合并。双缓冲、W8A8 runtime、QKV offload 不在本轮实现范围。

## 构建和正确性

原生库 SHA256：
`bdc78921f6b07c170faae8d66fd3f57ff137fa94a87cfe3c2613bf0ea0c3936c`。

原生构建成功。4 host + 8 Core ML/MLX 图/集成通过。新增测试先构造但不
求值 residual，再执行不同输入的下一次预测，之后检查旧 residual 与快照
一致；用于覆盖本轮减少同步后的输出所有权。测试也保留 full-probe、
untimed、失败 tail 重算和取消路径。

## 改动前 Q4 chunk 筛选

以下是上一 `f20b8e4…` 构建，不将它冒充新构建结果。
真实 Q4_0 来源见 [Q4 记录](runtime-ane-q4.md)。M4 Max 64 GB，512²、8 步、
狐狸雪景、seed42，resident，各一冷两热，runtime c352/auto → GPU。
证据：`outputs/runtime-ane/z-q4-c352-auto-screen/`。

| 路线 | 热请求 | 中位 |
| --- | --- | ---: |
| runtime c352 | 8.365275 / 8.303947 s | 8.334611 s |
| GPU | 9.164371 / 9.131557 s | 9.147964 s |

同轮 GPU/runtime 约 **1.098×**。此前 c288 为 8.597302 s（同构建、另一轮
GPU → runtime），c352 名义快 3.1%；不同顺序的小样本不作为严格参数因果
对照。没有复制或再编译 checkpoint，继续使用现有无权重常量 runtime 图。

## 新构建：Q4 与 BF16/冻结图

以下均为 `bdc7892…`，512²、8 步、原狐狸提示词/seed42、每路一冷两热。
Q4 顺序 runtime → GPU；BF16 顺序 frozen → runtime → GPU，串行执行，
每路有竞争负载预检。预检不等于整个测试期间设备独占。

| 工作负载/路线 | 热请求 | 中位 | 自身 GPU/该路线 |
| --- | --- | ---: | ---: |
| Q4 GPU | 9.101692 / 9.148307 s | 9.125000 s | 1.000× |
| Q4 runtime c352 | 8.229801 / 8.127761 s | 8.178781 s | 1.116× |
| BF16 GPU | 7.010695 / 7.015735 s | 7.013215 s | 1.000× |
| BF16 runtime c288 | 6.608536 / 6.575317 s | 6.591926 s | 1.064× |
| BF16 + frozen W8A8 | 5.348733 / 5.351068 s | 5.349900 s | 1.311× |

证据：`outputs/runtime-ane/z-q4-c352-full-block-screen/` 和
`outputs/runtime-ane/z-full-block-base-regression/`。

Q4 runtime 相对上一 c352 的 8.334611 s 名义快 **1.9%**，但这是跨构建
小样本，不将全部差额归因于某一个 fence。BF16 runtime 相对上一 6.551292 s
名义慢约 **0.6%**，尚在需重复确认的小差异范围，不能称本改动提升了 BF16
吞吐。默认仍不变，BF16 冻结图最快且其 PNG 和普通 GPU PNG 分别与上一
构建完全相同。没有重新跑 Qwen 整请求，不更新它的历史性能归属。

两组 runtime 各累计 685 hybrid、83 unsplit GPU（其中 64 full probes）、
558 untimed hybrid，3 次冷 headroom 重试，无错误回退。Q4 两次热请求各
248 hybrid / 8 GPU，全部 248 个 hybrid 免去完整 block 计时 fence，
不是后端已全部退回 GPU 的伪混合成绩。

已肉眼检查 Q4 最后 GPU/runtime 图：狐狸主体、姿态、毛色及雪景构图接近，
背景枝叶细节有变化，未见明显生成崩坏；不以逐像素相等作为质量门槛。
BF16 GPU/runtime 狐狸对照也已查看，主体与构图接近，局部细节不同。

Q4 累计 stage 3.316 s，host stage wait 0.000194 s；BF16 为 1.551 s / 0.000224 s。
这说明当前屏幕负载下 staging 很少暴露为调用者等待，**不等于**测得真实
GPU/ANE 硬件 overlap 或证明 99% attention-hidden。应先看完整 block 与
更长序列，再决定是否值得为双缓冲增加一套权重内存。

## 第二提示词：人像 ABBA

同一 `bdc7892…`、同一个 Q4/c352 图、512²/8 步、seed43，提示词是窗边自然
光的成年卷发女性人像、雀斑和绿色毛衣。完整原文保留在请求与 summary。
GPU → runtime → runtime → GPU，每个 trial 独立 resident batch、一冷两热。
四次热请求全部汇总，不挑最低值：

| 路线 | 热请求（按执行顺序） | 中位 |
| --- | --- | ---: |
| GPU | 9.376657 / 9.451386 / 9.377981 / 9.386839 s | 9.382410 s |
| runtime | 8.480129 / 8.449615 / 8.561975 / 8.480454 s | 8.480291 s |

GPU/runtime **1.106×**。证据：
`outputs/runtime-ane/z-q4-c352-full-block-portrait-abba/`。
两次 runtime trial 分别 684/685 hybrid、84/83 GPU、64 full probes、558
untimed hybrid，均无错误回退。已肉眼查看第一 GPU 和第二 runtime 的最后
PNG：脸部结构、眼睛、卷发、毛衣与窗边构图接近，皮肤/发丝细节不同。
这是第二提示词的视觉和重复性能证据，不代表广泛人像、文字、编辑资格。

上述新构建测试前后 swap usage 都为 958.38 MiB。BF16 frozen trial 的
系统 swapins 从 2790 增到 2794，其后以及人像四个 trial 保持 2794；
不能声称所有测试都零换页，更不能把系统变化全部归因于本模型。
仍缺过程级峰值 RSS/wired/Metal resident/pressure 完整采样。

## 完整 LoRA 共图状态回归

同一构建、graph-v2 c288、固定 chunks=1，warm base → base → trained A →
synthetic B → base。Z 的训练 adapter A 仍是原始 distill patch，没有改写或
合入 base；B 仅将 A 的 LoRA B tensors 临时缩放为 0.5，作为状态隔离测试。

`outputs/runtime-ane/z-runtime-lora-v2-full-block-switch/` 验证通过：
Core ML 累计调用 259/515/771/1027/1283，LoRA 绑定投影 0/0/238/238/0；
首尾 base PNG SHA 相同、A/B 输出不同，图未重载。此测试固定分区，不作为
auto LoRA 性能或第二个训练 LoRA 质量证据。开始前检测到 ComfyUI 推理并
等待一次后再运行；未终止其他进程。

随后 `make test-qwen21` 与完整 `make test` 成功退出；后者包含
`make test-acceleration-contract` 的 4 host、18 报告/预检及 11 CLI。
仓库布局 8 项和 125 处文档链接/锚点检查通过。缺失夹具、专用 audit/
test-hook 构建、GPU opt-in 等既有 skip 不计为已覆盖。微图 SDK 仍有
临时目录清理警告，但测试成功退出。
暂存区摘要未变，249 个原有暂存移除对应文件均在本地；没有提交或删除产物。

## 保留决策与下一步

保留完整 block 计划：让 Z 与 Qwen 使用同一 on/off 窗口、正确恢复原 GPU
实现，减少稳定混合块的计时同步。Q4 c352 保留为显式可用候选；不自动
改选 manifest，不将 Q4 收益外推 BF16、Q8、其他机器或任意 LoRA。
冻结图与普通 GPU 完全保留。

仍需 Qwen runtime 1024²/编辑、更多 LoRA 视觉与 auto 性能、真实设备
timeline、完整内存/压力资格，以及后续双缓冲/QKV/runtime INT8 等设计项。
本轮是有效进展，不据此宣告整个设计目标已完成。

## 后续同库 BF16 512² 对照

与 [Qwen QKV auto 40 步复核](runtime-ane-qwen-qkv-auto.md) 使用完全相同的
`875b0d3dcf5c766feeced21d71f1a366acf0a1238bc3f0ed332098aa3b27b8cd`
库。Z-Image Turbo BF16、512²/8 步、同一狐狸提示词与 seed42、resident，
runtime c352/auto，匹配的原冻结 W8A8 图；每条路线一冷两热，分别按
GPU→runtime→frozen 和 frozen→runtime→GPU 顺序串行测试。下表为热请求
`request_wall` 中位数（秒，含 VAE/PNG）；不拼接其他构建的时间。

| 路线 | 正序 | 反序 | 四热样本均值 | GPU/路线 |
| --- | ---: | ---: | ---: | ---: |
| GPU | 6.991 | 6.992 | 6.991 | 1.000× |
| runtime FFN c352 | 6.342 | 6.387 | 6.365 | 1.098× |
| 冻结 W8A8 | 5.348 | 5.343 | 5.346 | 1.308× |

六个 trial 的 100 ms 独立进程树采样均验证完整、零采样 swap-in/out；
峰值 footprint 约 GPU 22.948 GB、runtime 23.013 GB、冻结 23.071 GB，
不覆盖外部 driver/service 的全部归因。两个 runtime trial 各 685 hybrid、
83 普通 GPU block、688 次 ANE prediction，均无失败/错误回退；两个
冻结 trial 各 768 次 prediction、无失败。目视查看正序末次热图：狐狸
主体与构图接近，毛发、枝叶和背景有差异；GPU/runtime RGB MAE 为
7.687/255，GPU/冻结为 14.308/255。该单提示词不是质量资格验收。

原始请求、PNG、JSONL、内存流、验证报告及 `status=complete` 的 summary
保存在 `outputs/runtime-ane/z-image-512-ab-875b/` 与
`outputs/runtime-ane/z-image-512-ba-875b/`。此 BF16 负载冻结图最快，
runtime FFN 仍优于 GPU；不改变默认路线，也不外推到 Q4/LoRA/编辑。
