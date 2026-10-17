# 公开vImage与GPU媒体交接：保留CPU收益，也保留更强GPU分母

<!-- knowledge
id: technique-media-case-studies-bgra-to-model-input-cost
type: case-study
aliases: evidence-media-stage-20261016
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[总目录](../../README.md) · [机制](../tensor-texture-interop.md) · [SDK来源](../../sources/official/vimage-stage.md)

2026-10-16，Apple M4 Max，macOS26.6.2/build25G83、SDK26.5、clang21、installed MLX0.32.0。没有模型推理、视频decode/encode、产品重建/默认修改、私有指令或系统cache/权限调整；只对一个完整媒体交接阶段做受控实验。

## 输入、消费者与来源

已有Z-Image输出位置的PNG提供真实512×512 RGB样本，独立标准库PNG decoder核对CRC/解压边界/所有filter并与**实际链接的产品PE decoder**逐字节SHA一致。另一输入为513×257坐标/通道相关合成控制，seed17/29。两种输入均转成明确的padded BGRA并加入人工alpha/padding；每group的原字节与XOR1版本交替，用于抓stale，不声称捕获了真实视频sample或新模型hold-out。

16个scenario=两种source×8种离散方向。三arm是source-bound产品循环重写（保留周期cancel check）、公开vImage geometry/channel convert、compiled GPU配方；全部从prepared CPU BGRA到ready contiguous FP32 `[1,3,1,H,W]`，包含输入构造、归一化、布局和drain。解码、source准备/poison、验证在计时外。这里额外强制相同contiguous输出，不是完整产品reader调用或整请求时间。

## 全部尝试与负控

- 首轮遗漏显式CoreGraphics链接，编译失败、native/GPU未启动；保留原source、stderr hash和终态receipt，不叫硬件unsupported。
- 修正链接后10组探索：624次完整调用；独立进程30组确认：1,584次。合计2,208次（1,920 measured，288 first/warm），全部通过完整FP64/byte-recovery/source/host-guard门槛。最大observed abs error为`5.9370901084321304e-08`，未改1e−6预算。
- 两run各16次实际新GPU wrong-stride与16次wrong-channel求值，全部finite但被完整输出拒绝；另各16个上一实际GPU输出对新generation的CPU stale观察拒绝，不冒充新dispatch。总计64个新GPU负控求值与32个cached观察，不等于物理kernel数量。
- 当前位于旧Qwen capture目录的PNG有239,847/262,144个alpha255，其余alpha253/254。固定opaque-file gate在native前拒绝，保留文件hash与计数；Pillow只作另一次独立交叉检查，不是核心依赖。旧receipt没有image内容hash，本轮只证明当前文件状态，不能证明自原采集以来未变；未诊断为何有这些alpha，也未将拒绝改成模型画质或编码器归因。

## 30组确认的完整阶段中位数

单位µs。`o0`不转向；其余方向按bit合同，不用EXIF名称冒充视频display transform。

| source / orientation | product loop | vImage | compiled GPU |
| --- | --- | --- | --- |
| real file / o0 | 380.396 | 266.250 | 250.688 |
| real file / o4 | 423.021 | 281.521 | 245.355 |
| real file / o7 | 478.375 | 271.584 | 244.208 |
| synthetic / o0 | 289.917 | 243.917 | 216.667 |
| synthetic / o3 | 288.917 | 285.625 | 208.396 |
| synthetic / o6 | 279.166 | 297.750 | 195.395 |

真实文件8方向的paired loop/vImage约1.350–1.731×，vImage/GPU约1.071–1.238×，GPU方向均更快。矩形o3/o6的paired loop/vImage约0.966/0.946×；o3两组中位之比与paired中位方向不同，原始pair全部保留，不只选好看的aggregate。

旧wrapper同时存了逐trial `min(loop,vimage)/gpu`，这是不可部署的乐观oracle选择，不是固定baseline或调度策略。公开独立复核保留旧值、明确标记，并额外计算固定loop/GPU与vImage/GPU，不用per-trial挑赢家主张产品收益。first调用含编译/运行时准备且会受同process公共program先后影响，不当成逐arm独立冷启动；没有bootstrap显著性、长期热状态或跨机复现承诺。

## 可迁移结论与边界

应比较完整下游消费者：CPU库函数更快，不保证“CPU准备+GPU”赢过更少中间表示的GPU配方。vImage非原地多pass的scratch与方向选择很重要；BGRA输入逻辑import字节数比紧密RGB更大，GPU仍赢不等于无copy、低DRAM流量或真实texture alias。

curator只读明确原件，复核648项native source、五个产品配方文件、每run source/binary/config/native、383 headers/三library与终态receipt；标准库独立重建完整PNG/synthetic BGRA、所有方向和generation RGB SHA、顺序/门槛/host依赖与fixed-pair算术。它不重跑GPU FP32输出或证明硬件attestation。SDK header hash是本轮读源码身份，不回填未采集的actual-loaded-framework字段。

产品采用暂缓：真实CVPixelBuffer持有/锁/producer完成、resize/color/HDR/alpha、codec实际启用、PTS/FPS/EOF/flush、完整视频/VAE请求、CPU/GPU帧流水、逐arm峰值/低RAM/长时/取消仍缺。当前结果不与旧FFN收益相乘，也不授予完整模型或physical overlap资格。
