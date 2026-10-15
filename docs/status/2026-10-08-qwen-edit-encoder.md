# 2026-10-08 · Qwen Image Edit encoder：1–3 图 GPU 融合加速

本轮已将语言 GQA／causal mask 与 FP32 视觉 D72→D80 融合注意力接入 Qwen Image 2.1 原生 MLX pipeline，默认启用。M4 Pro／48 GiB 使用已有本地模型，真实 1024 reference processor 的 1–3 图视觉＋语言计算部分省时 **18.70%／20.54%／21.77%**。一组 512 reference、三图、512²、六步 LoRA 的完整请求通过质量验收，但 encoder 没有明显提速；不能把组件收益宣称为整图收益。

实现和 Splash 复用边界见[设计 18](../design/qwen-image-gpu-ane-command-graph/18-edit-encoder-acceleration.md)，结构化结果与证据摘要见[JSON](2026-10-08-qwen-edit-encoder.json)。本轮实际是 **GPU**，没有 ANE encoder、private encoder、W8A8 或 VAE 加速。

## 实现与构建

- [语言 encoder](../../native/models/qwen21/text_encoder.cpp)：rows≥128 时保留原始 8 组 K/V，避免展开为 32 组；无 padding 时使用 causal SDPA，有 padding 时保留 causal/key-valid additive mask。
- [视觉 helper](../../native/models/qwen21/encoder_attention.hpp)：Q/K/V 全部 FP32、D72、rows>8 时，在 feature axis 补零到80、保留 `1/sqrt(72)`、融合 SDPA 后切回72。其余 precision/geometry 使用旧实现。
- [pipeline](../../native/models/qwen21/pipeline.cpp)：调用新配置，保留尺寸、prompt/image tokens、mRoPE、DeepStack、norm/FFN/残差边界、层级 release 和既有视觉／VAE cache。`TURBOCIDER_QWEN21_ENCODER_REFERENCE=1` 为进程固定的旧路径诊断，用于独立进程配对。

完整 private package 构建通过，身份为 `tc-runtime-build-v1-769123f8797388419e24b1d4f2077583477ae2e535f4f6a1ed900ecec17901b2`，572 个 identity sources 前后摘要一致。只有既有 `ane_gpu.mm` 的三条 deprecated `fastMathEnabled` 警告。private 是整个库的 build feature，不代表 encoder 执行 ANE。两个 probe 配置结构有变化，必须链接本次对应库，不能混用旧 binary/header ABI。

[新库目录](../../build/qwen21-edit-encoder-runtime-v2-20261008)、[完整构建日志](../../outputs/qwen21-edit-encoder-20261008/full-build-v2.log)、[CPU/链接日志](../../outputs/qwen21-edit-encoder-20261008/probe-v5-build-cpu.log)保留。package-only 没有运行全仓库测试集。

## 正确性筛选

[生成数据回执](../../outputs/qwen21-edit-encoder-20261008/generated-v3.json)：6 组两层完整语言 block（129/257/385行，各有无 padding，含 mRoPE/DeepStack/raw/final norm）输出逐位相同；6 组视觉 attention（FP32/BF16，各64/192/256行）通过。FP32 最大 RRMS 0.000547%，BF16 回退路径逐位相同。CPU 模式不加载模型或运行设备。

初始 BF16 融合 fixture RRMS 为0.8327%，未通过0.5%门槛；没有提高此门槛，而是修正 production 为 **FP32-only**。初始 probe 真实 mode 漏传必要 Event callback，在实际视觉／语言执行前退出；已修正工具。真实第一组1图 conditioning RRMS2.0023%，略超 probe 自设2%严格线，原 failure receipt保留。本轮后续采用用户此前允许5%–8%范围的较严 **5% 张量筛选**，逐项报告2%严格线仍为 false；不是数值严格等价资格。

## 真实 1–3 图 conditioning

[回执](../../outputs/qwen21-edit-encoder-20261008/real-encoder-v5.json)与[监督记录](../../outputs/qwen21-edit-encoder-20261008/components-supervisor-v5.json)使用既有 teapot/fox/puffin，每图4096 patches，经2×2 merger为1024视觉 tokens。视觉27层旧/新各一次，顺序 AB/BA/AB；1/2/3图的语言36层旧/新各一次，同样 AB/BA/AB。最终 expanded sequence 为1074/2104/3134，保留文本/image delimiters，输出36/42/48行 conditioning。

| 参考图数 | 旧计算时间 | 新计算时间 | 本组省时 | conditioning RRMS |
|---|---:|---:|---:|---:|
| 1 | 4.784 s | 3.889 s | 18.70% | 2.0023% |
| 2 | 9.554 s | 7.592 s | 20.54% | 2.1506% |
| 3 | 14.749 s | 11.538 s | 21.77% | 2.4564% |

上表为 **视觉 arm 时间之和＋语言 arm 时间**，不是完整 request 或单次 pipeline phase；source load、resize、prompt assembly 排除，首次真实 shape compile 包含。组件显式 cache cap256 MiB、视觉源物化后释放，再物化语言源。完整监督墙钟44.37 s，MLX peak约15.74 GiB，语言物化前系统可用约24.43 GiB；这不是全请求或进程容量认证。

三个视觉塔分别由约2.34 s降到1.53–1.55 s，省时33.68%–34.77%；全部 merged/三个DeepStack RRMS≤0.003%。语言计算由2.446→2.355、4.875→4.531、7.728→6.924 s；视觉占主要 encoder 收益。全组 shape/dtype/finite、实际 `eval+synchronize`、held source generation 和 reference SHA一致通过。只有每形状一对，不能宣称稳定均值或其他机型性能。

## 完整三图编辑

[监督回执](../../outputs/qwen21-edit-encoder-20261008/e2e/receipt.json)与两个 request/result 保留。512×512输出、三张512 reference、Viggle r128/strength1六步、seed42、近似DiT cache关闭，MLX执行；新进程各一次，旧→新。共享OS/file/driver cache没有重置，组件的256 MiB cache cap没有用于这组请求。

| 阶段 | 旧路径 | 新路径 |
|---|---:|---:|
| text/vision encoder（含该 phase 的加载/assembly） | 3.871 s | 3.919 s |
| reference VAE encode | 0.767 s | 0.721 s |
| denoise | 23.537 s | 23.249 s |
| VAE decode | 0.904 s | 0.911 s |
| request wall | 32.058 s | 31.669 s |

六步、227 LoRA projections、GPU encoder、3072 reference latent tokens、prompt cache miss均核对。图像像素RRMS **2.7564%**，平均绝对像素差0.923/255，目视对象位置、轮廓和光照接近。两个完整请求成功，最终图见[旧路径](../../outputs/qwen21-edit-encoder-20261008/e2e/reference.png)／[新路径](../../outputs/qwen21-edit-encoder-20261008/e2e/optimized.png)。

本组总请求少0.389 s（1.21%），encoder反而多0.048 s（1.25%），差异很小；VAE/denoise也变化，但本轮没改它们。因此 **512 reference 的完整请求未证明 encoder 净收益**，不能把总时间差归因于encoder。较大1024 reference的计算收益已见组件，默认1024完整编辑请求尚未追加验证。不同输入规模、加载/预处理和DiT占比会限制总请求加速；不为提高跑分改变图像或prompt尺寸。

## App 与清理

[App 构建日志](../../outputs/qwen21-edit-encoder-20261008/app-build.log)确认 Swift App 与 model-library helper 编译通过，链接本轮对应库；[新 App 可执行文件](../../build/qwen21-edit-encoder-runtime-v2-20261008/TurboCiderNativeApp)保留。只验证构建和 CLI 完整请求，没有 App UI 验收或替换已安装的 App。

[owned 清理回执](../../outputs/qwen21-edit-encoder-20261008/owned-cleanup.json)记录删除810个文件、243,333,892 B（约232.06 MiB），只涉及本轮两个 runtime build 与五个 probe build 的 `.o`／`.a`／module caches；对应库、CLI、App、probe、manifest、日志／回执／图片保留。本轮不下载、转换或复制模型，没有导出 Core ML／ANE模型、生成持久张量capture或清理系统／driver cache。
