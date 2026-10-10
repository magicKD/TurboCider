# Qwen Image 2.1：GPU encoder / prefill 融合

2026-10-10，M4 Max 64GB / macOS 26.6.2。接续
[同内存 BF16 streaming 对照](qwen21-bf16-streaming-2026-10-10.md)。
本轮聚焦 encoder，不将 tokenizer 缓存、DiT GPU/ANE 分工和 GPU kernel
收益混成同一个倍率。完整优化目标仍 active。

## 实现与参考

只读参考 Splash 的 `prefill/linear_q4.metal`、`normalization.metal`、
`attention_qkv.metal` 和 `QwenTarget.cpp`，以及 Unsloth 的
`studio/backend/core/inference/sd_cpp_args.py` 和 sd.cpp 的 `te/llm.hpp`。
借鉴批量 prefill、投影合并、归一化/RoPE 融合和原生 GQA；没有复制
Splash 的 group64/BF16/tiled-N256 权重 ABI。当前 GPU 消费仍为原有
group32 affine FP16/Q4 或 Q8 三平面，不展开完整 dense checkpoint。

36 层 gate/up 合并；真实 mixed-K encoder 中 18 层合并 QKV，另 18 层
只合并 QK，保留原 Q6_K→affine Q8 的 V。down 的 18 层 Q6_K 也保持
原格式，不把整个 Q4_K_M 文件误称为每个矩阵都是 Q4。
codes/scales/biases 合并前后逐字段检验，首次 pack 成本显式记录。
有 additive bias 或不同位宽的投影不错误合并。

新增 FP16/BF16 typed Metal RMS 和 Q/K RMS + split-half/NeoX RoPE。
保持 F32 reduce/scale 和 typed 舍入位置；不是 DiT 的 interleaved RoPE。
compiled block 继续使用动态原权重参数，原生 32Q/8KV GQA 不重复展开 KV。
算术允许近似，但 source、dtype、shape、finite、取消和内存准入检查不放宽。

tokenizer 首次经过原 native 全文件 SHA 和 held-fd 构造；会话复用前后
复验 named-path / fd generation。processor 身份进入 conditioning cache
key。优化开关切换会使 fused/legacy 权重布局和 conditioning 分别失效。
缓存命中报告本次 encoder 未执行，融合计数/rows 为零，不重放旧计数。
卸载和异常释放 processor owner。

## GPU-only 组件：不是完整请求倍率

同原 GGUF、同预组装输入，legacy / existing compiled-GQA / fused 三臂
循环顺序，五轮 hot samples。source load 和 pack 单列。以下为 v4 组件
构建，不与后续最终库混称一个 binary。

| encoder 输入 rows | legacy GPU ms | compiled-GQA ms | fused GPU ms | fused 相对 legacy 耗时减少 | retained hidden relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 38 | 81.254 | 80.394 | 78.874 | 2.93% | .003234 |
| 150 | 186.073 | 183.560 | 176.727 | 5.02% | .013633 |
| 534 | 620.589 | 588.411 | 575.463 | 7.27% | .025070 |

真实 retained rows 分别 24/136/520；计时处理完整 38/150/534 rows。
full native source load 2.376–2.385s；首次 pack .090–.095s，首次 fused
forward .0855/.2693/.6307s。没有省略校验或把首次编译/setup 隐藏在 hot 中。
组件同时持有 control/fused views，logical peak 不是整请求物理内存证据。
8% hidden relative-L2 只是性能初筛门槛，不是普遍图像质量认证。

## 启用范围

显式 `TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU=1` 启用 compiled GPU prefill。
推荐实验同时设置 `TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS=1` 和此机器上
已测试的 `TURBOCIDER_QWEN21_GGUF_DECODE_WORKERS=8`。请求仍需显式
`allow_approximation=true`、resident 512²；与 encoder ANE manifest 互斥。
DiT 可以保持原 GPU 或 Private ANE5120/GPU7168 通道并行。
默认关闭，不提升为 Public、内存受限、GGUF editing 或 LoRA 资格。

## 最终完整验证

最终 v7 同库，原两份 Q4_K_M / BF16 VAE，512²/40steps/seed29，三种
fresh fox-light prompts；每臂独立进程，一 cold 两 warm，conditioning 全
miss。顺序 legacy GPU→legacy ANE5120，再 fused ANE5120→fused GPU；
均保留原 encoder、8 import workers、无 shared-down/时间复用，最终
native/CLI 510 个 source inputs 全部与库匹配。

| route | warm text s | warm DiT s | warm request s | tree peak physical footprint bytes |
| --- | ---: | ---: | ---: | ---: |
| legacy GGUF GPU | .242137 | 43.769972 | 44.608988 | 15,231,639,880 |
| fused-prefill GGUF GPU | .093179 | 43.747806 | 44.448922 | 15,231,819,288 |
| legacy GGUF ANE5120/GPU7168 | .235151 | 36.159359 | 36.989817 | 15,353,700,800 |
| fused-prefill GGUF ANE5120/GPU7168 | .093040 | 36.166140 | 36.857731 | 15,387,827,952 |

encoder 整阶段分别约 2.599× / 2.527×，主要来自 verified tokenizer
复用，不能将其冒称 GPU 矩阵 kernel 的同等倍率。40 步整请求名义少
.3588% / .3571%，约 .160/.132s；DiT 本身基本不变。本窗口样本少，不
推广这一小幅整请求差为所有负载的统计显著改善。新版混合对新版全
GPU仍有约17.08%的既有总请求优势，不是本轮新增的17%收益。

cold text：GPU 2.615506→2.841643s，混合2.820216→2.985942s。
cold request：48.959308→49.247899s、41.828268→42.019684s。
首次 pack/setup 有成本，冷首请求此次反而略慢；没有把它从 cold 计时
扣掉，不能宣布 cold encoder 目标已达成。

四个40步进程树记录 complete；system swap-in/out、compression均0。
新版 GPU system decompression0，新版混合737,280bytes；系统计数不当作
单矩阵/单进程归因。混合每请求1280成功Private calls/channel blocks，
phase32/1248，fallback/failure/retry0。仍不是物理并行或硬RAM上限证明。

另有同库4步、两种DiT路线与三个提示词的24对完整text/initial/latents/
pixels tensor对照：六对初始噪声SHA完全相同；最大relativeL2分别
.00326459/0/.00609925/.00587986。40步六对完整RGB PNG最大relativeL2
.00709204、normalized RGB RMSE .00465459。重新查看混合soft-light
control/fused整图，构图/姿态一致，局部细节有微小变化；不声称bit等价
或跨场景语义认证。40步性能窗不dump，不把4步tensor证据冒充40步tensor。

12项native/host回归（含64个新typed Metal/packed病例、48个原compiled
encoder病例、mixed-K bank与BF16双阶段streaming），真实prepare-only
会话10个生命周期请求、独立PyTorch conditioning10场景全部通过、无
skip。新增CPU-only tensor质量读取/截断/nonfinite/crop负例2项通过。
会话覆盖condition-hit计数归零、fresh prompt权重复用、profile双向切换、
owned processor变更拒绝/恢复、卸载后冷重建；不是只测组件缓存。

### Unsloth 持久会话

同时测试其原Apple默认CPU encoder和显式GPU开关，不选择较慢的CPU
路线代替GPU对照。使用同pinned sd-server、原source/max flags、12 threads，
每进程一cold两fresh warm的2步请求；只取原native condition span，不用
HTTP/decoder计时冒称encoder，不从2步外推40步DiT或图像等价。
最终有效进程树与condition数据见下述机器记录。

| encoder route | cold native condition/text s | fresh warm s |
| --- | ---: | ---: |
| Unsloth original Apple CPU default | 1.48 | .45 |
| Unsloth explicit GPU | 1.17 | .13 |
| TurboCider fused GPU（40步窗口GPU臂） | 2.841643 | .093179 |

原GPU对照warm约1.40×、默认CPU约4.83×；仅为这次本机短prompt的native
condition跨度诊断，Unsloth日志只有两位小数，且双方整体步数不同。
不推广为所有encoder/完整请求的倍率。对方cold包含首次lazy source
加载，我们cold另有native全文件SHA/affine import/fusion，因此仍未
达到更快cold encoder要求。两有效参考进程树均complete；峰值约
11.054/11.427GB，低于native约15.23GB的40步窗口，但不把2步/40步
生命周期不同的峰值直接当作相同负载内存结论。

首次v7两路都实际返回三张图，但wrapper未给已知sd-server子进程登记
role，严格sampler/verifier报告unknown_child，完整memory窗口失败。
原raw和failed summary保留。新helper仅允许精确、存在的绝对可执行路径
作为已知child role，拒绝通配符；不取消unknown-child门禁，不回写旧raw。
在新目录重新运行，默认原单进程native sampler调用不变。
新增7项进程所有权/精确child role/timeout/verifier门禁测试通过，无skip。

[机器记录](../design/validation/qwen21-gpu-encoder-prefill-20261010.json)绑定
组件与最终库、原sources、四个匹配窗口、完整tensor/PNG误差、会话/
回归、原Unsloth两种placement、memory和保留失败的hash。

## 保留失败与后续重点

v2 初次构建使用错误的 split 索引容器，已修为 MLX Shape。
v5 regression 的独立 diagnostic header 编译失败（缺少 common.hpp
直接依赖），最终修复；该轮数值 unit 与 Swift 包装尾部曾重叠，不作
性能窗口。v6 真实 prepare-only 测试发现结果桥漏报 prefill 字段，修复为
prepare/generate 共用转换函数，并保留原失败日志，不改写为成功。

冷 source 验证/导入仍是瓶颈，需继续完整校验与读取/导入重叠、正确
消费原 GGUF 格式的 tiled matmul 和更丰富场景质量验证。此轮不是
Splash 式 MPP tiled-GEMM 的完整移植，也不是 Public ANE 资格完成。
本轮主要是language encoder prefill；vision tower没有重写，GGUF editing
与更多scene/seed的完整验收仍未完成。
