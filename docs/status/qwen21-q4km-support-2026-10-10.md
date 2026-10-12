# Qwen Image2.1 Q4_K_M：下载、混合格式导入与首个完整GPU生成

2026-10-10，M4 Max64GB / macOS26.6.2。本轮开始新的完整目标：Q4_K_M
denoiser和text encoder、原BF16 VAE，encoder/DiT都要超过本地Unsloth对照，
继续GPU kernel与GPU/ANE并行，并比较BF16 resident及同内存预算streaming。
下面只是已核实的阶段进展，完整性能/ANE/streaming目标尚未完成。

## 模型和磁盘

经HF镜像下载两份固定revision的原GGUF，每份完整LFS SHA256验证后才
从`.part`发布，没有额外Hub缓存副本。工具为
`tools/validation/download_qwen21_gguf.py`，支持严格Range续传、大小/编码/
SHA检查和至少1GiB剩余空间。VAE和processor只读链接原BF16目录。

| component | repository / revision | bytes | SHA256 |
| --- | --- | ---: | --- |
| denoiser | unsloth/Qwen-Image-2.1-GGUF / 2c31ccd392b367a6637841a143813320a02dff55 | 4,199,565,024 | 631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe |
| text | unsloth/Qwen3-VL-8B-Instruct-GGUF / b93a7ee713758252c555be4210c00540df954dc2 | 5,027,785,568 | 108e7ff92b78eefd3db4741885104acba514255c11b617d3c7b197a5f46efe89 |

目录为 `models/unsloth-qwen21-q4-k-m`，receipt明确只证明assets，不证明
runtime或性能。原BF16模型保留，以便比较。只删除仓库build下已确认可
重建的28,171个`.o`，2,091,429,048 logical bytes（约1.95GiB）；未删模型/
adapter、现有library/CLI或用户缓存。对象可按原构建命令重建。

## 实际格式不是一个tensor type

原denoiser共有265个tensors：Q4_K163、Q5_K28、Q6_K1、Q8_0四个、F32
65、BF16四个。原text encoder399个：Q4_K217、Q6_K37、F32 145。
Q4_K_M是文件配方，不能只放行Q4_K或只凭文件名宣布支持。

新增显式mixed-K affine importer：Q4_K保留整数codes、展开32元素组的
scale/min；Q5_K整数codes扩展成Q8容器；Q6_K的16元素subscale无法直接
由现有MLX affine QMM消费，因此逐32元素组显式重编码到Q8。仅128bytes
栈上dense scratch，不展开整个dense模型。typed FP16/BF16 metadata及
Q4/Q5 ARM NEON搬运分别验证；Q6额外近似和系数舍入必须披露。

旧`pack_native_affine`和默认packed-bank路径仍只支持旧Q4_0/Q4_1/Q8_0；
新overload需显式K options，保留source-proof/owner/alias/cancel/finite/
ledger和事务发布检查。允许component过滤，但完整原文件仍由native
SourceLease验证。text只省去未消费的LM output head，并展开当前prompt
assembler需要的token embedding table；其余attention/FFN仍packed。

对7个真实source矩阵、两种typed recipe、M1/33/145，共42次实Metal
投影成功；每次全输出finite，前8个输出rows与独立原GGUF F32 decoder/
matmul对照。FP16最大relL2约.817%，BF16约4.263%，不外推成完整模型
latent误差或所有rows/输入保证。初始模型路径选FP16 I/O/coefficients，
VAE保持BF16。108个CPU scalar/SIMD numeric场景及非法span/alias/finite/
cancel checks通过，旧exhaustive affine与实Metal packed-bank回归通过。

## 首个完整GPU烟测，不当正式性能比较

新Session识别两份GGUF文件、映射官方denoiser/text命名，保留原BF16
入口。先接GPU base generation；GPU/ANE、编辑/LoRA的GGUF边界仍待接入。
示例为 `examples/requests/qwen-image-21-gguf-512-smoke.json`。

v1 Private native-only build exit0，503个runtime source inputs与当时tree
完全一致。同原BF16 VAE、512²、4steps、seed29、无LoRA/参考图，完整
请求正常生成PNG，所有latent finite：

| v1 native cold smoke | seconds |
| --- | ---: |
| text_encode（含导入/native source验证） | 4.920507 |
| denoise | 4.978405 |
| VAE decode | .538534 |
| complete request | 13.679689 |

100ms进程树证据完整：peak phys-footprint8,845,742,976bytes，system
swap-in/out和compression0；MLX logical peak10,993,179,268bytes不是同一
内存口径，不把两者相减归因或当全请求硬上限。图中狐狸/雪林可辨，但
4步不足以做质量验收，存在明显粗糙/颗粒细节，需40步及匹配BF16对照。
没有quiet-load资格，也没有新的普遍加速倍率。

v1 `encoder_runtime_precision`旧renderer仍标BF16，scope不准确；实际
源码和real-projection receipts是FP16。后续v2已修正GGUF encoder/graph
labels和retained-source revalidation，不改写v1原始JSON。

## Unsloth对照：修复版本，保留采样失败

只读本地 `../references/unsloth`。缓存的u13b9d92 sd-cli不能识别Qwen2.1，
首轮真实Unsloth脚本在加载时失败，无PNG，不作为速度分母。查其当前
installer后，使用原脚本在本仓库`.deps`安装其固定u1d02858 macOS-arm64
包；不改references或用户全局安装。Release/owned-install记录保留。

第二轮原Unsloth `scripts/sd_cpp_smoke.py`、相同两份GGUF/BF16 VAE、512²/
4steps/seed29/cfg1/fast，实Metal DiT成功生成PNG。其默认Apple policy
明确`clip_on_cpu=true`，text encoder为CPU，不能说双方encoder都用GPU。
原log中condition1.50s、sampling12.09s、VAE18.17s、generation31.77s，
脚本完整generate38.1s。这些不同边界不相加/相乘冒称严格matched倍率。

该wrapper的100ms进程树采样因短生命周期child报`unknown_child`，report
inconclusive而native exit0/PNG成功；不是模型失败，也不追认内存资格。
下一轮需在原Unsloth argv/策略下单独采样实际model child，保留预检和
wrapper成本，并筛选其max/FA/direct-conv路线，避免只选较慢默认作对照。
目前native冷text阶段4.92s并未超过其condition1.50s，不能宣布encoder
目标达到；需要对齐权重生命周期、冷/热/compute/load范围并实际优化。

## v2：40步、原BF16匹配GPU、retained fresh条件对照

同一v2 Private CLI/库，503个runtime inputs匹配当时源码，encoder标签
已修正。两route各一cold、两fresh warm，三个fox提示词只改natural/soft/
warm winter light，conditioning全miss，原source retention同开、实际warm
reuse=true，无LoRA/参考图/ANE或时间复用；所有40步finite完成。

| route | cold request s | 两warm request median s | warm text_encode s | warm denoise s | tree peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed Q4_K_M | 52.142410 | 44.525328 | .229649 | 43.731369 | 16,035,176,656 |
| 原BF16 | 48.001991 | 43.343858 | .598625 | 42.072556 | 36,863,172,152 |

当前GPU-only GGUF整请求比匹配BF16慢2.73%，DiT慢3.94%，不是量化就
自动更快。两过程tree peak比为.434992，名义低56.5%；这不是给未来
BF16 streaming配置的精确硬上限。GF cold text4.767760s；BF16为2.881597s。
BF16两个warm text分别.910111/.287139s，波动很大，不把中位差额全部
归因于quant kernel。更多顺序/提示词和独立load观察仍需补齐。

两份100ms进程树report独立重放complete。GF system swap-in65,536bytes、
swap-out0；BF16 swap0，但system compression8,045,346,816/decompression
6,503,858,176bytes。不得声称全窗口无内存压力或将系统计数归因特定
进程/矩阵。已查看soft-light case的两张whole512：姿态/脸、毛色、雪林
构图和细节非常接近，有少量毛纹/背景差异；仅此明确场景，不代替更多
seed/scene/最终图像资格或latent等价。

Unsloth v3改采原args builder生成的实际model child，不含短生命周期
version/help子进程，并开启其max（diffusion-fa/conv-direct），native exit0/
PNG成功且memory complete。仍是4steps：condition1.49、sampling11.90、
VAE18.14、generation31.53s，tree peak11,026,178,424bytes，system swap0。
未改变原Apple clip-on-cpu策略，未修改reference代码；不把它与native
40steps直接计算速度比，不将两种冷/热生命周期拼成已实现encoder目标。

完整source/PNG/hash/phase/retention/memory与未完成要求见
[阶段机器记录](../design/validation/qwen21-q4km-first-native-20261010.json)。

## 完整目标与后续

接下来测40步、更多提示词和retained fresh-condition requests，分别记
encoder load/compute与DiT；接mixed-K Private/Public通道分工和GPU kernel，
保留完整packed GPU fallback；实现/验证Qwen BF16同预算streaming并比较
实际峰值与整请求，而不是把BF16仅换成较小尺寸/更少步骤。

只读参考Splash `runtime/model/WeightStore.hpp`中的源mapping与no-copy
section owner边界；现有bank是显式重布局复制，不冒称已经实现其零拷贝。
原immutable source、finite/shape/内存/late-reader安全边界不因允许近似而
放宽。所有模型、binary、PNG及raw logs不入Git；本轮完整目标保持active。

原始依据为 `outputs/qwen21-q4km-*-20261010.*`、
`outputs/qwen21-q4km-native-v1-smoke-20261010/` 和
`outputs/qwen21-q4km-unsloth-v{1,2}-smoke-20261010/`。失败记录不覆盖。

收尾时确认全部owned模型/编译/测试jobs已终止，清理两个新build的436个
可重建`.o`、34,495,520 logical bytes及两个空module-cache；复查objects0。
library/CLI、probe、模型和全部失败/有效证据均保留。最终5项host tests和
1项旧packed-bank实Metal回归通过，无skip；不是全部模型/产品资格。
