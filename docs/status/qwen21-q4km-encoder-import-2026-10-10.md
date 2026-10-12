# Qwen Q4_K_M：冷导入加速、局部 embedding 与新的 Unsloth 对照

2026-10-10，M4 Max 64GB/macOS26.6.2。接续
[GPU/ANE 实际通道分工](qwen21-q4km-hybrid-2026-10-10.md)。完整目标仍 active：
不能用 DiT 优势代替 encoder 冷/热匹配和同内存预算 BF16 streaming。

## 本轮实际改动

`Weights::embedding_rows` 在 GPU 上先 gather codes/scales/biases，再仅解码
提示词行，保持相同 affine 数值/FP16 输出。Qwen prompt assembly 共用原
视觉替换、positions、DeepStack、retained indices 和 image slots；BF16
入口与词表 ID 检查保留。原 `TextEncoder::encode` 也能消费 packed embedding。
不再在 GGUF load 时展开整个词表，不额外引入 embedding 的量化近似。

这参考了只读 Splash `runtime/metal/kernels/shared/embedding.metal` 的
packed 消费思路；不是搬用其 group64/BF16、tiled N256 权重 ABI，也不
声称已有 Splash 式全模型 mmap/no-copy。当前原 GGUF 仍需显式 affine
重布局和 Q6_K group32 重编码，完整原文件 native SHA 验证保持。

mixed-K bank 增加 1..8 有界 CPU row workers，默认1。只在同一个已读取
chunk 内对互不重叠的源/目标 rows 并行：始终一个1MiB读取缓冲区；
MLX 分配、source lease 与输出发布仍由 owner 完成。全部工作线程 drain
后才复用/释放 buffer，异常/取消同样 drain 后事务失败，不发布半个 bank。
实际 worker 数写入 bank metrics 和 plan digest；legacy 默认计划不变。

M4 Max 显式实验可设 `TURBOCIDER_QWEN21_GGUF_DECODE_WORKERS=8`，不要
把当前机器最优线程数推广到所有硬件。已有 Private ANE5120/GPU7168
FFN 路线不变，shared down-kernel 仍默认关闭，Public qualification 未完成。

## 同 binary 正反序：完整 encoder 导入组件

每次独立新 process，完整5,027,785,568 bytes原 encoder 首次 native SHA，
仅过滤未消费的 LM output head。顺序1-packed、1-dense、4-packed、8-packed
及其反序；dense 是同 bank/库的旧整表展开，非重编码成新模型。
下表是各两次中位数，13个真实 token rows；不是 encoder layers 耗时。

| 导入/embedding | native SHA s | bank load s | decode wall s | 组件总 s |
| --- | ---: | ---: | ---: | ---: |
| 1 worker / packed | 1.663820 | 2.075948 | 1.853315 | 3.874572 |
| 1 worker / dense | 1.658773 | 2.075348 | 1.853523 | 3.869106 |
| 4 workers / packed | 1.659613 | .849034 | .615158 | 2.617613 |
| 8 workers / packed | 1.658768 | .677273 | .414167 | 2.447297 |

8-worker bank load 名义约3.07×，组件总耗时低36.84%；SHA 没有被省略。
先 gather 并未在这里显著缩短总时间：其首次 kernel setup 与旧整表解码
约同量级，主要确定收益是内存。相同 checkpoint 最终 weight logical bytes
5,221,410,816 vs dense6,077,114,368，少855,703,552 bytes。组件 MLX peak
5,222,459,392 vs6,466,070,580；不可冒充整请求物理峰值。
八份100ms进程树记录 complete；固定读缓冲、source SHA 和 plan 均验证。

## 完整40步：仍约17%混合收益，冷 encoder 变快

同一新库/CLI、同模型与BF16 VAE、512²/40steps/seed29；每臂一cold两
fresh warm，conditioning 全 miss，admitted original encoder source retention。
当前 window 顺序5120混合→GPU，均8 workers、shared-down off。

| route | cold text s | cold request s | warm text s | warm DiT s | warm request s | tree peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANE5120 + GPU7168 | 2.964763 | 42.365164 | .234375 | 36.176235 | 36.981695 | 15,674,122,688 |
| complete packed GPU | 2.695782 | 49.255209 | .233477 | 43.789452 | 44.619617 | 15,228,477,720 |

混合 warm 整请求少17.1179%，约1.20653×；DiT 少17.3860%。与前一
NEON4步窗口的 cold text4.18–4.23s 方向一致，但不是同窗口因果拆分，
不能把完整差额全部归因单项改动。稳态 DiT 没有额外宣称新倍率。

每混合 request1280成功 Private predictions/channel blocks，prefill32、
decode1248；三请求累计1280/2560/3840，actual async/GPU IOSurface/FP16
join 匹配，fallback/failures/retries0、headroom1。两臂进程树证据 complete，
system swap-in/out、compression0；decompression分别4,603,904/311,296 bytes。
不是物理 GPU/ANE overlap、硬RAM上限或 quiet-load 资格。

六张整图 PNG 与前一 kernel-off reverse window 对应 GPU/5120、三个提示词
全部 SHA 相同：本轮导入/embedding改动没有改变这些实际输出。重新查看
soft-light GPU/ANE整图，姿态/构图很近、细节少量差别；不替代多scene/
seed、原BF16质量或全部编辑/LoRA验收。

## Unsloth：更强的新冷 encoder 反例必须保留

只读本地 reference 的原 `build_sd_cpp_command`/`native_speed_flags("max")`，
同 pinned u1d02858 sd-cli、两份 GGUF、BF16 VAE、natural-light原提示词、
512²/40steps/seed29/cfg1、12 threads、默认 Apple clip-on-cpu。
直接采样实际model child，不含version/help短子进程，不改参考源码/缓存。

本轮实际 condition **1.48s**，不能继续选择旧3.37s分母宣告冷 encoder
胜出。native cold text2.70–2.96s仍较慢，包含独立全文件 SHA；其约1.66s
目前已超过对方整个condition。warm native .23s也不能与cold1.48s拼成
已资格的encoder倍率。本次sampling108.32s、generation127.97s，实际
model child采样/验证wall128.454019s；tree peak11,026,882,936bytes，memory
complete，system swap-in/out/compression0、decompression9,846,784bytes。
本轮native cold混合request42.365164s、DiT36.580159s明显有利，但DiT源
加载在denoise前，对方sampling包含首次lazy权重加载，故不直接宣布严格
普遍3×；native process更高峰值也必须披露。reference本次是单cold，未
匹配双方持久会话warm fresh prompts。两套runtime同seed不保证同初始
noise；reference整图是另一种狐狸/雪林构图，不当latent/图像等价对照。

## 后续按收益排序，不删除未完成要求

1. encoder：校验/导入重叠、减少重复读取；拆清load/compute，并用
   Unsloth持久会话 fresh prompt 对齐warm生命周期。保留完整SHA、取消、
   named-path/held-fd generation checks；不靠删验证或cache命中造收益。
2. GPU：继续研究 Splash packed-MPP/量化group sums、融合gate/up/激活/
   residual与正确的g32系数消费。已有shared-word down仅约1%组件/窗口
   信号、部分shape负收益；真正有界ahead-decode必须带完整消费者测试，
   不能将全矩阵每步展开当默认优化。
3. GPU/ANE：对实际prefill/decode shape分别选择通道数和Public/Private，
   保留完整packed GPU恢复、同source staging、有限bank和reuse fence。
   本地Splash是GPU实现参考，不声称已从它获取ANE完整实现。
4. BF16同预算streaming：需要encoder和DiT两段，而不是只stream DiT、
   留整份大encoder。复用 `MlxWeightPager`/SourceLease/bounded slot tickets；
   encoder已有dynamic source graph inputs可用，DiT当前compiled closure
   捕获 `weights_`，须先改成正确dynamic per-layer binding，eval/drain后才
   退役ticket。完整40步、同分辨率和实际近15–16GB峰值对比尚未实现。

新增component/import、原Unsloth命令与完整生成runner；结果均 diagnostic、
qualification=false。库503个runtime source inputs匹配当前tree。实Metal
packed bank回归含mixed-K两dtype、1/2/3/4/8 workers逐字节一致、Q4/Q5/Q6
token gather与整表decode一致、nonfinite/cancel/floor/source/escaped reader
负例通过；旧exhaustive affine和108 typed K CPU场景通过。独立PyTorch
conditioning10场景通过（0/1/2/10参考、F32/BF16），layout相关误差0；
四个host receipt/parser测试通过，无skip。

两次conditioning runner先因隔离环境缺torch/MLX失败，原失败log保留；
只在既有`.deps/qwen21-oracle-env`补同版本MLX0.32.0后第三次通过，未
修改全局Python或native运行库。模型/PNG/raw证据不入Git。

[机器记录](../design/validation/qwen21-q4km-encoder-import-20261010.json)绑定
component正反序、完整生成、新reference、source/library/PNG hashes和未完成项。
