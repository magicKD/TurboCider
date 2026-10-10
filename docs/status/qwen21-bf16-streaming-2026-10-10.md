# Qwen 原始 BF16 双阶段 streaming 与 GGUF 同内存对照

2026-10-10，M4 Max64GB/macOS26.6.2，接续
[GGUF encoder 并行导入](qwen21-q4km-encoder-import-2026-10-10.md)。
本轮实现此前缺失的 encoder **和** DiT 原始 BF16 有界 streaming，继续
完整512²/40steps同版本对照；完整目标仍 active，尤其 encoder 对
Unsloth 的冷/热匹配与 Public ANE 全覆盖不能用本项代替。

## 已实现的真实数据路径

显式 `TURBOCIDER_QWEN21_BF16_STREAMING=1`、`residency=component_staged`，
只允许512² base GPU生成/实际warmup、2..40steps，原BF16 encoder、DiT、
VAE。不改steps/分辨率/训练权重，不重编码BF16为Q8，不跳层，不使用
DBC/FFN时间复用、LoRA、编辑或ANE来替换这个streaming对照。
这是private diagnostic adapter，尚不是public preset或整请求硬RAM上限。

held-fd完整native SHA和generation验证后解析safetensors；验证全部BF16
目录、decoded JSON key唯一性、shape/offset/连续payload覆盖。base language
只省去未消费的LM head/visual tower，保留全部36层、embedding与norm。
DiT保留原9个fixed fields和32层，每层8个原tensor。

复用 `MlxWeightPager`：prefix提取为resident fields，suffix两份预分配
共享buffer，由一个CPU worker直接pread原BF16，下一层提前fill。
`SlotSafetyTracker`绑定request/content generation、pass/group/slot，
同一个物理槽在全部GPU读者完成后才退役/再次写入。所有job在异常/
取消销毁前join，设备drain失败不释放潜在live backing。

DiT的compiled closure原本捕获`weights_`，现streamed路径将当前层原数组
作为动态arguments绑定；prefill和decode/KV variants分别保留，完整GPU
block算术不变。encoder保留原F32 norm/attention、BF16 I/O和mRoPE顺序，
也在每层同步完成后retire。结束encoder后释放bank，再开始DiT；全部DiT
source/graph owners完成后释放，再加载同一BF16 VAE。

每个source backing使用fresh MLX bins并检查actual size不超过16KiB对齐
计划上界，避免小field误借大缓存块。core source按每个实际field增长
重新做物理准入，而不是一次预测11GiB optional增长：原4GiB system
reserve、inactive半数且最多8GiB信用、process/MLX限制均不放宽。
总managed weight capacity仍必须小于配置预算；所有准入/finite/source/
cancel失败仍fail closed。阶段加autorelease scope，但不将它单独称为
本轮性能/内存收益的因果来源。

本机同预算候选为encoder/DiT各prefix28、两个suffix slots、managed
weight预算13GiB；不是整请求13GiB保证。配置变量：

```text
TURBOCIDER_QWEN21_BF16_STREAMING=1
TURBOCIDER_QWEN21_BF16_STREAM_ENCODER_PREFIX=28
TURBOCIDER_QWEN21_BF16_STREAM_DIT_PREFIX=28
TURBOCIDER_QWEN21_BF16_STREAM_WEIGHT_GIB=13
```

## 数值/所有权与完整4步预检

实Metal小模型4层、prefix0/1/2、三个完整KV/Euler passes：encoder hidden、
compiled DiT输出和最终Euler latent逐元素相同；覆盖两个槽跨层/跨pass
复用、预算floor、取消、source truncate、source closure与顺序负例。
旧GGUF packed-bank ownership/mixed-K/gathered embedding回归同库通过，
host receipts拒绝伪造/漏层/少fill/缺reader/source/overflow预算。

新v4库512²/4steps/seed29、三个fresh fox prompt，retention/prefix-KV的
跨请求命中关闭。resident control显式保留原encoder source；streaming
只保留每阶段bounded weights，warm仍需要重载prefix，但 native同进程
generation-bound SHA proof确实命中。两完整process tree报告均complete：

| route | warm text s | warm denoise s | warm request s | tree peak bytes |
| --- | ---: | ---: | ---: | ---: |
| BF16 stream prefix28 | 1.344785 | 5.573748 | 7.492400 | 14,765,353,040 |
| BF16 resident | .575297 | 4.582022 | 5.846706 | 37,216,607,752 |

三张PNG以及每请求text/initial/latents/pixels的12份safetensor SHA均相同，
不是只比较极少输出rows或拿另一个seed的视觉接近代替完整消费验收。
4步仅数值/功能/内存预检，不当40步质量或普遍性能倍率。

stream system swap-in/out/compression0，decompression32,768bytes；resident
swap0，但system compression7,042,334,720/decompression6,746,308,608bytes。
不能把系统counter归因单个进程/矩阵，也不追认quiet-load资格。

## 40步四路同库完整对照

原两份Q4_K_M GGUF/原BF16 VAE、原BF16两component、同512²/40steps/
seed29和三个fresh prompt。每臂独立process、一cold两warm，顺序
BF16-stream→GGUF-GPU→GGUF-ANE5120/GPU7168→BF16-resident。
GGUF使用8 CPU import workers、原source retention；shared-down off、
Private prefetch/ahead/converted-code cache off，完整1280 FFN调用验真。

stream的`denoise`计时包含该阶段首次source验证、prefix加载与slot setup，
普通resident/GGUF原`denoise`在source准备后开始；所以整request是首要
分母，另记录实际prefill/decode完整step spans，不从source timings相减
拼成新的因果kernel倍率。实际内存是否与GGUF混合在±10%范围内由
同window process-tree peak核对，不是把managed预算当物理内存结果。

四臂全部terminal/finite完成；各两warm完整样本的中位数如下：

| route | warm text s | warm denoise s | warm request s | tree peak bytes |
| --- | ---: | ---: | ---: | ---: |
| BF16 stream prefix28 | 1.311323 | 43.468259 | 45.344398 | 14,753,081,448 |
| GGUF complete GPU | .225473 | 43.775951 | 44.605709 | 15,228,674,400 |
| GGUF ANE5120/GPU7168 | .232362 | 36.169814 | 36.989884 | 15,359,959,488 |
| BF16 resident | .652800 | 42.085535 | 43.448446 | 37,266,890,248 |

stream/GGUF混合峰值比.9604896，差约3.95%，满足本窗口±10%的实际
内存相近条件。与stream相比：GGUF GPU整请求名义少1.6291%，只有很小
优势，不能推广为量化kernel显著盈利；GGUF GPU/ANE少18.4246%，约1.22586×。
同库混合对GGUF完整GPU少17.0737%，对BF16 resident少14.8649%。
BF16 stream对resident整请求多4.3637%，但明显降低峰值；核心计算和
文件生命周期是不同收益来源，不把它们混成“量化自动更快”。

BF16 stream的实际完整denoise step spans warm中位数42.550899s，另有
prefix/source setup，因此上表43.468259s不与其他source-excluded DiT计时
直接计算kernel倍率。warm encoder1.31s仍包含每请求重载prefix；GGUF/
resident保留原encoder，source生命周期差异是同内存策略的实际成本。

混合每request实际1280成功Private predictions/channel blocks、phase32/
1248，fallback/failure/retry0；三请求各完成40步，不使用时间复用。
BF16 stream每request真实36 encoder层/8 suffix fills与1280 DiT层/
160 suffix fills/160 last-reader fences。原denoiser logical streamed reads
69,793,300,480bytes/request，resident prefix/fixed另12,485,416,960bytes；
warm native source proof确实generation-cache命中，不额外做全文件SHA。
这不是物理SSD bytes、GPU/IO物理overlap或零I/O证明。

40步三个BF16 stream/resident PNG全SHA相同；六张GGUF GPU/混合PNG也
与此前kernel-off对应control全SHA相同。本窗口保留原全图而非只挑最好
phase；质量范围仍是这三个同scene光照变化/同seed，不代替所有场景。

四份process-tree证据complete，system swap-in/out0。前三臂compression0；
resident system compression10,485,235,712/decompression9,939,419,136bytes。
stream/GGUF-GPU/GGUF-ANE的system decompression分别393,216/1,425,408/
425,984bytes，不能归因单个进程或称全窗口无内存压力/quiet-load。

最终9项native/host unit tests与独立PyTorch conditioning10场景通过、无
skip；508个runtime source inputs与最后v4库匹配。重新查看soft-light
BF16 stream/混合整图，狐狸姿态/构图很近、局部毛纹和背景有小差异。
这不将不同精度、不同runtime的整图一概称为bit等价。

[机器记录](../design/validation/qwen21-bf16-streaming-20261010.json)绑定同库
source/图像/完整样本与内存hash、4步12份tensor parity、失败与剩余目标。

## 保留失败与完整边界

v1小模型prefix2只剩一个suffix却声明两个slots，layout拒绝；改为4层
fixture验证真正双槽prefix0/1/2，没有放宽layout。第一次runner误用
`dump`而非`dump_tensors`，在模型前失败，原记录保留。随后v2/v3库的
prefix24窗口第一个cold生成完成，但warm的DiT按整个11GiB增长预测被
原guard拒绝；完整窗口inconclusive，不作为成功热态样本。

组件级autorelease/精确bins没有单独解决后一拒绝。最终按actual field
增量观察和准入，通过连续请求；未增加inactive信用或降低system reserve。
新路径早期误将普通生成标为prepared的原raw保留，后续v3起修复结果标记，
不改写旧输出使它看起来经过完整receipt验证。

默认路线、参考工程、模型/adapter未改。GGUF冷encoder此前2.70–2.96s
仍慢于新Unsloth cold condition1.48s，且source验证/放置/生命周期不同；
继续校验与导入重叠、持久会话fresh prompt、GPU kernel/ANE策略/质量覆盖。
本项不能宣告完整encoder/DiT/public目标完成。
