# 26 · Lossless fused affine import 与直接浮点转换

[目录](README.md) · [25](25-verified-encoder-metadata-cache.md)

本阶段完成Q4_0/Q4_1/Q8_0的CPU一次遍历packing、ARM NEON codes布局和direct-read
BF16→FP16 importer alias。真实不保留bank的Q4热请求约从6.716s降到5.058s，但仍未
满足最快BF16≤20%目标；不能把CPU/kernel改进称为完整加速交付，完整目标继续active。

## 1. 同字节、同精度，不增加量化

`pack_native_affine_all`一次读取每个源block，同时写codes、FP16 scales和biases，
无I/O/heap/MLX/额外dense或packed模型；固定小寄存器/栈状态。
原`pack_native_affine`三遍标量API保持未改，作为独立legacy oracle和诊断控制。

- Q8 codes保持signed byte XOR0x80；NEON两次16-byte load/store。
- Q4 codes按原GGUF low/high nibble顺序重排，不换group/row/column顺序。
- Q4_1 scale/bias原bit pattern保留；Q4_0/Q8_0 bias是`half(scale * -2^3/-2^7)`。
  新实现按整数half exponent/尾数处理，subnormal以2^-24单位正规化，±0保留符号变化。
  representable乘2幂严格exact；overflow失败，不flush/clamp，不修改舍入模式。
- 所有geometry/capacity/source-target及target-target overlap在写前检查；最多256个
  blocks（Q8为8.5KiB）间检查取消，返回总written bytes。失败目标可partial，绝不Ready。

packed bank仍有一个原1MiB read buffer和一次admit的最终fields；全部完成/source
generation稳定后才事务性bind。floating BF16字段直接pread到已admit最终F16 backing，
调用24已穷举验证的`bf16_to_fp16_inplace`，不额外常驻raw BF16副本。
其他float格式仍走原buffered decoder。它复现旧importer表示，不宣称源BF16计算精度。

CPU-direct实验默认fused；普通/MLX旧入口不变。明确同库控制：

```text
--affine-pack legacy|fused
TURBOCIDER_Z_GGUF_AFFINE_PACK=legacy|fused
```

无CPU-direct或不认识的值拒绝。policy/backend/float import recipe进入plan digest和
实际receipt，不能在timing/memory间换strategy或把scalar fallback标为ARM NEON。
new `affine_decode_seconds`与`float_decode_seconds`分账，GPU计算/数学recipe不变。

## 2. 验证

最终experimental library：
`d73c681050c07426040000ff44fb91849cd5c12083b89d03ff973871c407ec6c`。

- CPU：每type全部65,536 scale patterns，Q4_1另全部65,536 bias patterns；legacy
  scalar与fused scalar/NEON bytes、finite/subnormal/±0/overflow判定相同。
  36组rows1/7/33/257×K32/64/256、非对齐source/target、guards、返回bytes和capacity/
  alias/type/cancel负例通过。相同CPU component ASan/UBSan通过。
- 实际Metal packed bank：legacy/fused与原MLX importer所有12个字段逐byte exact、
  GPU projection exact；managed upper/read-once、last-reader/escaped view claims、
  cancel/floor/nonfinite/source replacement和释放归零通过；strategy有独立digest。
- 最终host/parser/measurement/screen/storage矩阵32项通过；Metal/parser/普通generate
  gates矩阵11项通过。原CPU格式矩阵等13项通过、13可选oracle/fixture skip未计为通过。
- Q4（Q8 bounded encoder）与Q8（原encoder）真实512²/portrait/seed42/4steps各136个
  block+最终latent N1通过；candidate所有保存端点与原对应legacy控制exact。
  中间版fused与最终direct-float版分别保留不同binary回执，不混池测速。
- Q4 import tensor180取消status2/无PNG，同engine新bank retry status0，全部端点exact。
  没有复用partial bank、修改源模型或放宽N1/媒体门槛。

[数值、取消与不同binary诊断](validation/fused-affine-diagnostics-20261002.json) ·
[CPU/同库对照回执](validation/fused-affine-progress-20261002.json)。仍不是48-case媒体资格。

## 3. 原瓶颈确实改善，但整体BF16硬门仍失败

同library、相同GPU tuning、34valid tokens/combined1088，每arm4个cache-hit热样本
+1保留warmup，timing无observer；memory各自独立进程10ms采样。

有界Qwen3 Q8 encoder控制（source/components一致，metadata cache均on）：

| 不保留packed bank、VAE前verified释放 | warm wall median |
| --- | ---: |
| legacy三遍packing +buffered float | 6.715956s |
| fused NEON +direct float alias | 5.058303s |

ratio=.753177，少24.68%时间，PNG exact。最终热样本的import约.665s，affine约.344s、
float约.082s，source read约.222s；legacy对应affine约1.980s、float约.129s。
额外一次0.66s导入仍不是22“保留bank/零重导入”热路径，不能嫁接4.4s成绩。
这是一轮local screen，不是正式ABBA/CI/p90/cold认证。

原encoder/VAE/tokenizer内容绑定相同的真正BF16对照：

| arm | warm median | 观测process peak bytes |
| --- | ---: | ---: |
| 纯GPU BF16 | 3.732799s | 23,012,206,288 |
| Q4 fused不保留bank | 5.035624s | 9,399,589,464 |

wall ratio **1.349021**，34.90%较慢，≤20%失败；memory ratio .408461，内存下降不能
替代速度。对应旧16GB retained候选仍保留，不被本阶段更慢的release路线替代。
[BF16负screen](validation/fused-affine-q4-bf16-screen-20261002.json)。

## 4. 分预算与剩余目标

fused Q4 +已有bounded Qwen3 Q8 encoder，独立full-request peak
**7,796,216,344 bytes**、managed peak4,850,649,728bytes、swapout0，最大gap16.03ms：

- 6/8 GB及GiB扣10%余量均不fit，不报告支持。
- 10/16 GB及GiB仅observed fit；没有whole-request hard cap或真实小容量设备资格。
- encoder与纯BF16对照不同，不混入上一节同组件screen或继承其质量资格。

[GB screen](validation/fused-affine-q4-budget-gb-screen-20261002.json) ·
[GiB screen](validation/fused-affine-q4-budget-gib-screen-20261002.json)。

后续应比较同预算CPU fused供给与raw-GPU/direct packed kernels、较低cache hints、
投影tiles/生命周期与实际GPU调度。不能靠保留完整dense模型、隐瞒VAE/driver峰值或只
测GEMM来达标。本cell raw仍是更低6/8GB的候选，16GB仍有已测较快retained profile；
每个预算要取得完整请求速度/质量/内存交集，不能只选择最容易通过的一项。
更多格式/组件/IQ、48-case媒体、formal性能/envelope闭包、ConvRot旋转域/非W8A8
低内存、GPU/ANE W8A8和M5静态接口继续保留在原完整目标内，尚未完成。
