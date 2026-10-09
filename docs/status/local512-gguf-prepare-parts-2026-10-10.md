# GGUF / ConvRot 提前解码前置诊断：完整 finite scan 值得融合

2026-10-10，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[有界dense window](convrot-register-dense-window-2026-10-06.md)与
[共享packed-word核](local512-affine-shared-word-2026-10-10.md)。在完成
[Qwen base三路](local512-qwen-frozen-base-2026-10-10.md)并提交后，本轮
继续检查真正ahead-decode的准备成本；没有下载、改写模型、dense sidecar
或更改模型默认路线。完整Z/Qwen、LoRA、编辑/encoder和异构加速目标仍active。

## 研究工具，不是新的异步 decoder

`gpu_weight_consumer_probe` 新增 `gguf-prepare|convrot-prepare` 模式，
只接受M占位1、无reuse选项，非法参数在读source之前拒绝。它交替测
原 `AffineDenseWindow::prepare` 与带额外eval边界的归因镜像，后者分别
记录cache hint/clear、dequantize+eval、finite graph+eval、synchronize、
publication/cache restore。时间全是host spans，不是物理GPU timestamps。

镜像保留原FP16/BF16 metadata、typed dequantize、finite要求、fresh-bin
capacity upper、先ledger admission后分配，以及Data-held storage claim。
completed finite才发布；没有去掉finite scan或把lazy输出当ready。extra
eval改变调度，归因总时间比原路线多约15–16%，不是性能优化候选，更
不能把各part median相加/相减当原路线的唯一瓶颈占比或预计提速。

## 三个真实本地权重，同binary串行交替

只读layer0 GGUF Q4 gate、GGUF Q8 down、ConvRot gate。每个dense目标
78,643,200bytes（75MiB），两dense容量上界157,286,400bytes。读/packing
不计时；每臂3warmup、15hot原始样本完整保留，循环交替control/parts。
没有GEMM、完整consumer窗口、模型/LoRA、图像或ANE计时。

| 原source / dtype | original prepare ms | attributed total ms | dequantize+eval ms | finite graph+eval ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF Q4 gate / FP16 | 3.918625 | 4.541167 | 1.893875 | 2.601208 |
| GGUF Q8 down / FP16 | 3.924708 | 4.532083 | 1.931042 | 2.575542 |
| ConvRot gate / legacy BF16 scale | 3.921084 | 4.543500 | 1.917333 | 2.572959 |

cache hint/clear中位约.0040–.0044ms，synchronize约.0200–.0219ms，
publication/cache restore约.0010–.0011ms。它们在这三个同步诊断里很小，
不据此取消模型route的ownership/readiness fences，或声称异步worker中
等待同样小。ConvRot仍在rotated basis，保留实际legacy BF16 stored scale；
不是inverse rotation或FP32原scale重建。GGUF仅MLX native affine
Q4_0/Q8_0 importer，不外推raw Q4_K/Q6_K。

三个real-source镜像对原prepare的typed weight coefficients逐字节相同；
这是准备系数，不是完整模型输出/图像逐字节等价。continuous host
observer无errors、thread joined，最大gap约.167/.128/.130s，binary
before/after unchanged；qualification始终false，没有设备独占或physical
overlap资格。原始payload/codes哈希和全部样本见机器记录及raw receipts。

## 对下一步优化的具体影响

完整finite graph是值得处理的大项，不能仅优化独立解码或删检查：下一步
优先研究**typed解码时同时产生finite status，再归约小status buffer**，
消除单独展开完整dense `isfinite`中间结果的可能成本。必须保留原dtype
舍入、FP16/BF16 overflow/nonfinite拒绝、source身份和bounded backing，
用真实whole prepare重新比较后才决定是否接入。

这还是一个基于测量的候选方向，不是已经实现了融合核，也不证明降低
prepare后512²能盈利。此前Q8 gate/down packed与dense差距只有约
.176/.087ms，R1逐层重解码仍可能亏损；两个矩阵槽轮转30层不是R30
复用。下一步还须真正current/next层异步准备、消费者ready等待、escaped
readers/bank generation隔离、ledger同时覆盖in-flight目标，以及完整
decode+GEMM窗口/实模型。不能用这些parts数字追认ahead-decode完成。

## 回归、编译与清理

最终1项opt-in实Metal unit通过无skip：原12个window dtype/group cases
及新增12个归因cases，含Q4/Q8、FP16/BF16、g32/64/128、精确系数、lazy
reader claim、nonfinite拒绝、admission/rollback；另10个非法CLI组合在
source access之前拒绝。三个real-source组件均exit0、dense/ledger claim
最终0；实际用于计时的保留binary也直接重跑上述24个typed cases通过，
重复运行不重复计数。不是整仓suite、完整模型/内存资格或新增端到端加速。

初次v1 build误用 `streaming::StorageLease` namespace，严格编译失败；
改为既有 `tc::StorageLease`，v2 build exit0，失败log保留。只编译
standalone probe，absolute rpath链接保留的generation-phases-v1 Private
库，无adjacent dylib/loaded-image证明；没有新production native源码、
library rebuild或App发行。probe SHA：

```text
3ea28c463c1f403aca97caa5bef53cf729ed28584c2ee5ece2fe188ae997f45a
```

全部owned builds/tests/components已terminal。临时test目录自动清理，
两个owned probe build均无`.o`/module-cache，保留小binary、有效/失败logs
和原始receipts，不删除原compiled模型、driver/用户cache或外部进程。
完整[机器记录](../design/validation/local512-gguf-prepare-parts-20261010.json)
保留源码/binary/source/evidence哈希、各样本与未完成边界。
