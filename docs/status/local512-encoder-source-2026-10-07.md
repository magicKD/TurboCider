# 512²：有界 encoder 源复用与独立 DiT/encoder 分区

2026-10-07 UTC；目录后缀 `20261008` 使用 Asia/Singapore。
接续 [executor retention](local512-encoder-retention-2026-10-07.md)。只读
本地Qwen Image 2.1、Viggle adapter与PNG；没有下载、改写模型或额外磁盘
权重副本。完整Z/Qwen base/LoRA、1–2参考图和盈利/质量目标仍 active。

## 实现

显式 `TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS=1` 可在resident512²
Session中保留一个已有encoder `Weights` owner；不是新的dense复制、
精度转换或sidecar。对GPU与ANE采用相同政策，普通请求默认仍释放。
完整数组保留前materialize；同一物理source generation使ANE scale cache
可以复用，而不仅是复用program。当前payload17,534,247,392 bytes。

源逻辑upper20GiB，physical<48GiB拒绝；复用前与首次保留前/后重新做
memory admission，保留4GiB system reserve、额外4GiB workspace allowance
及尚未驻留的模型/adapter/runtime增长。不是整个进程RAM cap。文件canonical
路径/dev/inode/size/mtime/ctime变化会释放source/executor并使两种conditioning
cache一起失效；编码结束重新检查generation。ctime覆盖普通same-size、
保留mtime的写入。这不是full payload signature或immutable file lease。

关闭源开关、错误与unload释放owner；JSON记录enabled/reused/retained、
source/retained bytes、累计load次数与decline reason。conditioning cache
hit不假称新encoder执行或新source reuse。`encoder_runtime_reuse` 的
weight-residency标签按事实区分request-local与resident source。

新增 `TURBOCIDER_QWEN21_ENCODER_ANE_CHANNELS` 固定override。HybridFfn/
factory接收immutable override，不临时修改process environment；独立验证
512对齐/小于完整width，不能覆盖calibration或使用auto。global/DiT5120
与encoder3072能并存；默认global选路不变。Public的positive channel
override仍拒绝，显式0保持其row ABI。

## 冷请求准入的负结果与初始化调整

v1真实session首先拒绝source保留：rawfree约396MiB、inactive35GiB，
后续cold DiT尚未驻留。v2虽然提前加载DiT/VAE，LoRA尚未绑定，cold源
仍因future adapter growth拒绝。v3独立encoder screen执行完整，但组合
cold arm又因保守2GiB future-runtime forecast拒绝；partial screen保留，
不挑warm请求追认成完整比较。

没有降低reserve或inactive-credit门槛。最终source profile提前加载并
绑定本来就要驻留的DiT/VAE/adapter；固定分区runtime先创建，准入观察
其真实arena。auto calibration仍等模型workload。普通request-local路径
保留原加载顺序。抽出的transformer准备函数保留原pinned/alternate adapter
identity、独立low-rank、alpha/FP32 rank边界、QKV/source失效与227投影检查。

另一次v2测试启动发生在dylib尚未产出前，loader失败、无model执行；
原log与之后retry分开保留，不计为通过。所有最终benchmark/build/硬件
job串行，无external process signal。

## 公平encoder源复用对照

工具扩展GPU、GPU+source、ANE executor-only、ANE+source；同库、六步
真实Viggle、512²、seed29、三个不同prompt，每个condition cache miss。
第一个请求单列，两个fresh warm请求全保留。下面不是strict load或
physical-engine资格，没有把GPU source cache的收益算作ANE独有收益。

| screen / arm | fresh request median s | 三次text s |
| --- | ---: | --- |
| v1 GPU | 10.762165 | .828 / 1.085 / .880 |
| v1 GPU+source | 10.747053 | .950 / 1.081 / .731 |
| v1 ANE executor-only | 11.245663 | 3.250 / 1.623 / 1.272 |
| v1 ANE+source | 10.854059 | 1.396 / 1.066 / .835 |
| v3 GPU | 10.793740 | 1.561 / 1.212 / .853 |
| v3 GPU+source | 10.974645 | 2.508 / 1.386 / .674 |
| v3 ANE executor-only | 11.332249 | 2.040 / 1.744 / 1.291 |
| v3 ANE+source | 11.144366 | 3.061 / 1.575 / .832 |

源cache每臂只load1次，新request确实reuse；ANE source-cache hits累计
0/108/216，不再是前轮的全miss。GPU有/无source对应PNG exact，ANE
有/无source对应PNG亦exact。变更没有改矩阵值或输出精度。但两个有限
screen都没有给出可靠整请求盈利；shape/JIT/内存带宽等成本还需单独定位，
不能从最后一个text样本挑胜利。v1/v3构建与初始化不同，不合并分母。

## 两参考图：真正组合与分区扫描

最终v4 Private library，六步真实adapter、两个原reference、三个不同
prompt；GPU baseline、DiT-only、DiT+encoder全部使用同一source-retention
策略。global/DiT share与encoder share各自记录；encoder一直3072。

| DiT share | GPU+source median s | DiT+source median s | DiT+encoder+source median s |
| --- | ---: | ---: | ---: |
| 5120 | 13.146825 | 14.692104 | 14.840383 |
| 1536 | 13.133736 | 15.803558 | 15.985268 |

全部arm条件miss、source只load1次、227实际adapter projections、实际6步。
DiT实际调用累计256/512/768；encoder本次72、累计72/144/216。它们是
两个不同executor与计数，不把encoder calls混进DiT。failure/fallback0；
memory报告六个trial完整、swap-in/out0，process-tree phys-footprint峰值
约39.0–41.1GB（十进制），不是整个系统/driver上限。

这两个显式分区都慢于公平GPU；小DiT share更慢，不推荐默认。目视5120
组合whole与原像素三crop：构图、壶形/光照很接近，有釉面/高光变化。
1536组合仍可识别，但blue壶身、盖/钮的大小和形状有显眼差异，记录为
`visually_very_similar=false`，不作为close-fidelity改进。无多seed或用户
acceptance，`qualification_passed=false`不变。

## 验证、构建与磁盘

最终Private/Public各24项选定回归通过；额外1项真实source/executor
switch/error/off/unload session test通过：cached source跨GPU/ANE复用，
关source后retained0，unload后load counter重新从1开始。factory真实
build-on/off/Public-fallback测试1项通过；上述执行无skip。包含新source
ctime/alias、48GiB/20GiB/growth/unknown-memory guard、immutable share
override和新工具的paired/cumulative-call拒绝测试。不是全仓make test。

Public release-binary guard通过。两库489 source inputs各自匹配；构建
仍包含原ConvRot drafts，提交不夹带它们。库SHA256：

```text
Private eea37c749bf612ba37ea34ab8e60c2b5b54736ea66b7759800af4e14bb5cb610
Public  e76a2969ac80087df0ddd6aa439be62eb951096f7a9fdfeb01c1f33fc30f0ed7
```

原始receipt/hash、完整冷/热样本、失败记录与review见
[机器证据](../design/validation/local512-encoder-source-20261007.json)。
完成owned jobs后仅清理五个source v1/v2/v3/v4 build的1065个`.o`，
83,319,960 logical bytes（约79MiB），可重建；保留库、CLI、manifest、
log、PNG、模型和adapter，没有大activation dump或新模型下载。

后续仍需：真正盈利的Qwen LoRA GPU/ANE FFN分区/融合、base generation
与编辑的独立profitability、编码权重residency的内存/带宽收益、GGUF
ahead-decode与ConvRot kernel/视觉修复。兼容与cache命中不替代这些目标。
