# Qwen完整block上下文：数值执行完成不等于交付合格

<!-- knowledge
id: technique-gpu-ane-case-studies-qwen-block-context-and-storage-boundaries
type: case-study
aliases: evidence-qwen-block-context-20261016
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[总目录](../../README.md) · [完整FFN边界](../../gpu-ane/private-ffn-boundary.md) · [前批shared FFN](qwen-shared-ffn-consumer-cost.md)

2026-10-16已有实验的整理记录；本次代码整理未重新运行GPU/ANE。设备沿用M4 Max/64GiB、macOS26.6.2/build25G83、SDK26.5、MLX0.32.0。没有修改产品默认、重建产品、替换App、下载模型、清理系统cache或删除旧实验。

## 边界与强分母

`bicycle-s43`使用seed43及不同prompt，二者均不同于baseline seed42；公开证据只保留prompt SHA。复用匹配648项native source的已有runtime，在ordinary GPU request的step1/layer0 observer中借用实际9个参数、原始输出与原compiled block callable。完整M1024/H4096/F12288、base/no LoRA、private paired prefix2048固定。

五臂为原始GPU callable、staged GPU、copy hybrid、shared serial、shared hybrid。attention/norm producer、完整FFN与modulation/residual consumer都计时，staged GPU对原输出exact0；zero-FFN返回实际residual、zero-private返回实际GPU tail也exact。非零FFN改变4,023,583个block元素，不省略下游工作。

固定whole-block relative L2≤0.001/cosine≥0.999；完整FFN/prefix及独立full-K/F FP64 sample均≤0.03/≥0.999。不是将旧FFN replica直接改名为完整block，也不是未改变compiled内部FFN的直接capture。只有owned CPU BF16副本可离开observer，借用参数和输出前后hash保持不变。

## 原始失败与部分成功全部保留

- bulk holdout capture被原267MiB admission拒绝，未启动native inference。
- owned TMPDIR的block run通过staged/FP64/runtime-argument前置检查，但private load返回false；NSError为nil，domain为unknown/code0，timed调用0。compile selector未调用、compile耗时为null，load约604.9ms。不能由nil error/code0推断成功或硬件不支持。
- inherited TMPDIR run load约1.038s，165次完整block调用/99个exact FFN join均通过；正常unload、两份adoption deleter、mapping释放与原borrowed存储检查通过。最后16MiB raw-pair导出会突破128MiB reserve，协议仍failed。
- lossless archive run保持同一timed-loop/validation源码与实际输入hash，又完成165次/99个join；内存中按BF16高/低字节平面与zlib压缩，按实际压缩大小加128MiB reserve和metadata门槛拒绝。没有真实archive文件或压缩大小metadata，不声称实际模型decode成功。

三份block记录的实际9参数、exposed FFN输入、reference和timed-loop SHA一致。TMPDIR上下文的观察不同，但cache/时序未控制，不宣称因果、通用修复或内部无编译。全请求配置为20steps，实际只完成step0并受控停在step1/layer0；未VAE decode/image export，候选未回灌请求。

## 绝对耗时诊断，不发布合格倍率

| Arm | raw-export-refused中位ms | archive-export-refused中位ms |
| --- | --- | --- |
| gpu-original | 33.322 | 33.323 |
| gpu-staged | 33.670 | 33.711 |
| copy-hybrid | 30.891 | 30.917 |
| shared-serial | 38.318 | 38.769 |
| shared-hybrid | 30.672 | 30.687 |

两份记录合计330个完整调用、198个actual-part exact join，所有first/warm/measured均保留。whole-block最大observed relative L2约0.00057705；完整FFN约0.00266944，prefix约0.00452714，full FP64 sample约0.00364866，prefix FP64 sample约0.01049101。两份wrapper comparisons均为null；artifact协议未完成，不能由上表派生合格加速比分数或晋升候选，更没有物理ANE placement/overlap与完整请求/图像质量资格。

## 离线复核与剩余缺口

合同verifier与复核source/library/原权重spans/bank/installed assets、完整样本顺序、copy量、host依赖、actual alias和正常回收。公开capsule不含raw tensor、prompt、地址或机器路径；data verifier不能离线重新执行原始block/private输出，也不是硬件attestation。

回归测试覆盖失败状态不得晋升、固定预算、原GPU callable、完整join与owner/dependency反例；便携C++小矩阵将FP64 oracle与标准库math.fsum独立比对。archive roundtrip/坏header/trailing/解压cap仅用合成字节fixture，不回填真实模型持久化资格。

下一步须先解决受约束的raw持久化与复核，再讨论完整请求/图片质量、更多层/LoRA、冷启动/内存/长时/失败取消；不为通过协议降低原存储reserve或继续更换GPU/压缩变体。完整研究范围仍未完成，本记录只归档当前证据边界。
