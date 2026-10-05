# ConvRot 寄存器核、GGUF 有界解码与 ANE 精度探针

本地实验时间：2026-10-06，Asia/Singapore（UTC 仍为 2026-10-05）。
接续 `convrot-group-scope-2026-10-05.md`。完整双后端、Z/Qwen 四格
base ≥1.2×、实模型 LoRA、质量/内存/trace 目标仍未完成。

## GPU：保留舍入边界的寄存器 H256

`convrot_rotation.hpp` 新增 `SimdRegister`。一个 SIMD group 的每个 lane
持有八个相隔 32 列的值；stride1/4 使用 shuffle，stride16 跨 lane 与
寄存器，stride64 完全在寄存器内。一个 128-thread TG 处理四组 H256，
不使用共享 scratch/barrier，仍保持 Comfy H4⁴ 次序、原左结合加减、
FP32 中间值与一次最终输入 dtype 舍入。不是 Sylvester 或 A8。

72 个 FP32/FP16/BF16、contiguous/strided、不同 M/K 的 case 对 Shared
一致；寄存器输出按字节比较。另有 256 个独立基向量以及三种 dtype
的 cancellation、signed zero、tiny/large value、768 列尾组检查。
旧 Shared/Simd/Quad 保留为对照，不删除负结果。

同 binary、BF16、每 arm 三次 warmup、27 个串行循环换序样本；wrapper
每 100ms 记录 host load/进程 comm，保留全部 samples 与观察间隔。
这些是带观察器的 **operator host span**，不是 GPU timestamp、整请求
资格或物理 GPU/ANE trace。交互式桌面和其他进程并未被隔离。

| M / K | Quad median ms | Register median ms | Quad / Register |
| --- | ---: | ---: | ---: |
| 1056 / 3840 | 0.271584 | 0.262541 | 1.03444× |
| 1056 / 10240 | 0.436958 | 0.412000 | 1.06058× |
| 4224 / 3840 | 0.587125 | 0.544750 | 1.07779× |
| 4224 / 10240 | 1.256375 | 1.176625 | 1.06778× |

`mlx.cpp` 仅在原本已有的显式 Metal ConvRot 路径、M4 Max、BF16、
M1024–4224、K3840/10240 使用寄存器核。默认 dense-H、其他 device/
dtype、小向量与模型文件不变。对旧 butterfly 精确并不能消除其与
默认 dense-H 不同 reduction 带来的历史 block N1 失败。

证据：`outputs/convrot-register-components-20261006-27samples.json`。
首轮九样本也保留，但不混入 27 样本的 median。

## GGUF / ConvRot：真正的消费算子对照

新增 `AffineDenseWindow` 消费已经生产好的 MLX affine Q4/Q8；包括
Q4_0/Q4_1/Q8_0 GGUF importer 输出，不是 raw Q4_K/Q6_K decoder，也
不是 inverse ConvRot。使用原 FP16/BF16 metadata dtype 调用 typed
dequantize，不先宽化 metadata；ConvRot 保留 rotated basis 和 legacy
BF16 stored scale，不把它冒称原 FP32 scale。

这是 owner-thread 的同步研究 consumer，目前 **没有自动接入模型默认
路由**。两个槽是两个矩阵槽，不是两个完整 FFN/layer 槽。不宣称异步
提前解码或物理 overlap；native window 为后续 pager 接入提供受限消费
接口，而不是已经完成跨 step 的复用/预取策略。

容量同时受 local retention upper 与共享 ledger 约束，先 admission
后 decode。内容 ticket/generation、所有 source array ID、bits/group/
dtype 隔离缓存；保留 source owners 防止 ID 回收。bank 内容被重填时
调用者必须更新 ticket/generation，不能仅按地址/shape 命中。完成并
验证 finite 后才发布，失败丢弃全部新目标。escaped views/lazy readers
通过 Data 持有 ledger claim；eviction 不假装释放它们，新 admission
仍可能被拒绝。ledger 仅登记 dense backing，不等于整个进程 RAM 上限，
编译、验证 reduction、其他 tensor/cache/ANE arena 仍需整体预算审计。

`gpu_weight_consumer_probe` 读取真实 layer0 gate/down 权重，使用合成
sine activation；同一 binary 中串行交替 packed QMM / prepared GEMM。
解码样本每次明确 clear、重新 decode、finite 验证、完成等待，包含
该窗口的 allocation/cache/validation 成本，不称纯 kernel decode。
compute 各三次 warmup、15 个 hot 样本。输入旋转、文件读/初始 packing
不计入两条 compute arm；M1056/4224 是诊断矩阵行数，不等于完整请求。

| 来源 / projection / M | packed ms | dense GEMM ms | decode ms | 每次重解码合计 ms | 回本复用次数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GGUF Q8 / gate / 1056 | 6.102875 | 5.926417 | 3.889041 | 9.815458 | 23 |
| GGUF Q8 / gate / 4224 | 23.930541 | 22.390083 | 3.885375 | 26.275458 | 3 |
| GGUF Q4 / gate / 1056 | 6.086500 | 5.943167 | 3.942084 | 9.885251 | 28 |
| GGUF Q4 / gate / 4224 | 23.705584 | 22.276083 | 3.968958 | 26.245041 | 3 |
| GGUF Q8 / down / 1056 | 6.262917 | 6.175792 | 3.962375 | 10.138167 | 46 |
| GGUF Q8 / down / 4224 | 24.070167 | 22.669416 | 3.933541 | 26.602957 | 3 |
| ConvRot / gate / 1056 | 6.580750 | 5.926125 | 4.115709 | 10.041834 | 7 |
| ConvRot / gate / 4224 | 25.694042 | 22.317667 | 4.018083 | 26.335750 | 2 |
| ConvRot / down / 1056 | 6.735083 | 6.173042 | 3.972167 | 10.145209 | 8 |
| ConvRot / down / 4224 | 25.918084 | 22.658750 | 3.930709 | 26.589459 | 2 |

回本条件为 `decode/R + dense < packed`，表格用分离样本的 median
估计，不是实际 R 次完整窗口速度/置信区间。十个有限输入对照的输出
按 bytes 一致；不推断所有 source/dtype/trajectory 都逐位一致。
GGUF screen 使用 FP16，ConvRot 使用 BF16，因此不作跨 dtype 速度排名。
每张 dense 为 75MiB，两矩阵 upper 为 150MiB；完整 Z FFN 为225MiB，
两个完整 FFN 至少450MiB，不能用本窗口的150MiB替代。

结论：当前逐层 eviction 后每 step 重解码没有收益，不提升为默认。
大 M 若能够跨调用真正保留热点矩阵才值得继续；需要测实际 residency/
miss/eviction 和完整窗口，不能把多层轮转误计为同矩阵 R 次复用。
原 GGUF container、packed master、model/adapters 均不改，不生成 sidecar。
fused raw decode/GEMM、异步当前＋下一层、M=1 modulation 与 Qwen21
消费者仍未在本轮实现/资格验证。

原始 `outputs/*-consumer-20261006*.json` 保留 binary hash、payload/
codes/actual stored-scales hash、逐样本时间、host load 与 gap。最初一个
GGUF invocation 参数错误与数次 ConvRot provenance SIGBUS 也保留，不
计为性能/数值样本。SIGBUS 定位到 hash 读取逻辑 nbytes 的 broadcast
metadata backing；最终先 contiguous 化 metadata 再 hash，并使用
持有 fd 的 CPU loader/mtime/ctime 校验。不是模型 kernel 的质量结果。

## ANE：新增正控制与舍入模拟的负结果

新增可重建的 Private cast probe；FP16 cast 正控制编译并执行，256个
样本零 mismatch。同形状的 `bf16` / `bfloat16` 两种文本 MIL cast 都被
当前 M4 Max / macOS26.6.2 的实际编译器拒绝。这是当前 tuple 的证据，
不泛化成其他系统/编译器永远不支持 BF16。

纯 FP16 模拟另用精确 power-of-two bin + select + real_div/round/mul，
没有 FP16 log2 near-boundary 的 shortcut。图能编译；16次实际 driver
调用，63,480个有限 FP16 encoding（排除8个 BF16-RNE 后超 FP16 范围
的 encoding）对独立 CPU BF16-RNE→FP16 oracle 有 **14,977 mismatch**。
保留首八个错误与完整数目，进程正确返回失败；不把 parse/compile
成功当数值通过，也不接入 ConvRot W8A8 图或放宽门槛。尚未独立定位
小量 divisor/reduction/select 的 compiler lowering，不能只凭首个
错误推断唯一原因。

证据：`outputs/ane-bf16-cast-and-emulation-20261006.log`；最初没有正控制
的 probe log 仅保留作历史补充。此诊断不说明原生 INT8 MAC、FP32 ANE
arithmetic、完整模型精度、速度或物理 overlap。

混合建议仍是按 operation/layer 的 shape/source/precision recipe 选择
一种 ANE backend，与 GPU complement 配合：Private 用已资格的长行
W8A8，Public 用需稳定 FP16/兼容性的 consumer，短/敏感层先保留 GPU。
同一片权重不同时常驻 Public FP16 与 Private W8 两份，cache identity
必须隔离 backend/recipe/ABI。**该 per-layer 三选一路由尚未自动接入**；
现有 auto 是构建期 Private→Public capability fallback，真实 chunk
失败仍须整个 operation GPU 重算，不能发布部分 scratch。

前一轮 row/group/input-only/hidden-only 四步质量失败仍有效，本轮探针
没有修复它们。没有正式 1.2×、实模型 LoRA 或四格 Z/Qwen 完成声明。

## 构建与回归边界

Public 新库：`build/convrot-register-public/libturbocider.dylib`，
SHA256 `bc248acb8b82900570e82c77dcd3663bfa44d1a3757b93c81a74feb1e7a2cd50`。
Private=0、experimental-probes=0，actual release-binary guard通过。
构建包含原本未提交的 ConvRot drafts，是 working-tree snapshot，
不是 clean staged-tree rebuild；selective commits 不夹带那些 drafts。

`TURBOCIDER_TEST_GPU=1`、上述新 Public 库，7项 suite 全绿（6项实际
Metal test、1项 source opt-in contract），涵盖96项FFN helper、两种大M
完整packed FFN对Shared逐位一致、24项channel/base projection、24项
typed FP32 partial、rotation独立基向量/边界与12种window geometry。
另外7项CPU ConvRot数学＋2项observer单元测试通过。不是完整
`make test`绿色，Private舍入模拟仍是明确失败，不列为通过的优化。

本轮 standalone rotation binary SHA256：
`1dc47c6abbd205f432696efd1bd12b9a7dccf3a5f3180ad91fe3cd340a3718dd`。
ANE probe binary SHA256：
`fcc79670b051cff55e54192e98e77fc3aaf45fdfe22f0a1aa4b85410732ddfd9`。
各 consumer screen 的实际 binary hash 在各自 receipt；其中部分实测
发生在最后的 fd-reader 身份加固之前，不能冒称全部是最终 staged
tree binary。最新consumer source的window测试通过，fd-reader 的真实
ConvRot smoke 也逐位一致；该 smoke 时 Swift App 构建尾部仍在运行，
**不计其时间为性能证据**，不混入上表。旧 raw provenance 错误与
compiler/driver 数值失败均未删除。
