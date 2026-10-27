# 压缩 attention 的真实 ps0/ps512 对照：crossover、尾部分区与舍入位置

<!-- knowledge
id: technique-metal-case-studies-compressed-attention-crossover
type: case-study
stage: prefill, decode
operator: attention, quantization, layout, scheduling
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: memory-bound, launch-bound, precision-loss, wrong-layout, measurement
semantics: numerically-equivalent, approximate, contract-only
representation: fp16, q8, packed
boundary: producer-consumer, cpu-gpu
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU 导航](../README.md) · [原生 writer](native-kv-writer-and-consumer.md) · [materializer](compressed-kv-materialization.md) · [来源](../sources.md#compressed-attention-sources-20261016) · 全部样本 · [核查（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，第六轮。上一轮是 native encode→明确 copied bridge→小型 MLX SDPA；本轮**直接执行原始压缩 paged attention 的 ps0、ps512 producer 和 reducer**，与原始 materializer→MLX SDPA 放到驻留缓存、host result-ready 的同一完成边界比较。384 个正式输出全部对独立 CPU FP64；不把实际原生 shader 执行扩大成已运行完整 vLLM Primitive/scheduler。

本机 Apple M4 Max/64 GiB、Python3.11.16、MLX0.32.0、macOS26.6.2。K8 signed/V3、D128、8Q/2KV heads、8193 KV tokens 的两个独立 seed 中：单 query 时 ps512 压缩路径更好；Q16 materialization 仍较慢；Q64/Q128 时 materialize+SDPA 相对 ps0 的 paired median 为约 **1.45–1.47× / 2.66–2.71×**。这些是具体 API/shape 的热态观测，不是生产阈值、模型 TTFT 或跨引擎总排名。

运行后另行核对的工具链为 SDK26.5、Apple clang21.0.0、OS build25G83；这是 post-run 环境快照，不回填成每条 GPU dispatch 的 compiler attestation。

后续[第七轮](mlx-runtime-handoff-and-observability.md)已在新隔离的MLX0.32.2环境运行真正原始 Primitive 交接和同 runtime 比较；本页保留第六轮的独立 launcher/旧 wheel scope，不将两轮不同边界的倍率当作版本升级结果。

## 1. 比“一个 materializer 快多少”更强的比较

| arm | 实际工作 | 用途 / 限制 |
| --- | --- | --- |
| `native-ps0` | 原始 per-query compressed attention，最后对 weighted V 做逆 FWHT | 一次 native dispatch；不是所有 decode 情况的最快基线 |
| `native-ps512` | 原始 512-token producer + 原始 log2 reducer，最后逆 FWHT | 两次 native dispatch；本实验可强制执行，不代表生产 prefill 会自动选它 |
| `materialize-sdpa` | 原始 fused gather/decode/FWHT→FP16 K/V→MLX SDPA API | 包含 materializer 与完整 attention API，不再只测解码 |
| `ready-sdpa` | 相同 FP16 K/V 预先解好→MLX SDPA API | 少做工作的成本下界；不是 compressed attention 候选 |

每个点四臂读取同一份 packed bytes/metadata、相同逻辑页序、相同两套 Q/causal mask。跨 Q 点会生成不同合成 cache/Q，**不是同一模型 prompt 的 nested suffix sweep**。没有从 Q16/Q64 两点插值得出“最优 cutoff=某个整数”。

固定上游 `vllm-project/vllm-metal@468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e`；16 个文件均匹配 commit blob/SHA，前后不变。原始 `_build_v2_paged_attention_source` 生成 200,279 bytes library source，未修改 shader body。自有 C ABI launcher使用公共 Metal，MSL3.1、`fastMathEnabled=NO`；driver不安装/加载与现有 wheel 不匹配的 private-ABI extension。

关键源码：[pagedattention.metal](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/pagedattention.metal)、[host bindings](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/paged_ops.cpp)、[共享名字/内存 helper](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/paged_attention_kernels.h)。精确阅读区段、其余编译依赖的有限审计范围见 source manifest，不把完整编译输入列表称作全文件逐行审计。

## 2. 真实 ABI：function constants 也是算法的一部分

本轮 native pipeline 为 `paged_attention_half_cache_char_uchar_hs{64|128}_bs16_nt256_nsl32_ps{0|512}`；reducer 为 `paged_attention_v2_reduce_half_hs{D}_nt256_nsl32_ps512`。K 的 char/uchar、T、head dim 和 partition suffix 都是独立特化身份。

| function constant | 本轮设置 | 错配风险 |
| --- | --- | --- |
| 10 `use_partitioning` | ps0=false；ps512=true | runtime flag 与 compile-time partition size 不应混用 |
| 20/30/40 | alibi/fp8-scale/sinks=false | 改变参数存在与数学语义 |
| 50 `use_turboquant` | 正向 producer/reducer=true | reducer 若为 false，会把旋转域结果当普通 V 输出 |
| 60/70 | K8/V3 | 位解包、codebook、cache 格式 |
| 110 | window_q=0 | 本轮每 query 一组，不验证 speculative window dispatcher |

producer grid 为 `(8,Q,ceil(N/512))` 或 `(8,Q,1)`，threadgroup256；reducer grid `(8,Q,1)`。动态 threadgroup memory、FP16 partial 与两份 FP32 stats 按原 helper 计算，reducer 的 static+dynamic 总量先检查设备容量。

host 绑定中 Q/packed K/V、K/V metadata、int32 cache stride、int64 scale stride、block table、seq length、`cu_seqlens=[0,Q]` 都明确传入。page/head padding 为 0 或 4 个元素；MLX 侧用独立诊断 kernel 读实际 strides，证明各臂不是只在 shape 标签上相同。

生产 host 的 split gate 限定 pure-decode/window 等条件并考虑 grid/core/partition 数；TQ 也不走非压缩 GQA 专用 kernel。**强制 ps512 在 Q>1 数值通过，不等于 production prefill dispatcher 已允许/采用该算法**。这里没有更改默认 routing。

## 3. ps512 的尾部：rectangular pitch，不是每行实际分区数

对于 N 个历史/当前 KV、Q 个新增 query，第 r 行有效长度为 `N-Q+r+1`，只有其有效分区能参与 reduce。scratch 的物理 pitch 却仍按整批 `ceil(N/512)`：

```text
N=513, Q=17
effective lengths = 497,498,...,513
active partitions = 1,1,...,1,2
scratch pitch      = 2（每个 query/head 都一样）
```

不能把前 16 行的统计/partial row stride 改为 1，也不能因为最后一行有两分区就读取前面行的第二槽。本轮每次 native 调用前将 partial、max、sum 全部置 NaN；输出 body 置 half NaN，前后各 256 bytes guard。输入上传/poison 在四臂计时外；每个正式 native 调用完成后检查 guard，并对全部输出查 finite/FP64。

同样覆盖 N8193、Q1/16/64/128 的第 17 分区尾部；Q128 的前 127 行只有 16 个有效分区，最后一行才是 17。另有 N257 的 reducer 单分区 early-out：TQ 路径即使只有一个贡献分区，仍需逆 FWHT，不能直接 copy。

CPU 布局/reducer 合同验证 active count、rectangular scratch、log2 rescale 与错误最大 pitch/未逆变换负控。它是原创 math model，不是 CPU 自动执行 GPU。

## 4. reducer 的数学与真实 dtype 边界

producer 以 log2-space max、exp-sum 做 online softmax；reducer 用 `exp2(m_i-global_max)` 对齐统计，合并归一化 partial。不能把 log2 max 传给自然指数的 combine，也不能把相同 sink 在每 warp/partition 重复相加；本轮 sinks=false，没有顺便授予 sink 资格。

TQ producer 累加 centroid×scale 的**旋转域 V**；ps0 对最终向量逆 FWHT，ps512 把旋转域 partial 写到 `tmp_out`，reducer 合并后才变换。源码的线性恒等式是设计依据，但实际 `tmp_out` 类型是 **T=FP16**，还存在 epsilon 与浮点归约次序。不能把注释中的线性描述扩展成“全链没有中间 FP16 舍入”或 bitwise 等价。

对应两个独立 CPU FP64 reference：

- 两者 K 都按 decoder 合同先 cast FP16；Q 也是 FP16。
- native reference 的 V 使用 decoded FP64 数值，不做每 token FP16 cast，模拟 deferred inverse 的数学目标；ps512 partial 的 FP16 rounding 由误差门限覆盖，不伪装成逐位 oracle。
- materialized reference 在每 token decoded V 后 cast FP16，再做 FP64 QK/softmax/PV。

两个 reference 本身最大 relative L2 差约 `2.44e-4`，明确留档。正式输出最大 relative L2：ps0 `2.18e-4`、ps512 `3.10e-4`、materialize+SDPA `2.99e-4`；门限为 native `0.004`、MLX `0.002`，不是只拿候选互相比较。

FP64 bit-plane unpack 先对标准库逐 bit 解码核对，FWHT 对 scalar 参考核对。共有 384 个正式完整输出，每 arm96，交替两套新 Q；全部元素 checked，不只检查摘要/finite。raw tensor 不入库，合成输入 hash、客户端/CPU oracle 前后 hash、逐次误差与时间全部保留。

另外把原始 reducer 的 function constant50 故意设为 false，保持同 ABI 与 producer，产生有限但错误的输出（relative L2 约 1.36–1.51）。此负控单列，不进入正向计时，也不向 GPU 提交非法地址。

## 5. 结果：优化后的 decode 基线会改变结论

两进程 seed17/29 顺序执行，6 点/进程、每点4 block，顺序 `ps0,ps512,materialize,ready,ready,materialize,ps512,ps0`。下表是两个 run 各自 arm 中位数的范围，单位 ms，不是置信区间。除首行 D64，其他 D128；均 8Q/2KV heads、FP16、signed K8/V3。

| KV N / query Q | ps0 | ps512 | materialize+SDPA | ready-KV SDPA |
| --- | ---: | ---: | ---: | ---: |
| 257 / 1（D64） | 0.173–0.200 | 0.185–0.208 | 0.208–0.233 | 0.174–0.208 |
| 513 / 17 | 0.231–0.241 | 0.257–0.259 | 0.351–0.352 | 0.311–0.318 |
| 8193 / 1 | 0.941–0.994 | **0.253–0.272** | 0.394–0.421 | 0.273–0.287 |
| 8193 / 16 | 0.877–0.895 | 0.666–0.719 | 1.494–1.496 | 1.365–1.366 |
| 8193 / 64 | 2.085–2.103 | 2.014–2.062 | **1.412–1.419** | 1.339–1.400 |
| 8193 / 128 | 3.826–3.847 | 3.827–3.830 | **1.421–1.428** | 1.364–1.410 |

paired ratio 按 block 内各 arm 两次 mean 相除，再对4个 block 取 median，不是表中两列中位数之比。N8193/Q1，`ps0/materialize≈2.21–2.43`，乍看 materialization 更快；但 `ps512/materialize≈0.63–0.64`，split 压缩路径更快。**省略已有 split-KV 优化，会把基线缺陷错当成 materialization 的适用条件。**

N8193/Q16，ps0/materialize约0.60–0.61；Q64约1.45–1.47；Q128约2.66–2.71。Q16 与 Q64 的 ready-KV SDPA 也呈不同成本区间，说明消费者 dispatch/shape 不能从单 materializer 倍率推算。本轮没有独立 capture MLX SDPA 内部 kernel，不为此差异擅自命名硬件根因。

这些点支持“query 足够多才能摊销完整 K/V 展开”的机制，但不能给 M4 推广全局 cutoff；未测 Q32、其它 GQA geometry、BF16、其它 K/V codec、混合 batch/共享 prefix、真实 prompt 和 background/thermal 隔离。强制 ps512 的多 query 优势只是一项后续 runtime 实验候选，不自动成为 production 可选项。

## 6. host wall、GPU timestamp 与内存预算不要混列

正式计时为 Python API 调用→实际输出完成：native 包含 ctypes→encode/submit→`waitUntilCompleted`；MLX 包含 graph/API→`eval+synchronize`。输入/poison/upload、CPU oracle、验证读回在计时外。输入驻留与完成边界匹配，但 runtime 包装不同；这是 API-level 组件对照，不是同一 runtime 的 shader-only A/B。

native ps512 使用两个 tracked-resource encoder、同一 command buffer，使 producer→reducer 有可观察的完成顺序；不模拟 private MLX encoder 的自动 fence。native 输出/scratch 预分配复用，MLX arm 会产生新的 lazy output/allocator 请求。timing 前也预留 ready-KV 对照，进程同时持有 native、MLX 与 CPU pool，因此不是相同模型 KV capacity 下的峰值/吞吐验收。

每个 native command 的 GPU start/end 单独记录，MLX 同列明确 null；没有用 host wall 代填 GPU 时间。[Apple GPUStartTime 合同](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)说明其单位秒、与 system mach time 关联、完成前为0；本客户端在 wait 后相减换 ms。这是 command-buffer span，ps512 包括 producer/reducer/encoder 间隔，不是两个 kernel 的独立时间或 DRAM/occupancy counter。

N8193/D128/Hkv2 最终 materialized K/V 逻辑约 8.001 MiB。ps512 scratch 为 `Q*8*ceil(N/512)*(D*2+8)`：Q1为35,904 bytes，Q128为4,595,712 bytes。这里只说明 shape 与逻辑分配；本客户端即使测 ps0，也保留 ps512 scratch，不能据此宣称 ps0 的生产驻留等于该数。没有完整链 RSS/峰值、队列 backlog、取消/回收或长时热状态验收。

## 7. 上轮 half division 假设：补了 source-expression 级实测

自有 companion shader 与原始 encoder 分开编译，固定同一 MSL 版本，用256个有限范围输入、buffer divisor=254，分别测四个表达式并保留全部结果：

| 表达式（最后均 cast half） | `fastMathEnabled=NO` | `YES` |
| --- | --- | --- |
| `half a / half(254)` | 匹配 half reciprocal 模型；与理想 RNE 不同31/27项 | 同左 |
| `half a * half(1/254)` | 匹配 reciprocal 模型 | 同左 |
| `float(a) / 254.f` | 仍匹配 reciprocal 模型 | 本样本匹配理想 RNE |
| `half a / half(buffer_divisor)` | 本样本匹配理想 RNE | 同左 |

31/27分别为两个 seed 的差异数，不是通用误差概率。上一轮具体 scale `0.006603240966796875` 对 `0.0066070556640625` 在这里重现。常量与 buffer divisor 的数值相同，但这个表达式矩阵的观测不同；甚至“先 cast float”也不能无条件当作修复。

这比只猜 reciprocal 更强：改变可控 source 表达式/编译开关，结果跟着改变；但仍不是机器指令、IR 或编译器 pass 归因证明。`xcrun --find metal` 返回72，当前仅 CLT 的离线 metal tool 不存在；runtime compiler 可运行，不代表能获得 AIR/disassembly。本轮未安装工具链、未改写上游 shader。

官方 [legacy Boolean](https://developer.apple.com/documentation/metal/mtlcompileoptions/fastmathenabled) 与 [mathMode](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode) 的 Markdown 内容身份已固定。新页说明 Boolean 映射同时影响 mathMode 与 mathFloatingPointFunctions；legacy discussion 另有 single-precision variant 描述，不能抽出一句当逐表达式 bitwise 保证。这里记录实际使用的 Bool/SDK/输出，不从 `NO` 或 `YES` 的名字推断“全局必然更准/更快”。换新属性或工具链后重新测；上一轮 encoder exact-byte gate 仍未通过。

## 8. 复现与尚未完成的完整链

两个 output 目录均须新建，先审核固定 source checkout；driver 不 clone/install/model download，不更改系统或生产默认。小型原始 shader source 只在进程内组装，第三方 body 不进入 tracked Wiki。source/client/CPU oracle 前后固定；仅读明确 final run 与显式 smoke 历史，不联网或重跑 GPU。9项离线 tests 重算全部配对、全部输出计数/门限、CPU division 模型、真实 stride/pitch、反例和身份。

[版本复查登记](../../guides/knowledge-base.md)将本页绑定到 source/evidence SHA、精确记录环境和30天复查窗口；变更 SoC/SDK/wheel/编译器或 capsule 内容会提示重审。测试也检查换成 M5 后必须 needs-review。即使环境匹配，检索也不授予设备兼容或性能资格。

本轮比上轮推进的是 **actual compressed attention/reducer 消费者与匹配完成边界**；仍未执行生产 `PagedAttentionPrimitive`、native writer→同 backing 的无 CPU 交接、scheduler admission、NAX、真实 checkpoint/LoRA 或模型质量。完整 package 的 Python/MLX 精确 pin 仍不匹配。下一步应把同一原始 kernel 放回受支持 runtime，再用真实 KV/请求、相同预算和独立 timeline/counters 检验局部结论；同时继续编译器/其它引擎的研究，不把本轮9项 tests 当整个持续目标完成。
