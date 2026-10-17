# 原生 KV writer 到 attention 消费者：原地写、图依赖与编码器非逐位等价

<!-- knowledge
id: technique-metal-case-studies-native-kv-writer-and-consumer
type: case-study
stage: prefill, decode
operator: attention, quantization, layout, scheduling
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: wrong-layout, stale-cache, precision-loss, lifetime, measurement
semantics: numerically-equivalent, approximate, contract-only
representation: fp16, q8, q4, packed
boundary: cpu-gpu, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU 导航](../README.md) · [上一轮 materializer](compressed-kv-materialization.md) · [本轮来源](../sources.md#native-kv-sources-20261016) · [验证记录（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，第五轮。**真正执行了原始 `tq_encode` Metal kernel，再接原始 materializer 与 MLX SDPA API**。两个 seed、六个配置、每配置两代原地写，合计 24 次 native writer dispatch、24 对 materialized K/V、384 个正式下游输出；guard/padding/跳过 slot、读回新数据与完整输出数值通过。

但**编码器 bitwise 等价没有通过**：11/24 代与理想 FP16 除法模型的 K 位流不同，合计 27 个 byte；与原始 MLX K 编码路径也有 365 个坐标代码、381 个 scale 不同。不能因下游能正确消费实际字节，就改写成两个 encoder 逐位等价。当前不是上游要求的完整 Python/MLX 环境，Metal 编译选项也单独固定，因此这些是本次受控环境的差异，**不证明受支持配置的上游缺陷**。

后续[第六轮](compressed-attention-crossover.md)已补真正 compressed attention/reducer 对照和独立 division 表达式校准；本页保持第五轮历史 scope/数字。第六轮也没有执行 private-ABI Primitive，或撤销本页 encoder bitwise gate 的失败。

## 1. 资格逐项拆开

| 要证明的事情 | 当前证据 | 状态 |
| --- | --- | --- |
| 原生 encode/scatter shader 是否真实执行 | 原始 source builder、目标 pipeline 名、函数常量、写出完整 bytes/metadata | 已执行 |
| 同一 native cache 是否原地更新 | 两代间五个 MTLBuffer contents 地址相同、完整 backing hash 改变 | 此 owner/同步协议通过 |
| 是否越过目标写集合 | prefix/suffix guard、页/head padding、未写 slot 的完整补集检查 | 全部零 mismatch |
| native writer 与理想 RNE 编码器逐 byte 相同 | 完整字节对照与具体差异坐标/scale/input | 未通过 |
| native writer 与当前原始 MLX K encoder 相同 | 原声明 `turbo_quant_encode_key` 三方比较 | 未通过 |
| writer→materializer 是否零 copy | native 完成后 CPU snapshot，再新建 MLX pool | 明确不是 |
| 是否证明 MLX primitive 的 reader-after-writer fence | 本客户端不加载 private-ABI 扩展，只独立提交 Metal | 未证明 |
| materializer→SDPA 是否完整数值正确 | actual-byte 独立 FP64 decoder→FP16→FP64 GQA oracle，全输出 | 此小消费者通过 |
| production paged attention/模型/HTTP 是否通过 | 本轮未运行 | 未通过/未验收 |

这些状态不是一个“整体 pass”开关。evidence 中的执行 `status=passed` 只说明声明的组件/守卫检查完成；另有两个 encoder bitwise qualification 布尔值明确为 false。

## 2. 原始源码与本地执行边界

固定 `vllm-project/vllm-metal@468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e`；15 个输入文件核对 commit blob、SHA256，执行前后不变：

- [原生 `turboquant.metal`](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/turboquant.metal)，特别是 L460–768：`tq_encode`、byte packer、midpoint preload、kernel instantiation。
- [原始 library builder](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/__init__.py)：AST 执行未经改写的 `_read_metal_source` 与 `_build_v2_paged_attention_source`，读取原 constants，产生 200,279 bytes 的完整 v2 library source；与第四轮固定 commit 相同。
- [原生 wrapper](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/paged_ops.cpp)，L1370–1585、L2344–2398：native buffers/graph identity、函数常量、element stride、descriptor binding；**只阅读，没有执行原 C++ Primitive**。
- [真实 SDPA call site](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/impls/sdpa.py)，L640–742 与 L870–970：输入 contiguous、原地输出 rebind、fresh reader、每层 drain；只阅读。

其余 `float8/utils/reshape/pagedattention/pagedattention_tiled` 是原 builder 的编译输入，逐文件 hash 固定，但不是逐行审计，也没有 dispatch 其中 attention/普通 scatter kernel。真正选取的 native pipeline 为 `tq_encode_half_hs{D}`，函数常量 80=K bits、81=K signed、90=V bits。

本地 C ABI launcher只编译自己的 Objective-C++，直接使用公共 Metal runtime compiler；传入的 shader body 不作改写，不分发到 tracked Wiki。MSL 3.1、`fastMathEnabled=NO` 单列，不声称与生产 extension 编译选项完全相同；不安装完整 Xcode、vLLM 或新 MLX，不加载与当前 wheel 不匹配的 `_paged_ops.so`。

现有 Python 3.11.16 / MLX 0.32.0 与上游精确 Python/MLX 要求不符；这轮能绕开 Python `searchsorted` 的原因是**执行原本就独立进行 centroid search 的原生 writer**，不是给 Python 编码器注入替代函数。Python V encoder 的完整资格仍未获得。

实际设备为 Apple M4 Max、64 GiB，macOS 26.6.2；runtime 身份在各 run 中记录。实验后的工具链核对为 build25G83、SDK26.5、Apple clang21.0.0，单独标为 post-run observation，不追认成每条 dispatch 的编译器 attestation。

## 3. writer 的布局合同比 reader 窄

输入 K/V 必须是连续 `[tokens,KV heads,D]` FP16。本轮只测 FP16 activation、真实 V3 codebook。生产 call site 会先 transpose/cast/`mx.contiguous`；这一准备成本不能从真实请求中删除。原 shader 也实例化 BF16/FP32，但本轮没有运行它们，不能把模板存在等同于精度/质量通过。

目的地址以**元素**为单位：

```text
slot < 0 → 整个 threadgroup 返回，不写 cache
page, offset = divmod(slot, page_size)
base = page*page_stride + (offset*KV_heads + head)*head_stride
```

kernel 只接收 K/V/scale 的 page/head stride；token stride 固定为 `KV_heads*head_stride`，坐标 stride 固定为 1。三个 metadata 使用同一 scale layout。materializer 却可以独立读取五个输入各自四维 stride；**reader 支持任意 view 不代表 writer 也支持**。本轮 padding 同时加在 head 尾部和 page 尾部，没有伪造 writer 支持任意 token/coordinate stride。

每个 backing 两端各保留 256 bytes guard，未写区域为 `0xa5`。slot 顺序 `[47,0,33,15,-1,17,5]`，覆盖倒序页、最后一页/最后一个 token、跨页与跳过 token。每次检查整个“本次目标写 footprint 的补集”，不只是最后几个 guard。

目的 slot 唯一；重复目标会使不同 threadgroup 写相同地址，不能假设有稳定的“最后 token 胜出”。本客户端在 GPU 提交前拒绝 duplicate slot，这是**本地安全边界**，不是上游 wrapper 已提供的能力。负 slot 的提前返回是 group-uniform，所有线程都不进入内部 barrier；不能照搬成任意 per-lane early return。

## 4. 原生融合真正省了哪些工作

一组 `(token,KV head)` 由 D 个线程组成；每 32 线程对应一个 scale group：

1. K 用 SIMD min/max，生成 scale/zero，量化并直接写出；低比特先存 threadgroup indices，再按连续 LE 位流组装 byte。
2. V 在前五级 FWHT 用 lane shuffle，跨 SIMD group 的级数使用 threadgroup scratch/barrier，最后归一化并 cast half。
3. 旋转后 V 的 group RMS 得到 half scale，centroid midpoint search 产生 V codes；V3 用真实八个 centroid。
4. centroid midpoint 只需每 threadgroup preload 一次，再显式 barrier 发布给所有 search 线程；不能依赖恰好存在的 K-pack/FWHT barrier。buffer 按最大 255 boundaries 分配，本轮 V3 仅用七项。
5. 五个 pool 直接按 slot 写入，省去 Python 编码结果与五次 paged scatter 的显式中间步骤。

这里“省去”来自源码数据流，不是本机测得五次 scatter 比一次 dispatch 快多少。没有 matched Python V3 encoder benchmark，也没有通过独立 counter 证明带宽/occupancy/barrier cycles。源码注释中的周期/吞吐说法不作为本机物理结论。

## 5. 一次真实失败揭示的 rounding 边界

最初使用独立 scalar FP16 RNE oracle，要求 K/V 所有 packed bytes 精确相同。D64 通过；D128 首代出现四个 K byte mismatch，实验如实 failed。加入差异坐标、输入范围、scale、zero 与 quotient 诊断后，观察到 K scale 可与理想 half division 相差一个 half ULP。例如：

```text
group min = -0.73046875, max = 0.94775390625
sample x = 0.360107421875, zero = 16
ideal: half(half(max-min) / half(254)) → scale 0.0066070556640625 → code 38
native observed:                        scale 0.006603240966796875 → code 39
```

另一个独立模型 `half(half(max-min) * half(1/254))` 正好得到后者。扩大到全部 24 代：理想模型有 11 代、27 个 K byte 不同；**half-reciprocal 模型的全部 K byte 都一致**，V3 packed bytes 也全部一致。metadata 仍以相同 `relative L2 <= 0.0011` 数值界检查，完整数据保留。

这是一个可否决的 **constant-division lowering 假设**，不是已经反汇编证明的根因；没有把旧理想模型偷偷改名为正确答案。两个模型都保留在 scalar lab，精确差异案例写入 CPU regression，evidence 保留原失败身份与新的并列比较。

另外执行未经改写的 `turbo_quant_encode_key`（它不需要 `searchsorted`）：在 58,368 个 K 坐标中，与 native 相比有 365 个代码不同；1,824 个 scale 中有 381 个不同，zero 全部一致。它是现有 wheel 上的原始声明执行，**不是已支持的 full-package 环境**。既不能从这些差异宣布 upstream bug，也不能继续宣称“Metal 是 Python 路径 bit-for-bit 的通用替换”。更不能拿本轮 decoder 对“actual bytes”正确的结果，抵消 encoder parity 未通过。

零 K group 使用最小正常 FP16 scale，常量 K group 使用 fallback；全零 V 的 scale=0，native 内部 `0/0` 的比较均为 false，得到 code0，decoder 输出零。因此“最终存储/输出有限”不等于 shader 所有中间值都有限。FP32/BF16 activation、极大幅值、subnormal 或其它 V codebook 仍需独立验证。

## 6. 原地 backing 与 lazy graph identity 是两种证据

原生 `TQEncodePrimitive` 的关键不只是 kernel：五个输出 `copy_shared_buffer` 输入缓存，使相同物理 buffer 获得**新的 graph provenance**。Python caller 重新绑定 `kv_cache`，当前 materializer 和下一 decode 的 writer 读新 handles，MLX 才能插入跨 command buffer 的依赖。绑定层通过新 array placeholder 的 descriptor 替换接到 Python；涉及私有 MLX ABI，说明精确 wheel pin 不是可忽略的版本装饰。

“旧 view 的地址没变，所以可继续读旧 graph node”不成立；物理共享不能表达 writer 发生前后的执行顺序。相反，上一轮仅替换 lazy metadata handles，也不证明 native cache 真的原地写。

本轮补了后半个证据：**自己的 native owner 内两代确实复用五个 MTLBuffer contents 地址**，并在各自 command 完成后检查 backing。第二代输入为 FP16 `K/2+0.125`、`V/2-0.25`；旧 decoded snapshot 对第二代的 relative L2 在 K 约 0.91–1.26、V 约 1.00–1.08，有限但失效。

随后刻意使用明确的、较重的桥：

```text
native write → waitUntilCompleted → CPU backing snapshot/copy
             → 新 MLX array → 重建 page/head padding view → eval/drain
             → 原始 materializer → MLX SDPA API → eval/drain → CPU 检查
```

新 MLX pool 与 native pool 不是同一 allocation。独立诊断 kernel 检查复制后实际 element strides 与原 writer family 一致；这证明 copied layout 正确，**不证明 zero-copy、生产 fence、external backing owner、MLX donation 或物理 overlap**。也没有将 CPU copy 的已完成依赖冒充未经同步的 native primitive 交接。

## 7. 消费者 oracle：mask 与 GQA 顺序不能省

六个有效缓存 token 按 source 输入顺序读取，物理页号并不表示时间顺序。Q 为三个 query、四个 query heads、两个 KV heads；lower-right causal mask 允许第一个 query 看四个 key，最后一个 query 看全部六个。CPU reference：

1. 独立按实际 native packed bytes/FP16 metadata 解 K/V，不调用 upstream decoder。
2. 按真实 materializer 的存储合同 cast FP16。
3. KV heads 按 GQA 分组重复成 `[kv0,kv0,kv1,kv1]`，用 FP64 QK/softmax/PV 计算完整输出。

两 seed 的 24 对 materializer 输出最大 relative L2 `2.17e-4`；384 个 SDPA 输出最大 `2.93e-4`。门限分别为 `0.001` 与 `0.002`。这比较的是**实际量化 bytes 代表的 K/V**，不是证明源 FP16 未量化 K/V 无损，更不是模型 perplexity/画质门禁。

有限负控使用同一 Q/decoded K/V：错误 upper-left mask 的 relative L2 约 0.41–1.41；错误交错 GQA `[kv0,kv1,kv0,kv1]` 约 0.30–1.07。它们不会因“输出 finite”而通过；没有提交 OOB/UAF 地址。

## 8. 时间证据仅回答本边界的问题

每代 consumer 在预热后按 `materialize+sdpa,ready-kv+sdpa,ready-kv+sdpa,materialize+sdpa` 做四个 block。两臂都全输出检查，计时包含 Python→`eval+synchronize`，不含 CPU oracle/readback。ready-KV 来自独立 decoder 的相同 FP16 K/V，是**省掉 materialization 的成本下界对照**，不是另一个 compressed attention 实现，不能用它得出 crossover。

下表为每个 seed/代各自中位数的范围，不跨不同 D 汇总一个速度分数：

| D / K | materialize+SDPA ms | ready-KV+SDPA ms |
| --- | ---: | ---: |
| 64 / signed8 | 0.205–0.242 | 0.171–0.202 |
| 128 / signed8 | 0.203–0.212 | 0.165–0.182 |
| 256 / K5 | 0.221–0.230 | 0.179–0.188 |
| 512 / K2 | 0.333–0.341 | 0.271–0.290 |
| 128 / unsigned8 | 0.198–0.209 | 0.166–0.169 |
| 128 / K4 | 0.198–0.211 | 0.166–0.172 |

N=6/Q=3 是 ABI/正确性的小消费者，不能取代真实长 context prefill/decode。不同 D 的 MLX SDPA 内部 dispatch 未独立 capture，API 名称不保证同一 fused kernel。结果中也没有新硬件最优 tile/threshold 结论。

native writer 另外记录单 command buffer 的 `GPUStartTime/GPUEndTime` 差与 native submit/wait wall；GPU span 约 0.011–0.032 ms。它在完成后读取，单位来自 Metal API 秒→毫秒；不是 CPU/GPU 原始 tick 相减，也不是 DRAM bytes/counter。只有 24 次、未作匹配优化 A/B，保留为诊断值，不宣传 writer 性能倍率。

native library/pipeline setup、host upload、copied bridge、consumer 各自分列。正式 run 的 setup 可能命中进程外 Metal compiler cache，不能叫隔离 cold compile；初次 smoke 的更长编译也不与热态平均。内存只记录 backing allocation bytes，不声称全链峰值/RSS/临时驻留已验收。

## 9. 复现、失败账本与下一步

CPU 独立模型、全部保存证据的检查不依赖外部仓库、不 import MLX、不自动启动 GPU：

显式 GPU driver需既有 MLX/NumPy、CLT 和固定 checkout，两个输出目录均须新建；全程不安装依赖或修改参考源码：

scope 只覆盖六个配置、四个 D、五种 K、FP16/V3，不是 Cartesian matrix。重复执行请顺序测第二 seed，不让自己的两个 GPU workload 并跑。原始日志/编译产物仅留 ignored 输出，文件/原 shader/assembled source/client/CPU oracle 在运行前后校验。整理工具 只读取明确成功的两个 final run；不挑最快历史。

历史保留四次 failed（一次本地 launcher 文件名 preflight 错误、三次严格 ideal-byte 比较失败）和两次早期 passed 的 raw hash；未把失败冒充 GPU 不可用或删除不利输出。后续数值完成不撤销 encoder bitwise gate 失败。11 项离线 tests 重算计数、完整误差、footprint、stride、源码身份和反例；可选 integration 移除时仅其身份检查显式 skip，CPU/evidence 继续运行。

接下来仍有实质工作：独立 Metal division 校准/IR 证据，受支持 Python/MLX 下的 native/Python codec parity；真正 production Primitive→materializer/paged attention 的无 CPU 交接；长 context 的直接 compressed attention 对 materialize+attention、相同 workspace/KV capacity、共享 prefix/mixed batch、BF16/其它 V 位宽；最后是完整模型质量、TTFT、timeline/counters 和 collector-off 请求。研究目标仍 active，既不被本次绿色 CPU tests 关闭，也不因一个未通过的 bitwise gate 丢弃已获得的 writer/consumer 证据。
