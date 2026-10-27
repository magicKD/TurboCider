# 压缩 KV 的真正 Metal materializer：位流、FWHT、stride 与 fresh handles

<!-- knowledge
id: technique-metal-case-studies-compressed-kv-materialization
type: case-study
stage: prefill, decode
operator: attention, quantization, layout, codec
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: memory-bound, launch-bound, wrong-layout, stale-cache, precision-loss
semantics: numerically-equivalent, approximate, contract-only
representation: packed, q4, q8, fp16, bf16
boundary: producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU 导航](../README.md) · [上轮 serving 实现](../vllm-metal-paged-runtime.md) · [源码身份](../sources.md#materializer-sources-20261016) · [本轮验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，第四轮组件验证。**原封不动执行了 vLLM Metal 的 materializer shader 和两个 Python 包装函数**，没有把自己的近似 kernel 叫作上游实现。Apple M4 Max / MLX 0.32.0 上，160 组正确性配置、480 次 K/V 成对输出，以及另外 288 次正式配对计时均通过全元素 CPU FP64 检查。六个性能点的两个独立 seed 中，相对本实验 compiled 分解实现的配对中位倍率为 **1.34–2.88×**；这是 materializer-only host wall，不是 attention、TTFT、NAX 或模型质量结论。

另一个同样重要的结果是**完整编码器没有通过**：当前 wheel 缺少原始 Python 编码器使用的 `mx.searchsorted`；上游包本来要求 Python `>=3.12,<3.14` 和精确 `mlx==0.32.2`，而本实验使用已有 Python 3.11.16 / MLX 0.32.0。这里没有安装依赖、注入替代 `searchsorted`，也没有将组件通过扩展成受支持的 vLLM 环境。

后续[第五轮](native-kv-writer-and-consumer.md)已运行原生 `tq_encode` shader 与 copied-bridge 小消费者，补上 native 原地写的证据；它不撤销本页 Python encoder 的失败，也没有运行生产 Primitive/serving。发现的编码器逐位差异在新页单列，本页保留第四轮历史数字。

## 1. 执行了哪段代码，没执行什么

固定仓库 `vllm-project/vllm-metal@468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e`。七个本轮来源文件均核对 HEAD、工作文件 SHA256 与 commit blob；前后身份不变。核心入口是：

- [turboquant_materialize.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant_materialize.py)：完整 `_SOURCE`、`_kernel()`、`materialize_turboquant_pages()`。
- [turboquant.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/caches/turboquant.py)：格式、codebook、pack/unpack、signs、FWHT、Python encoder/decoder 合同。
- [turboquant.metal](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/turboquant.metal)：固定 signs、正逆 FWHT、旋转域累积；此原生文件仅阅读/比表，不编译执行。
- [sdpa.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/impls/sdpa.py)、[prefill planner](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/attention/impls/turboquant_prefill.py)：current writer handles、页去重、预算、restore、每层 drain，只有源码证据。
- [上游文档](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/docs/turboquant.md)、[依赖合同](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/pyproject.toml)：版本与上游校准范围；没有在本机复现其模型/HTTP 表。

显式 source integration用 AST 读取原始声明，`ast.literal_eval` 提取 `_SOURCE`，执行原包装函数，而不是 import 整个 vLLM。原始 body、grid、threadgroup、template 与 `ensure_row_contiguous=False` 保持不变；未把第三方 shader body 复制进 Wiki。来源为 Apache-2.0。选取的 codec 声明清单不等于全部函数都执行；V3 不进入昂贵的动态 Lloyd-Max 路径。

本轮没有执行原生 encode/scatter、上游 paged attention、scheduler、HTTP 服务或真实模型；没有下载权重、改变生产默认/系统设置、清理磁盘缓存。`mx.clear_cache()` 仅在未计时的 memory probe 清理本进程 MLX allocator 的空闲设备分配，不删除用户缓存文件。

## 2. 最容易混淆的存储合同

| 张量 | 逻辑形状 | 存储与含义 |
| --- | --- | --- |
| `k_data` | `[P,S,H,D*Kbits/8]` | K8 可为 signed int8 或 unsigned uint8；低比特为 uint8 位流 |
| `v_data` | `[P,S,H,D*Vbits/8]` | unsigned centroid index 位流；本轮仅 V3 |
| `k_scale/k_zero/v_scale` | `[P,S,H,D/32]` | FP16；每 32 坐标一个 group |
| `pages/offsets` | `[N]` | 本轮用 uint32；分别是 scheduler 物理页及页内 token；不是展平 token 地址 |
| `centroids/signs` | `[2**Vbits]` / `[D]` | codebook FP32；sign 由 key42 与 D 生成 |
| `k_out/v_out` | `[N,H,D]` | FP16/BF16；两者合计 `4*N*H*D` 个逻辑字节 |

支持的 `D` 为 64/128/256/512。元数据和 packed pool 各自有四维元素 stride，不能把 shape 当地址，也不能混入 byte stride。非零 view base 由 MLX 传递给 kernel 的输入指针承担。

K 的公式是 `(code + zero) * scale`，不是常见的 `(code - zero) * scale`。这里 `q8_0`/`int8` 指 signed int8 + 独立 affine 元数据，**不是 GGUF Q8_0**。8-bit 分支依赖输入 Metal 类型保留符号；把 signed 字节强转 uchar 仍会产生有限但错误的 K。

低比特按 little-endian **连续位流**打包，不是每 byte 留下未用位。例如 V3 的 `[0,1,2,3,4,5,6,7]` 为 `88 c6 fa`；第 3、第 6 个代码跨 byte。忽略第二个 byte，在 2/4-bit 上可能侥幸正确，却在 3/5-bit 上失效。原实现仅在 `shift+bits>8` 时读取下一 byte；最后一个代码自然不越界。

单坐标的 K/V 压缩存储（不计页表/对齐）为 `(Kbits+Vbits)/8 + 6/32` bytes；K8/V3 为 1.5625 bytes，相对 FP16 K/V 的 4 bytes 是 2.56×。解码后仍要额外保留最终 FP16/BF16 K/V；压缩容量收益不能直接当瞬时峰值收益。

## 3. V 的顺序与真实并行结构

设 `H_D` 为归一化 Hadamard 矩阵，`S` 为对角 ±1 矩阵：

```text
encode: x → S*x → H_D*(S*x) → group RMS → centroid index → bit packing
decode: packed → index → centroid*stored_scale → H_D*r → S*(H_D*r) → output cast
```

逆变换不是重复正变换顺序；虽然 `H_D²=I`、`S²=I`，通常 `H_D*S != S*H_D`。FP16 元数据、FP32 centroid/运算与最终 FP16/BF16 舍入也是合同的一部分。

上游 fused kernel 把一个 `(token,KV head)` 向量交给一个 32-lane SIMD group；每 lane 持有 `E=D/32` 个值：

1. 从原 stride 读 packed bytes、scale、zero；K 直接写最终输出。
2. V centroid lookup/rescale 放入 `values[E]`。
3. mask 1/2/4/8/16 用 `simd_shuffle_xor` 完成 lane 间 butterfly。
4. 高位 stage 在同一 lane 的不同 `e` 间交换，先写 `next[E]` 再整体 commit，不能边读边覆盖伙伴。
5. 乘 `1/sqrt(D)` 和 `signs[d]`，只做一次最终 output cast。

这是源码的显式寄存器数组设计；本轮没有反汇编/counter，**不据此保证编译器从不 spill、所有读都是免费或某条指令只花一个 cycle**。真正验证的是输出语义、原始 stride 与 allocator 可见的额外分配。

四种 D 的当前 MLX key42 signs 与 `.metal` 常量表均逐项一致（两进程各 960 项，零 mismatch）。哈希留在 evidence；它是本 wheel/源码的兼容观测，不是所有未来 RNG 实现的保证。

## 4. 为什么不能把本实验倍率当 compressed attention 的倍率

源码中有两条不同的优化目标：

| 路径 | 主要取舍 | 本轮证据 |
| --- | --- | --- |
| 直接 compressed decode | 按 token 解码 centroid/scale，在旋转域累积，最后对输出做逆 FWHT | 读原生 helper；未运行完整 attention |
| bounded materialized prefill | 每个引用的 K/V token 解码一次，生成普通 K/V，再交给矩阵化 attention，让多条 query 复用 | materializer 真实运行；后续 attention/dispatch 未运行 |

若所有 token 共享同一 `F=S*H_D`，则 `sum_i w_i*F(r_i) = F(sum_i w_i*r_i)`。这里移动的是**解码后线性逆变换**，不是把 centroid lookup、分组 scale、量化、softmax 也移到求和后。权重仍由 K/Q 决定。CPU 合同验证这项线性等价；浮点累加、partition scratch dtype 和最后 cast 会改变舍入，不能承诺 bitwise。

原生 [paged attention](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/kernels_v2/pagedattention.metal)的 selected branch 在非分区输出或分区 reducer 后调用逆 FWHT；身份见既有 [deepdive manifest](../sources.md#deepdives-sources-20261016)。不要把“每输出向量少做逆变换”简化成所有 head/partition 场景通用的精确操作次数。

因此必须另测 `compressed attention` 对 `materialize+attention+drain` 的 crossover。上游 planner 按当前 chunk **新增 query 数**、历史 KV 长度、geometry、encoding 和 workspace 选择，不是按 prompt 总长度全局打开。共享物理 prefix 可去重，独立历史仍相加；每个 layer、每个 scheduler step 重做 materialization，不缓存永久 decoded prefix。源码中的阈值和 M3/M5 校准不能继承为 M4 的验收。

## 5. 正确性矩阵与独立 oracle

两进程分别 seed17、29；顺序运行，未让自己的两个 benchmark 同时争抢 GPU。每进程 80 case：4 个 D × 5 种 K（signed8、unsigned8、K2、K4、K5）× 2 输出 dtype × 2 layout。V 固定真实八项 V3 codebook，不把随意 V5 table 伪装成上游 Lloyd-Max。N 按 D 分别取 1/7/33/65，KV heads 为 1 或 3，覆盖单 SIMD group、非整 threadgroup 尾部和较宽寄存器向量。

strided 布局在 pool 和三个 metadata 的所有四轴上做 `1::2`，原 backing 用有限 sentinel padding；每个 pool 都有非零 base。独立诊断 kernel 实际读出 Metal 接收到的 strides，而不是仅相信 Python slice 标签。对于长度为 1 的轴，MLX 可规范化无地址影响的 stride；记录差异，只在 `size>1` 的轴要求与预期一致。head=1 时该轴始终乘 0，这不是放宽有效地址检查。

路由包含非单调页序、最后一页/最后一个 offset、重复 token；大 N 也会重复读取有限物理池。本轮验证重复路由的值正确，**不宣称 integration 执行了 upstream planner 的页去重**。

CPU FP64 oracle 用 NumPy bit-plane 解包和普通 butterfly，不调用上游 decode。bit-plane 结果先对独立逐 bit 标准库实现核对；FWHT 对标准库 scalar 参考核对。三臂都以全输出相同 oracle 验证，计时后才读回到 CPU；不是只抽样、只比较 fused/eager，或只测试 finite。

正确性矩阵最大 relative L2：FP16 K `3.26e-4`、V `2.82e-4`；BF16 K `2.40e-3`、V `2.15e-3`。门限分别 `7e-4` / `5e-3`，输出直接对 decoded FP64，而非原未量化模型 K/V。量化本身的损失没有被这些误差数字覆盖。FP16 scale=0 group 单列输入覆盖，不能据此证明编码器的极值/溢出质量。

## 6. 三臂热态性能：匹配 host-ready 边界

`fused` 是原始 wrapper；`eager` 是本地 gather→unpack→centroid/scale→MLX Hadamard→cast；`compiled` 是同一分解函数的 `mx.compile`。后者给出更强控制，但不等于所有引擎中最优分解器。三个 arm 输入都已驻留、完全相同的 view 与路由。每点预热三次，每 block 顺序 `eager,fused,compiled,compiled,fused,eager`，四个 block；分别计算 block 内 mean 比值，再取四个比值中位数。

两 seed 的范围如下（不是置信区间）；所有点 KV heads=2、V3。毫秒是各 arm 热态样本中位数，倍率是上述 paired median，不能用两列中位数之比代替。

| N / D / K / dtype / layout | fused ms | compiled ms | eager/fused | compiled/fused |
| --- | ---: | ---: | ---: | ---: |
| 33 / 128 / signed8 / FP16 / contig | 0.181–0.184 | 0.246–0.247 | 1.74–1.77× | 1.34–1.35× |
| 257 / 128 / K5 / FP16 / strided | 0.173–0.175 | 0.262–0.263 | 2.03–2.07× | 1.52–1.53× |
| 4097 / 128 / signed8 / FP16 / contig | 0.276–0.303 | 0.548–0.584 | 2.76–2.81× | 1.94–1.98× |
| 4097 / 128 / signed8 / FP16 / strided | 0.265–0.284 | 0.544–0.572 | 2.84–3.04× | 2.01–2.02× |
| 8193 / 128 / K4 / BF16 / strided | 0.358–0.366 | 0.994–1.035 | 4.41–4.60× | 2.73–2.88× |
| 1025 / 512 / K2 / FP16 / contig | 0.321–0.323 | 0.645–0.648 | 3.14–3.33× | 2.00–2.03× |

计时包含 Python 调用、图构建/提交、`mx.eval` 和 `mx.synchronize`；不含输入/CPU oracle 准备、host 输出转换与误差计算。共 288 次正式计时调用，即 576 个 K/V tensor，不把 480 次另外的正确性调用或 warmup 混入热态样本。完整矩阵各臂的 `first_observed_ms` 单列；它们可能命中前面 case 的 JIT cache，**不是隔离的 cold compile latency**。

没有独立 command-buffer GPU timestamp、counter、功耗、热状态控制或后台 GPU 静默保证，短点很可能受 host/launch 成本影响。两 layout 点也使用不同随机 pool，不能将表中几微秒差异解释为 strided 比 contiguous 更快。没有把 M4 的普通 GPU 路线标成 M5 NAX。

## 7. 峰值内存与 freshness：分别证明什么

未计时的单次 probe 在 drain 后清本进程空闲 allocator cache，记录 baseline active、peak delta、retained delta，并保持输出 live。最大 N8193/D128/H2/BF16/K4 点：最终 K/V 逻辑大小约 8.001 MiB，fused peak additional **8.031 MiB**，compiled **42.347 MiB**，eager **91.241 MiB**；两 seed 相同。它支持“避免显式 context-sized packed/FP32 中间数组”的源码设计，但不测总 RSS、全模型驻留或硬件 spill；allocator 粒度也使实际 allocation 高于逻辑 bytes。

freshness 使用同一 `pages/offsets` 对象，初始 strided K8/V3 pool 后生成新的 lazy metadata handles：`K_scale/2`、`K_zero+1`、`V_scale/4`。新 handles 不先 eval，原 materializer 必须追踪 producer→reader。两代输出均对独立 FP64 和原始 Python decoder 通过；旧 decoded snapshot 对新期望的 relative L2 为 K 约 1、V 约 3，有限但明确错误。

这个控制证明**路由复用与数据新鲜度是两件事**，不是 native encoder→external backing 原地写、零 copy、CoW 或 DLPack owner 生命周期证明。packed bytes 未变，只变三个 metadata handles；原生 cache writer、原地 byte 更新及真实 prefix planner 仍待单独验证。

其他有限负控：丢失跨 byte 高位在 128 项 V3 中误解 19/24 项；将 signs 错放到 inverse 前造成 relative L2 1.41/1.32；标准库合同的错误 contiguous stride 也产生有限错误。没有故意向 GPU 提交 OOB/UAF 地址来制造负例。

## 8. 失败历史与版本边界

初次 smoke 的 materializer/性能已成功，但随后的**未经修改上游 Python encoder**报 `AttributeError: mlx.core.searchsorted`，整次原报告标为 failed。查 `pyproject.toml` 后确认该组件环境不是上游 pin；没有把这个错误称作上游在受支持版本的缺陷。后续把 encoder attempt 状态与 synthetic materializer 数值状态分开；最终两进程继续保留 `original_encoder_attempt=unsupported`，不是重新包装成 encoder pass。

第一次完整矩阵还暴露了客户端 stride guard 错误：对长度为 1 的 head 轴也硬性要求 slice stride，导致拒绝合法 view。修的是本地 guard；有效地址轴检查保留。两份 failed 报告和三个早期 passed 报告的 hash、probe hash、排除原因均在 evidence；最终统计仅取同一 probe 的两个 final 进程，不挑选最快历史。

## 9. 可重放入口与下一步

不需要外部引擎的 CPU 模型和证据检查：

显式设备实验需现有 MLX/NumPy、审核过的固定 checkout；下面的路径和输出由调用者明确选择，`--output-dir` 必须是新目录。不自动 clone/install/model download。

下一步是更强消费者：在受支持且明确准备的环境中核对 native encoder/scatter→materializer→attention、原 cache backing、prefix 去重和 mixed-batch workspace；比较完整 compressed attention 对 materialize+attention，在相同 KV budget 下确认 crossover/峰值/模型质量。V2/V4/V5/V8、动态 Lloyd-Max 初始化、真实模型 TTFT 和独立 timeline/counters 仍未验收。新的 wheel/RNG/SDK/SoC、encoding、view layout、page lifetime、consumer 或 compiler 修订都需要重跑，不能沿用本表授权生产。
