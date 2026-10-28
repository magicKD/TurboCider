# 真正的 MLX native 交接：精确 ABI、共享 backing、新 graph handle 与观测边界

<!-- knowledge
id: technique-metal-case-studies-mlx-runtime-handoff-and-observability
type: case-study
stage: prefill, decode
operator: attention, quantization, layout, scheduling
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: compile-bound, lifetime, stale-cache, wrong-layout, measurement, precision-loss
semantics: numerically-equivalent, approximate, contract-only
representation: fp16, bf16, q8, packed
boundary: producer-consumer, cpu-gpu
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU 导航](../README.md) · [上一轮独立 launcher](compressed-attention-crossover.md) · [原生 writer](native-kv-writer-and-consumer.md) · [来源](../sources.md#mlx-runtime-sources-20261016) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，第七轮。本轮不再用独立 Metal C ABI 或 CPU copied bridge 代替 vLLM 的 MLX Primitive：**原始 `CacheViewPrimitive`、`TQEncodePrimitive`、`PagedAttentionPrimitive` 均已真实构建/执行**，writer→materializer/attention 之间无显式 CPU 复制、无中间 `eval/synchronize`。两个进程、FP16/BF16、16个配置，256个正式完整输出通过独立 CPU FP64 检查；增量 cache 最终字节与 fresh arena 的全量 native encode 一致，原 backing 变化与 padding 也通过。

这不是完整 vLLM 运行：未安装/启动 engine、worker/scheduler、模型或 HTTP；高层 TQ prefill admission 没有运行，materialization arm 是研究客户端显式选择。精确 native 依赖匹配、实际 native kernel dispatcher 与整个 serving 请求资格是三件不同的事。

## 1. 从版本缺项到精确 native build：没有升级产品环境

前几轮使用 Python3.11/MLX0.32.0，原始 Python V encoder 缺少 `searchsorted`；不能在那里加载要求另一版 private C++ ABI 的 `.so`。本轮发现本机已有 Python3.13.12 解释器，使用它创建**新的 ignored 任务 venv**，从 PyPI 固定安装：

| 组件 | 本轮版本 / 身份 | 意义 |
| --- | --- | --- |
| Python | 3.13.12 | 落在上游 `>=3.12,<3.14` 范围 |
| MLX / mlx-metal | 两者均0.32.2 | 与上游精确 MLX pin 一致，不拿 patch 版本近似兼容 |
| nanobind | 2.10.2 | 与上游 pin 一致，绑定涉及 `inst_ptr<array>`/descriptor |
| NumPy | 2.5.2 | 明确固定的 CPU oracle/数据生成环境 |
| 硬件 | M4 Max/64 GiB，检测到40 GPU cores | 两进程；NAX supported/ready 均 false |
| native artifact | 原始 input digest、自己的 `.so`、libmlx/private headers SHA | 源码/依赖身份，不是 reproducible-build 或 IR 证明 |

新 venv 约250 MiB，没有修改原 `.venv`、产品 `Python` 或基础解释器的包。没有下载模型、安装 Xcode、清理用户缓存或调整系统设置。新环境的 `searchsorted` 实际存在，原始 Python V3 encoder 能执行；这解除的是旧环境功能缺项，**不解除下文的 encoder parity 差异**。

运行后另行核对的 native 编译工具为 Apple clang21.0.0、SDK26.5，OS build25G83。它是 post-run 工具链快照，不回填为每个 shader JIT 的独立 attestation；本轮使用 MLX 原 source-mode 默认，没有给 runtime 手工重设 MSL/math policy。

固定 [vllm-metal commit](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)，19个文件匹配 commit blob，前后不变。AST 执行 [build.py](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/build.py) 的原始 resolver、`_BuildSpec/_build_spec/_input_hash`；**只将 `_OUT` 指到任务目录，编译 flags 与 C++ body 不改**。不调用会写原 checkout stamp 的整个 `build()`，也不触发完整安装脚本。

编译直接使用 wheel 的 private headers/libmlx、nanobind source、原始 `paged_ops.cpp/mlx_patch.cpp`；先 import 当前隔离环境的 `mlx.core`，再加载自己的扩展。真实 shader 由原始 builder 组装200,279 bytes，通过 `init_v2_library` 进入 MLX 的 source-mode cache。没有依赖不可用的离线 `metal` 工具或伪造预编译 metallib。

这是**native 组件构建要求匹配**，不是 pyproject 全依赖已安装或 full package 通过。MLX 的包路径、版本、头文件、libmlx 与扩展 hash 都保留；不从 version 字符串单独推断任意旧 binary 可以复用。

这里的 private ABI 指 MLX 内部 C++ 实现接口，不是私有 ANE API；没有关闭系统保护或申请额外系统权限。

## 2. 真实执行链与仍未执行的层

```text
MLX backing → 原始 CacheViewPrimitive（原 page/head stride）
  → 原始 TQEncodePrimitive（suffix encode + in-place scatter）
  → 返回五个 fresh graph handles，客户端 rebind
      ├─ compressed：原始 PagedAttentionPrimitive（TQ=true）
      └─ materialized：原始 materializer → reshape view
                       → 原始 PagedAttentionPrimitive（普通 K/V）
  → eval+synchronize → 计时外完整 CPU 检查
```

主代码：显式 integration。它没有重写 upstream Primitive，只选择原始编译/codec/builder 声明，不 import 整个 vLLM Python 包。writer 与 reader 中间不提前 eval、不通过 NumPy，也不提交另一个独立 Metal queue；上游 MLX evaluator 负责这条图的顺序与资源生命周期。

输入、arena 初始化和 CPU reference 准备在计时前完成；这不是从模型 embedding/projection 开始的 full layer。高层 worker 预算、prefix 去重、mixed request restore、生产 callback、外部 Torch/DLPack owner、取消/回收均未验收。M4 高层默认是否允许该 TQ prefill lane，也不能由手动选择 materialization arm 来证明。

## 3. shared backing 不等于 fresh graph provenance

[mlx_patch.cpp](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/mlx_patch.cpp)直接创建共享 view，避免公共 `as_strided` 先 flatten 大 arena 的历史问题；flags/span/offset 以元素计算。边界和非重叠写仍是 caller 责任。本轮只测小型安全 arena、非负 stride、offset0；不把它扩大成5 GiB/int32 flatten 极限通过。

[TQEncodePrimitive](https://github.com/vllm-project/vllm-metal/blob/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e/vllm_metal/metal/paged_ops.cpp)让五个输出 `copy_shared_buffer` 输入缓存，物理数据复用，但 graph identity 指向新 writer。caller 必须保存新 handles，并把它们传给当前 reader 和下一步 writer；Python 对象不同不自动证明依赖正确，地址相同也不自动建立依赖。

本轮同时验证三层：

1. 每次 writer 返回的五个 Python handle 都不同于旧对象，正向 reader 只接新输出。
2. 一次全量初始写、反复 suffix 更新后，从**原 backing**重新读到与最终 fresh-arena payload 相同的值；所有未写 page/head padding 仍是 sentinel7。
3. 16个配置、交替两代 K/V/Q 的完整 attention 输出匹配独立 CPU reference；不是只测试最后 cache finite。

没有导出 Metal buffer 地址，也没证明“整个请求所有中间数组零 copy”；这里是原始共享实现的源码证据、backing 值别名观察与正向图交接证据。没有故意提交缺少 writer 依赖的旧 graph handle 去制造 GPU race。安全负控使用独立旧 decoded snapshot：相同新 Q 消费旧 K/V，relative L2 约0.028–0.920，明确有限但失效。

离线 DAG/descriptor 模型强调：两个 view 可共享 allocation，但旧 node 没有新 writer 的 dependency path；它不是 GPU fence 模拟器，不因 CPU case 通过授予硬件资格。

## 4. reference 不把 encoder、增量状态与 attention 混为一谈

两代 suffix 使用相同 slot map：第一代原 K/V，第二代 `K/2+0.125`、`V/2-1`；Q 也交替两套。物理页非单调，page/head 有0或4个元素 padding，metadata 与数据各有自己的 pitch。每个配置一条序列；不验证跨请求 shared prefix、任意 token/coordinate stride 或 offset 非零。

为防止“错误 writer 与错误 reader 互相自洽”，第二代参考不从待测增量 cache 现读生成，而是：

- 将同一完整新 K/V **全量 encode 到另一个 fresh arena**，定义该 native encoder 的正确状态 bytes。
- 独立 CPU bit-plane unpack、FP64 FWHT、signed K affine、GQA causal QK/softmax/PV；bit/FWHT/BF16 rounding 对标准库 scalar 控制核对。
- native compressed 的 V 目标不做每 token FP16/BF16 cast，materialized 目标保留该存储舍入；两者不是 bitwise 契约。
- 正式序列结束后，再对每 arm 全量 cache bytes/metadata、原 backing、padding 与 fresh-arena state 对照，全部零 mismatch。

fresh-arena encode 是**增量状态 oracle**，不是独立 encoder 正确性/质量 oracle。CPU decoder/attention 独立；encoder 与原始未量化模型间的质量仍需模型门禁。最大 relative L2：FP16 compressed/materialized约 `2.99e-4/3.05e-4`，BF16约 `1.70e-3/1.70e-3`；门限分别0.004、0.015。256个正式输出全部检查所有元素，没有只用两条 GPU arm 互相对照。

## 5. 原始 Python encoder 现在可运行，但仍不能宣称逐位等价

在精确 MLX0.32.2 环境调用未经修改的 `turbo_quant_encode`（V3，无 `searchsorted` shim），全量比较 native initial cache。16个配置合计不同项：K代码/byte **1,717,682**，V packed byte **50,300**，K scale **288,628**，V scale **429,602**，K zero **14,121**；这些列的单位不同，不拼成一个“错误率”。BF16 activation 对照占较多差异。

原生 writer 显式 half intermediate/float reduction，Python 图可能保留输入 activation dtype、不同 fusion/FWHT/RMS 舍入；前轮也验证过常量 division 的 source-expression 差异。这里没有归因到某个单一编译器 pass，也没有对所有差异做质量测量。因此：

- 旧环境“缺少 `searchsorted`”已经在新隔离环境解除；旧记录不改写为当时也能运行。
- native/原始 Python 的 exact-byte gate **仍不通过**；不能再仅以旧 wheel 不匹配解释所有差异。
- 两条待测消费者都使用同一 native encoding 与 fresh-state reference，性能比较没有混用两种 encoder；数值通过也不抵消 encoder parity/模型质量未通过。

## 6. 原始 dispatcher 的实际观察：capability 不等于选择

诊断只在 GPU idle 时短暂启用 `_set_paged_dispatch_diagnostics`，各 arm 做一次 eval 后立即关闭/清空。正式计时不开 observer，结束后查询为空；它是 process-wide last-family 观察，不是每 request trace 或硬件 counter。

| 几何 | compressed 实际 family | materialized 后普通 K/V 实际 family |
| --- | --- | --- |
| D64/N257/Q1 | `per_token_ps0` | `per_token_ps0` |
| D128/N513/Q17（两 dtype） | `per_token_ps0` | `tiled_prefill` |
| D128/N8193/Q1 | `per_token_ps512` | `per_token_ps512` |
| D128/N8193/Q64、128（两 dtype） | `per_token_ps0` | `tiled_prefill` |

这些来自**未改写的 native C++ dispatcher**，没有调用强制 partition test entry 或覆盖 GPU core count。虽然 capabilities 中有 `gqa_decode=true`，本矩阵没有观察到该专用 family；不能用 capability 表把实际 `per_token_ps512` 改名为 GQA 专用 kernel。NAX supported/ready 均 false，不把 tiled prefill 标成 TensorOps/NAX。

这比上一轮的手工 ps0/ps512 launcher 更接近真实执行，但仍不是高层 TQ admission 自动选择展开：客户端手工选择两个 arm，只有进入 native 普通 attention 后的 kernel 选择是原始 dispatcher 决定。

## 7. 同一 runtime 的热态边界：包含 suffix writer

每点4 block，`compressed,materialized,materialized,compressed`，block 交替输入0/1。resident suffix writer→consumer 图构建、MLX submit、`eval+synchronize` 都计入；CPU oracle、输入准备、reference readback 不计。验证转换为 FP32 本身也能发 GPU 工作，最终版本在验证后及下一次计时前额外 drain，防止落入下一 arm 的边界。两份未加强该边界的早期全矩阵只保留为 preliminary，不挑最快混入最终表。

高层 compressed 路径通常继续保持 lazy；本客户端选择两臂相同的输出完成边界来评价组件，不建议因此给生产 compressed 每层增加强同步。完整模型的 batching/提交与 per-layer materialization drain 仍须按真实调度另测。

下表为两个 run 各自 median 的范围；倍率是 block 内 mean 比值的 median，不等于两列 median 相除。所有点8Q/2KV heads；除首行外 D128。完整正式输出256个、每 arm128。

| dtype / KV N / Q | compressed ms | materialized ms | paired compressed/materialized |
| --- | ---: | ---: | ---: |
| FP16 / 257 / 1（D64） | 0.235–0.242 | 0.238–0.258 | 0.96–0.97× |
| FP16 / 513 / 17 | 0.274–0.283 | 0.417–0.429 | 0.65–0.66× |
| FP16 / 8193 / 1 | 0.319–0.414 | 0.384–0.407 | 约0.87× |
| FP16 / 8193 / 64 | 2.151–2.189 | 1.878–1.883 | 约1.15× |
| FP16 / 8193 / 128 | 3.892–4.009 | 1.842–2.233 | 1.88–2.12× |
| BF16 / 513 / 17 | 0.237 | 0.324–0.327 | 0.72–0.73× |
| BF16 / 8193 / 64 | 2.147–2.606 | 1.943–2.508 | 1.05–1.12× |
| BF16 / 8193 / 128 | 3.890–3.899 | 1.936–1.954 | 约1.99–2.00× |

小差异和单 query 的长尾波动不能当阈值依据。没有后台 GPU 静默/热状态控制，四个 block 不是置信区间。Q128 的局部优势在两个 dtype/seed 出现，仍不能直接成为模型 TTFT/吞吐或全局 cutoff。

与第六轮不能做版本升级速度排名：这里换了 Python/wheel，数据来自真实 native encoder、每次包含 suffix 写，candidate 用原生 tiled attention 而非 MLX SDPA，scratch/output 分配也由原始 Primitive 管理。materializer 为满足 page reshape 还 gather 整个 kernel page：8193 logical tokens 对应8208行，而不是上一轮只 gather8193行。

source/header/lib 编译、首次 library init、初始 cache 全量 encode、CPU arena/golden reference 另列，不混入热态。最后两次 native compile约2.86秒，已可能命中文件/编译缓存，不称隔离 cold build。没有完整链 RSS/峰值或相同模型 KV capacity 的对照。

## 8. 真实 capture 再次反证“解析成功就有设备时间”

单独第三进程，seed41、D128/N513/Q17/FP16，在正式 collector-off run 之后做两个小窗口；child-only `MTL_CAPTURE_ENABLED=1`，两端 drain，窗口内真实执行新 writer→consumer 图，捕获输出也对完整 CPU reference 通过。没有 viewer、replay、Xcode profile 或全盘 source scan。

尽管工作量小，每份 capture 仍有60个文件、约189 MB，因为库与资源资产也被封装；两份原件约361 MiB，仅留 ignored 输出，不进入 Wiki。不要按“只抓一层”假定磁盘成本必然很小。

复用 [先前审读的 gputrace](../../sources/repos/gputrace-timing-provenance.md) `3d7d32e...` 与原 owned binary hash，只运行只读 `stats --json`、`timing --json - --table=false`，结果：

| 观察 | compressed capture | materialized capture | 可解释范围 |
| --- | ---: | ---: | --- |
| parsed command buffers | 1 | 1 | 解析出的结构 |
| parsed compute encoders | 1 | 3 | 结构差异，未证明每 kernel 端点/耗时 |
| parsed dispatch calls | 0 | 0 | parser 覆盖未验证，绝不是物理零计算 |
| `unique_kernels` / timing rows | 17,319 | 17,319 | 包含库函数/identifier，不是实际执行数量 |
| `timing_source` | synthetic | synthetic | 名称驱动的预测 |
| reported total | 27.63 s | 27.63 s | **不是 GPU duration**，两份实际链不同却同总数 |

本轮没有取得硬件 counter/可靠 kernel timeline。既有 normalizer将17,319行逐项保持为 predicted，physical GPU duration 和 dispatch count 都为 null；公共 capsule 仅保留计数/出处/hash，不导出原始 UUID、路径或17k事件列表。

原始源码的预期链与 dispatch diagnostics 已知，捕获输出也数值正确；仍不能把两者反向填到 parser 的空白 dispatch 计数里。新增1/3 encoder 结构观察是进展，**不是已解决独立 source→device timeline 对齐**。

## 9. 可重放、失败分类与后续

CPU core 和保存证据无需 MLX/外部 checkout：

先显式准备新的隔离环境并固定上表依赖，再使用其中解释器；脚本不自动安装、clone 或修改 source。`TC_EXACT_PYTHON`、`TC_VLLM_METAL_CHECKOUT` 由调用者指定，output目录必须新建：

只读选定 run/receipt/parser JSON，不运行 GPU/联网。10项普通测试重算全部 pair ratios/输出计数与误差，检查共享 backing/padding、fresh-state oracle、parity 差异、实际 dispatcher、捕获未知值及本地身份。移除 integration 后仅身份检查显式 skip，CPU 模型和 capsule 可独立使用。

[版本登记](../../guides/knowledge-base.md)绑定本轮 source/evidence SHA 与精确环境；换回 MLX0.32.0、换 SoC/SDK/编译器或30天到期都需要重审。即使查询环境恰好匹配，也不会自动授予性能/兼容资格。

最初 smoke 在真实 native 图执行后因客户端把 `last_paged_dispatch` 写成 `_last_paged_dispatch` 而 failed，原报告与 hash 保留；不是 ABI/GPU 失败。另有短 smoke、两份验证边界较弱的早期完整数值记录也单列，不删除或混入最终256样本。

下一步继续：将精确 native runtime 放回完整 worker/admission/真实模型，同预算比较 KV/质量/TTFT；验证外部 backing、跨 stream/mixed requests/取消与峰值；补齐可靠 profiler streamData/时间域与 kernel 对齐，或有计划地准备更完整工具链。MLC/TVM model、tinygrad BEAM、其它引擎和更多 PR 也仍有研究任务。本轮解除的是实际 Primitive/同 runtime handoff 的缺项，不把它或绿色 tests 改写成整个持续 GPU 研究完成。
