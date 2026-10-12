# 03 · 当前层 + 1–2 层前瞻的有界解码

[目录](README.md) · [GPU 候选](04-gpu-performance-and-convrot.md)

## 1. 配置语义必须无歧义

本节是拟议 `execution` 配置片段，不是当前产品 JSON schema。
完整配置、冲突规则及阶段能力以 [10](10-execution-and-product-integration.md) 为准：

```json
{
  "quantized_execution": {
    "schema_version": 1,
    "enabled": true,
    "mode": "bounded_dequant",
    "source_residency": "packed_resident",
    "prefetch_layers": 1,
    "decode_backend": "cpu_simd",
    "precision_profile": "z-source-mixed-v1",
    "granularity": "layer",
    "ane_compute": "off",
    "persistent_dense_layers": 0,
    "oversized_layer_policy": "reject",
    "allow_requantization": false
  }
}
```

- `prefetch_layers=0/1/2`：当前层之外提前准备的层数；默认 1，可设 2。
  R1 实际能力仅 0/1，2 在 R3 开放；上限由 capability 给出，不另加首版 max 参数。
- 同构整层执行时 dense slot 数通常为 `1 + prefetch_layers`；绝不能把“提前两层”
  实现为隐藏的三层之外再加无界 ready 队列。
- 此默认是新功能建议，不覆盖现有 layout-first 手动配置。手动布局不符合前瞻
  所需 slots 时 plan 明确拒绝，不能运行中改布局。
- 自动规划可在请求前依次考虑 2→1→0 前瞻和分片候选；选定后 identity 固定。
  请求中遇到压力按现有 guard 安全停止，不暗中重编 layout。
- 全模型 dense 展开不在该模式的候选集合；为研究测上限也要明确独立模式和预算。

R1 只支持整层，不足一层明确拒绝；第 7 节 tile 属于 R3。tile 模式的新 revision
以 `prefetch_units` 表达前瞻，不把单位悄悄从 layer 换成投影。自动规划也是后续能力，
首版不会按运行时可用 RAM 擅自搜索 2→1→0。

总预算、系统余量和guard使用现有request/streaming内存合同，不在上述子配置里
另造互相矛盾的“GPU预算”和“ANE预算”。未显式guard时仍遵守槽容量，但不承诺
整个进程的物理RAM硬上限。

## 2. 内存不是“模型大小 + 两层”

以现有 [内存合同](../streaming/05-memory-contract.md) 的唯一 backing/live interval
为准。每个阶段至少分账：

```text
packed source resident pages OR bounded packed read buffers
current + lookahead dense slots (field-wise aligned capacities)
CPU decoder scratch / GPU unpack scratch / transpose or derotate scratch
activations + attention workspace + conditioning + latent + output
ANE graph/model residency + input/weight/output surfaces + framework scratch
LoRA tensors and corrections, if qualified
allocator retained capacity + pending GPU/ANE readers
worker queues / metadata / process baseline / OS safety margin
```

文件 mmap 的虚拟大小不是 RAM；读过的 mapped pages 又不是免费。强制保留整套
packed MLX arrays 和 mmap 页可能造成额外驻留，必须分辨 alias 与独立副本。
不把 Core ML bundle 大小当作 runtime memory，也不把未知驱动内存记为零。

按兼容 slot field 求最大需求再相加，不能取“最大层总 bytes”代替异构字段容量。
checked arithmetic；metadata/scales、row pitch、K/N padding、split-K partial planes
全部纳入。源和目标若需同时存活，不能只记较大者。

### Z FFN 的可复核下限示例

H=3840、F=10240；gate/up/down 每张 H×F，BF16 每张 75 MiB，三张 **225 MiB**。

| 前瞻层数 | 同时 dense FFN 权重 | 若每层整块展开还需加什么 |
| --- | ---: | --- |
| 0 | 225 MiB | 当前层 attention、调制、小 tensor、解码 scratch |
| 1 | 450 MiB | 两层相应字段，不能只计算 FFN |
| 2 | 675 MiB | 三层相应字段，以及可能更深的 I/O buffer |

以上不是请求峰值。若 runtime ANE 另有一套全宽 FP16 FFN slots，至少再加约
225 MiB 的逻辑权重数据及 pitch/内部 staging。token-row 分割并不会减小 FFN 权重维度。
同一源 BF16 backing不能原样作为 FP16 解释；布局相同不代表 dtype 相同。

## 3. Source 两种模式，统一预算

1. **packed-resident**：预算足够保存完整压缩模型，但只能保存少量 dense 层。
   解码从压缩 RAM 读取；无需每步 SSD I/O，仍需重新解码被逐出的层。
2. **packed-streamed**：压缩模型也不能全部驻留。只读 fd + 有界 `pread` buffers，
   必要小 tensor 常驻；源缓冲和目标槽分别计入容量。

plan 显式选定；不能按“GGUF 只有几 GB”忽略 encoder、激活、VAE。
顺序读与随机 tensor slice 的真实 bytes/页放大分别记录；不通过全文件 `mlock`
或大预读抵消内存限制。

## 4. 扩展已有 slot 协议，不在 decoder 内自由分配

现有状态机仍由 `SlotSafetyTracker` owner 持有：

```text
Vacant → Loading(read + decode) → Ready → InUse → AwaitingFence → Vacant
```

CPU decoder 完成全部写入后由 owner 发布 Ready。GPU 初版待完成事件确认后才
Ready；若未来以 device dependency 提前发布，须独立协议 revision，不能以
“command 已提交”当 Ready 的数据保证。具体时序和资源所有权见 10。

每份 ticket 至少绑定 request/pass/step/component/layer/slot/generation，内容身份
还包含源文件身份、tensor type、目标 dtype、transform recipe、slice 和 LoRA recipe。

- owner 创建 backing/MLX arrays；CPU I/O worker 只写独占容量，不改变 MLX streams。
- GPU decode 由有明确 stream 所有权的调度线程提交，不能塞进原本只允许 `pread`
  的 worker 回调而绕过协议。
- GPU 和 ANE 都读某份 backing 时，以**所有最后 reader**完成作为复用条件。
- 异步模型输出必须拥有独立存储或显式 lease；不能指向下一层将覆盖的槽。
- cancellation/short read/decoder 失败先停止发布，再 join/drain；不能继续用旧内容。
- 不能证明 reader 完成时 quarantine；超时不能赋予提前释放权。

图编译必须把权重视为参数，不捕获可变 slot 内容为常量；禁止因更换 backing 地址
或层号反复编译。已有 safetensors pager 的“无转换”限制保留，GGUF 转换使用新的
capability/descriptor revision，旧 certified tuple 不继承资格。

Qwen3 bounded adapter 必须逐层提交并登记 reader；原四层 lazy eval/defer 开关
不能直接用于两槽模式。dense slot 可复用并不意味着 encoder tap/conditioning 可释放。
ANE 失败的 recovery live interval 与 normal 路径同样进入 plan，详见 10 第 4–5 节。

## 5. 前瞻时序与真实收益

```text
slot A: decode L0 → GPU/ANE reads L0 → fence → decode L2
slot B:       decode L1 ──────────→ reads L1 → fence → decode L3
```

依赖顺序仍为 L0→L1→L2。只预处理不依赖激活的权重；不能为跨 step 复用权重而
把不同 denoise steps 的同一层移到一起运行，后一步激活尚未产生。

令 `Tprepare = read + decode + required layout conversion`，`Tcompute` 为当前
完整层。理想独立流水吞吐接近 `max(Tprepare,Tcompute)`，但这是上限模型，不是
承诺。统一内存带宽争用、CPU cache、GPU 调度和 fence 会破坏理想重叠。

测量项：prepare active、ready wait、decode wait、exposed stall、GPU/ANE compute、
join、整个 layer/request。不能将重叠的 prepare 与 compute 相加后当 wall time。

前瞻 2 优于 1 的条件：存在 burst/I/O 抖动或层间差异，且额外槽不会挤出激活/ANE
内存。若 sustained decode 吞吐本来低于消费速度，增加一层只能缓冲，不能解决瓶颈。

## 6. Decoder 放哪儿：同时保留三个候选

| 候选 | 优点 | 风险 / 验收关注 |
| --- | --- | --- |
| CPU SIMD→共享 BF16/FP16 | 可与 GPU 大 GEMM 重叠；直接写 ANE slots | IQ 查表/转置慢、CPU争用、统一带宽 |
| GPU decode→dense slot | 容易融合 bit unpack/scale/cast/layout | 与当前 GEMM 争用 GPU；不假定两个 queue 真并行 |
| packed kernel | 无 dense 写回/槽，适合小 M | compute-bound 吞吐、scale 开销、格式专用实现复杂 |

不为所有格式预设同一赢家。decoder 与 GEMM 组合耗时才是选择依据；IQ 解码可能
适合 CPU/查表专用 kernel，Q8 SIMD 又可能被带宽主导。每种候选必须满足同一预算。

## 7. 一层太大：从整层退到数学正确的 tile

推荐顺序是投影级/输出 N 分片，不默认 K 分割：

- attention Q/K/V 可按投影准备，仍要保证所需激活生命周期和后续 attention。
- FFN intermediate-channel 分片：对同一通道集合计算 gate/up、SiLU 和乘积，
  解码 down 的对应输入列后，累加到完整 output。bias 只加一次。
- K 分片的 gate/up 部分和必须先完全累加，再施加 SiLU；不可逐 K tile 做 SiLU。
- down 的 K 分割可在 FP32 accumulator 中累加，最后一次转换；但浮点归约顺序
  改变，需要与原执行路线分开的数值资格。
- ConvRot 若保持旋转表示，down 的输入切分边界必须是完整 H256 group；非连续
  channel routing 不可随意打散旋转块。逆旋转权重路线另有独立 recipe。

即使投影/通道分片也不满足预算，返回具体不可缩减 floor；不分配全模型作为 fallback。
GPU fallback 同样受本预算约束，不能先保留失败的 ANE slots 再无条件展开 GPU 副本。

## 8. 跨 step、跨请求与自适应

默认 `persistent_dense_layers=0`。小循环缓存对顺序扫描多层通常没有跨 step 命中；
不把 LRU 命中率写成未经测量的收益。后续允许用户显式固定极少数 decode 很贵的层，
该内存与前瞻槽同计预算，并进入 layout identity。

跨 pass 预取第一层可减少启动泡，但必须使用现有明确的 carry 协议/新认证范围，
不能在 pass barrier 外偷偷保留 Ready slot。条件改变、LoRA/checkpoint 更换时内容
generation 失效；复用 shape graph 不等于复用旧权重。

最终评估既看低内存省了多少，也看 wall time。宁可报告“该预算下正确但慢”，
不能用 swap、隐式清缓存、少步数或少 token 换取虚假达标。
