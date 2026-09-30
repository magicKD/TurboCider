# 06 · 静态与 runtime ANE 的 W8A8 设计

[目录](README.md) · [实施与验收](07-implementation-and-acceptance.md)

## 1. 三条路线分别命名

| 路线 | 权重在哪里 | 运行时实际目标 | 当前状态 |
| --- | --- | --- | --- |
| static W8A8 | checkpoint-bound Core ML artifact | 主要 GEMM W8A8，其他算子浮点 | 已有非原生 ConvRot 研究路线，不能泛化其资格 |
| runtime FP16 | 每层填入 FP16 slots | FP16 MatMul/SwiGLU | 已实现，可扩展有界 GGUF decoder/ConvRot数学 |
| runtime W8A8 | 每层更换 codes/scales 的通用图 | 动态权重仍由 ANE INT8 执行 | 未证明；本设计单独设置可行性 gate |

`cpuAndNeuralEngine` 只排除 Core ML GPU 选择，不排除 CPU fallback；Q/DQ 图或
模型包含 ANE program 不证明每个 GEMM、每次请求都执行了硬件 INT8。
report 同时记录 requested backend、observed placement、arithmetic evidence；未知写 unknown。

## 2. 静态 ConvRot W8A8 两种实现

### S1 · 保持旋转域，尽量复用已有 I8 codes

- 每层常量 signed I8 + per-output-channel scales，H256 recipe固定。
- 输入 H256 后 quantize，gate/up 共用；hidden 经 SwiGLU 后 H256，再 quantize down。
- 校准使用**旋转后的真实激活**，同时覆盖 steps/image/caption；不是把旧
  非旋转 SmoothQuant 的 scales 粘过来。
- 可以图内旋转，也可 GPU 预处理首个输入；但 hidden/down的再次旋转若放 GPU
  会拆分 FFN、增加往返，必须比较完整路径。
- 对 Core ML 不能原样表达的 signed range/scale 先失败或明确新量化 recipe，
  不暗中把 -128 改成 -127。

优势是尽可能保留原权重误差；难点是 H/QDQ 与编译器融合、静态activation scale
的质量、图内中间值范围。第一版浮点 norm/SiLU/residual不强求 A8。

### S2 · 解码+逆旋转，再做已知可导出的 W8A8

已有 exporter 的 `--convrot-mode derotate` 是可复用入口。逆旋转得到浮点有效
权重，然后 SmoothQuant/A8/重新 W8 校准，移除在线 H。它**新增权重量化误差**，
不能声称仍直接执行原始 ConvRot I8 codes。需要 source/derotate/calibration/
quantizer identity，并与 S1、ConvRot浮点参考分别比较。

### 静态图与低内存的冲突

不能同时常驻所有层的 Core ML模型、全部压缩 GPU权重与多层 dense槽，随后只报告
GPU权重省了多少。静态图没有动态权重复用，限制 loaded-model bank 到1–2层会
引入 load/unload、driver编译/cache与wired memory成本。

候选：只 offload 少数最有收益的层；只冻结一部分 FFN channels；受限图bank预载；
或禁用静态ANE改runtime。源 artifacts可以在磁盘，加载时仍需独立 admission。
不得每个 denoise step 重新导出/编译，不能在 timed inference 并行跑编译服务。

## 3. 现有 runtime FP16 可以先解决格式兼容

新增有界 decoder 直接写 FP16 slots，跳过完整 GPU dense中间副本。基础层支持
source format/stride/slice后复用已有 stage→launch→finish。

ConvRot有两个正确候选：

- 在runtime图中加入 H，传解码后的旋转权重；shared gate/up与post-SwiGLU H都保留。
- 解码时逆旋转，传普通FP16权重；每次layer refill都需支付逆旋转成本。

当前 `AffineView` 不理解 rotation，不能直接用 packed ConvRot 冒充普通 affine
然后删除现有门禁。runtime route descriptor、graph_version、source recipe一并升级。
FP16 headroom策略保留，精度与内存允许时才选择这条路线。

## 4. runtime W8A8 的核心障碍

本机 coremltools 8.3 的 `quantize/dequantize` 定义将 scale/zero_point设为常量；
现有runtime图的全部输入/slots是FP16。因此同时需要解决：

1. 同一shape图不断更换权重，compiler仍选择目标INT8实现。
2. 层间/行间变化的scale怎么表示，且不导致每层重新编译。
3. 外部数据类型、IOSurface绑定和内部arithmetic分别是什么。
4. A8引入的质量与量化/旋转/交接开销。

不能把 `mb.matmul(int8,int8)` 写出来就假设可执行；实际 MIL op type domain/
QDQ pattern/target支持要由小图确认。升级工具链也是新版本资格，不自动继承。

## 5. 动态 W8A8 最小证明实验

先做**一个投影**，不从完整模型开始。四个arm：

| Arm | 用途 |
| --- | --- |
| A：runtime FP16 | 现有通用图控制 |
| B：frozen QDQ W8A8 | 测目标shape的静态pattern/质量上限，不能代替动态 |
| C：runtime QDQ，固定规范化scale | 验证动态权重INT8候选 |
| D：CPU-only/不可用设备 | 错误和fallback控制，不作为ANE速度分母 |

要求同图切换 `W_A→W_B→W_A`，包括非均匀row scales、负数、尾块；输出随权重
正确变化，graph hash/编译计数保持不变。cold load、首次predict、warm predict、
staging、rotation/A8、output恢复分别记录。self-test不止一次单位矩阵。

### 固定规范化尺度的候选（未验证）

数学上可以令 `a=Qx/128`、`b=Qw/128`，图内 Q/DQ 使用固定 `1/128`：

```text
C = a b^T
Y[m,n] = C[m,n] * 16384 * sx[m] * sw[n]
```

Qx/Qw 的规范化值可精确用FP16表示。它绕开了“QDQ scale必须随层变化”的
表达问题，**没有绕开 compiler是否融合为动态INT8的问题**。量化可以在GPU先做，
或图内先除动态scale再固定QDQ，后者仍需验证设备支持/同步成本。

不要直接输出未归一化整数点积到FP16，它可能溢出。上述规范化也不消除所有
range风险：必须检查K上界、输出尺度恢复的FP32/BF16位置、subnormal与compiler
融合行为。若接口仍FP16，此方案可能改善计算却不减少slot字节。

输入 Q/128 可精确表示不等于输出点积仍 exact：FP16 的 C 出口会有额外舍入。
因此其 execution oracle 与 GPU INT32→FP32 epilogue 不同，按
[11 第 2 节](11-acceptance-profiles-and-feasibility.md) 分别判定，不拿宽松 A8 质量门
掩盖整数/出口错误。actual arithmetic 无法观察时仍填 unknown。

完整FFN还必须在每个gate/up恢复正确尺度后施加SiLU，在hidden处重新quantize。
把所有尺度推到整个FFN末尾是错误数学。外置GPU epilogue意味着多次跨设备，
因此单MatMul成功之后仍要检验完整FFN是否值得。

### 备选和退出条件

- 按层/scale编译固定图：归类static路线，不再叫checkpoint-independent runtime。
- 私有ANE/直接MIL产物：独立research backend，不修改本项目public路由默认。
- compiler无法保留动态INT8、CPU fallback、scale处理过慢：停止该候选，保留
  runtime FP16+有界解码。不能为证明W8A8而去掉SiLU、缩短K或跳过质量门。

## 6. 图内 tiling：三个独立维度

`Mchunk` 是token行；`tile_k` 是归约；`tile_n` 是输出通道；不能混为“矩阵切小”。

```text
one prediction:
  for each output N tile:
    accumulate all K tiles
  combine output tiles
  continue activation / next projection
```

- K tile部分和先完整相加，再SiLU；bias仅一次。
- 尽量整个FFN一张图，避免每个tile一次host predict。
- tile大小要和dtype、scale group、rotation group共同对齐；尾部按逻辑slice验证。
- 块更小可能减少单op工作集，但增加节点、临时tensor和调度；更大可能提高复用
  却超出后端高效范围。没有普适“越小越快”或“正方形必胜”。
- vpipe的权重buffer拐点是启发，不写成设备常量。本项目
  [tile记录](../../status/runtime-ane-tiles.md) 已显示某些N512候选好于N1024，
  但N256并不继续变好，且需重新平衡GPU rows才能改善整层。

起始搜索：Mchunk=256/320/352/512/1024/1792/2048（按真实rows过滤），
K/N=512/1024/2048及合法尾块。先单层含staging筛，再完整request；不导出笛卡尔
积全部大图，不并行编译污染测速。每个dtype独立搜索，FP16最优tile不照搬INT8。

## 7. GPU/ANE 分区的数学和内存

### token-row split

FFN逐row独立，拼接输出即可；GPU/ANE各需完整FFN权重逻辑维度。适合runtime复用，
但**不能按ANE只做20% rows就只预算20%权重**。caption/image可分开选择A8资格。

### intermediate-channel split

两个分支计算各自gate/up通道并用down对应列得到partial output，再相加；每个
分支的weight payload可缩小，但输出都为完整hidden。ConvRot down保持旋转域时，
通道组必须是完整256块，旋转组不可跨设备任意分裂。bias位置/浮点累加独立验证。

### 不允许的简化

不能在SiLU前后任意切K；不能用row split处理全局attention而忽略跨row依赖；
不能只算prediction更快就宣称fork/join更快；不能用不同GPU融合开关比较两个arm。

完整层成本包括input可用等待、H/A8、staging、GPU分支、ANE分支、join和post。
采用完整block probe选择on/off和chunk，至少有滞后阈值避免每层频繁震荡。

## 8. 模块复用与队列

复用单位：`kind + shape + dtype + layout + tile + rotation + quant protocol +
graph version + OS/compiler/device qualification`。静态图额外带权重/校准身份。
只有同shape与同数学合同才可换权重；encoder和DiT不能因都叫FFN而误共图。

一条持久ANE dispatch worker，独立且有界的转换worker；runtime graph bank默认
只保留当前阶段必要shape，不把全部层/shape的图留在内存。encoder→DiT交接前drain，
有预算才保留可复用模型。公开runtime继续使用已校验的私有artifact快照生命周期。

两组ANE权重slots可以将下一层staging与当前prediction重叠，但增加了接近一整套
FFN权重以及潜在driver buffers。第一版保持一套，只有完整内存/收益验证后才
增加第二套。GPU前瞻槽和ANE slots属于同一ledger，不能各自“都只占两层”。

## 9. 故障和发布

数值非有限、错scale、artifact身份错误、预算不足、部分chunk失败均不能发布
半成品。drain后用同预算GPU完整重算该范围；若GPU也无法容纳，返回可诊断错误。
不得在low-memory路径以whole-model dense fallback掩盖问题。

static/runtime FP16/runtime W8A8分别验收；只有整请求更快、内存可接受、质量合格
且placement证据明确，才讨论默认策略。硬件INT8算力潜力本身不是发布依据。

动态 A2 可先使用单投影 frozen QDQ 控制，不依赖 A1 完整静态 FFN 已完成。
固定两种候选 recipe、证据类型、筛选样本与 unsupported/inconclusive 的退出规则见
[11 第 7 节](11-acceptance-profiles-and-feasibility.md)。失败恢复按
[10 第 5 节](10-execution-and-product-integration.md) 的 transactional FFN 和 recovery
预算执行；不能在不清楚 drain 状态时继续 GPU 重算。
