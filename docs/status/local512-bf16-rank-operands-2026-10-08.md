# Qwen512²：性能优先的 LoRA delta、BF16 source ranks 和配对 GPU kernel

2026-10-08，Asia/Singapore，M4 Max64GB / macOS26.6.2。接续
[split down ranks](local512-split-down-ranks-2026-10-08.md)与
[完整 ConvRot LoRA](local512-convrot-lora-2026-10-08.md)。本地原模型、
adapter和参考图只读；没有下载、重写权重或生成 dense sidecar。
整体 Z/Qwen base、真实 LoRA、1–2参考图、encoder、GGUF/ConvRot及
快于优化 GPU 的目标仍 active。

本轮结论：A-rank 单算子明显较快，但整图 GPU 收益不足1%，混合
路线两种 workload 均变慢。画面接近；这不是质量门槛阻止采用，而是
整次生成收益不足。新开关保持默认关闭，不拿单算子收益冒充整图收益。

## delta 误差：按用户要求，允许近似并优先实测速度

这条实验的 component delta 相对 L2 筛选预算设为 **5%**。它是允许
继续性能/视觉评估的预算，不是目标误差，也不是最终 latent/图像
自动验收。原始诊断照常保留；NaN/Inf、非法 dtype/shape/source、
内存 admission、load 检查及晚期失败 complete-GPU 重算不放宽。
原精确 API 的回归不受影响。

独立 BF16-source/F32-rank 夹具108 cases 的最大 delta relL2 为
`1.6809e-5`，即约 **0.00168%**，远小于5%预算。F32 rank/output
契约仍保留；rank小误差检查和原 projection/shared rounding 回归
不是最终画面等价要求。新增预算自检接受3%/5%，拒绝5.1%、负值和
非有限值。没有通过丢掉 LoRA、alpha 或负 strength 来降低成本。

最初 strict delta1e-5 失败日志、随后1e-4诊断日志都保留，不改写为
最终通过。rank8/M1 曾约1.2e-5，rank256/M67约1.6809e-5；当前
native recipe 对小 M/rank 沿用原路线。允许近似不代表重新标注旧失败。

## 实现：原 BF16 A/X，F32 ranks，原 B/delta

`TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS=1` 默认关闭，
要求0/1、approximation、resident512²六步 LoRA GPU/runtime，且关闭
FP16 rank option、streaming、memory guard和 prompt enhance。合法
base请求忽略开关。实验范围不自动扩展到1024、Public row调度或Z模型。

`Weights::runtime_lora_rank` 在 BF16 X/A、batch1的3D输入、M>=32、
rank>=64、A row-contiguous时，直接使用现有 MPP range kernel，
BM16/BN64、F32 accumulator/output。其他 dtype、短序列或小 rank
保持原 F32 路线。full projection、slice delta、shared ranks 都走
同一 helper；B/delta 仍F32，alpha、叠加顺序和一次输出 cast 不变。
避免仅为 A 投影先生成大型F32 A/X；不称 reduction order 逐位等价。

请求 owner 在开始时 snapshot开关，切换时 drain/synchronize。early/
late两处 runtime identity、calibration GPU configuration及 prefix key
都隔离该策略；compiled factories/ranks绑定同一请求的resident Weights。
selection marker说明选择，不是物理 kernel计数或证明所有投影均 eligible。
没有重新量化 adapter、修改 base ANE W8 slots 或缓存跨请求裸 ranks。

真实本地 Viggle adapter 的 layer0 BF16 A、合成 BF16 X，warm3/hot15
循环换序、串行 operator host span：

| M / K / rank256 | 原F32 ranks ms | BF16 operands/F32 ranks16×64 ms |
| --- | ---: | ---: |
| 1056 / 4096 | 0.431958 | 0.329583 |
| 1056 / 7168 | 0.605750 | 0.461417 |
| 1056 / 12288 | 1.013458 | 0.748417 |
| 3137 / 4096 | 0.991167 | 0.757083 |
| 3137 / 7168 | 1.639084 | 1.078000 |
| 3137 / 12288 | 2.661333 | 1.689667 |

这些格子的rank relL2最大约9.35e-7，部分0；不包含B/delta、完整
FFN、encoder或视觉，也不是GPU timestamps。只加载 layer0 MLP
adapter前缀，未为probe保留整份adapter。observer完整，最大gap约165ms。

## 一/两参考图：四路线，同库、同来源的真实模型

原 BF16 Qwen DiT、Viggle v0.2.1 r256、六步/seed29、512²，三个
fresh prompts、conditioning全部miss；四臂相同 encoder source retention。
encoder都是GPU；两 hybrid臂 Private DiT5120、共享gate/up ranks，
down split及W-code cache都关闭。GPU encoder不能报成combined加速。
每臂独立进程一冷两热，100ms process-tree采样，无continuous-load资格。

| route | 一图 cold / warm median s | 两图 cold / warm median s |
| --- | ---: | ---: |
| GPU operands off | 15.819303 / 10.991971 | 17.907361 / 13.113089 |
| GPU operands on | 15.819968 / 10.933379 | 17.922028 / 13.076818 |
| hybrid operands off | 16.284862 / 11.965645 | 18.452813 / 14.468293 |
| hybrid operands on | 17.467148 / 12.366685 | 18.782794 / 14.854974 |

一图顺序 hybrid on→off→GPU on→off；两图 GPU off→on→hybrid off→on。
GPU on比off分别快0.533%/0.277%，不足以称稳定资格；hybrid on
分别慢3.352%/2.673%，仍慢于同库GPU on。全部 cold/warm、负结果
保留；没有挑有利分母、把不同workload合并或补写strict-load资格。

每请求227个adapter bindings；成功channel blocks累计192/384/576。
一图 actual ANE calls累计224/448/672，两图256/512/768；fallback/
failure/overflow retry0。两hybrid臂每请求共享192组、384个rank arrays。
source只load1次，后续reuse；GPU臂无ANE调用。这些是host/executor
回执，不证明物理INT8 MAC或GPU/ANE overlap。

八个memory reports complete，最大采样gap约111ms，系统swap-in/out0；
phys-footprint peak约38.18–41.02GB。范围为进程树load+cold+warm+exit，
不含外部服务/driver归因。compression/decompression仍有大量流量，
不能从零swap推出完全没有内存压力。

host分段提示收益被前段抵消：一图两热请求的pre-FFN host spans
off约4.81/4.81s，on约5.37/5.51s；hybrid-FFN部分off约4.61/4.60s，
on约4.28/4.31s。两图同样前段增长、FFN部分略减少。计时有嵌套/
同步边界，不相加当成设备成本；没有硬件trace，不能断言compile
fusion、额外dispatch或内存竞争中的某一项是唯一原因。

保留目录 `outputs/local512-qwen-edit{1,2}-rank-operands-v2-diagnostic-20261008/`。

## 画面：不逐位一致，但本scene的结果接近

全部24张PNG保留，18个whole pair已目视；另外检查18个detail pair：
一图GPU off/on case0、hybrid off/on case1、GPU on/hybrid on case1；
两图GPU off/on case0、hybrid off/on case2、GPU on/hybrid on case1，
每组center/top-left/bottom-right三同坐标原尺寸裁剪。

壶体、盖/钮、壶嘴、把手、左右布局、桌面与阴影非常接近；釉面细纹/
高光稍有变化。两图case1 GPU对hybrid有可见整体亮度差异，但未见
新增明显断裂或色块。这是单scene/seed的有限agent观察，不是用户
批准或多seed保证；其他case的detail虽已生成，未冒充已逐一检查。
自动visual manifests保持pending，历史N1/数值失败不改写。

视觉工具固定标题写GPU reference/ANE candidate；GPU off/on组右侧
实际也是GPU，hybrid off/on组左侧实际是Private hybrid，应以目录及
manifest来源为准。数值PNG SSIM/RMSE仅诊断，不替代上述肉眼检查。

## GPU paired gate/up：保留物理来源，不引入权重拷贝

新 `dense_gpu_pair.hpp` 是研究consumer，未接入默认模型。单次Metal
dispatch消费同一immutable BF16/FP16 W的两个等宽物理row interval，
保留pitch和column offset；seam按BN对齐，不gather/concatenate W。
原operand dtype、F32 accumulation、可选F32输出，支持BM16/32/64、
BN64/128。144个actual Metal cases与原两次range kernel字节相同，
覆盖tails、strided X、offsets、六tiles、F32/narrow输出与非法seam。

真实layer0 BF16 gate_up、合成X，控制是原两次matmul的独立输出，
没有人为加入concat拖慢分母；warm3/hot15循环换序：

| M / half7168 | 原两次MLX ms | 两次MPP ms | paired32×128 ms |
| --- | ---: | ---: | ---: |
| 1056 | 8.755083 | 8.492625 | 8.437084 |
| 3137 | 24.979708 | 24.683667 | 24.623708 |

全部tested recipe relL2=0，BM64×BN128反而更慢。pair相对已优化
two-MPP收益很小，未扩大到全模型或宣称ANE盈利。observer完整，
最大gap约167ms。probe没有adjacent dylib，使用absolute rpath链接
已保留Private库；包装器adjacent-library字段null，不伪称loaded-image
trace。首轮Metal变量名`half`和非法seam夹具错误日志保留。

## 构建、回归、提交与清理

最终选定Private38项、Public40项通过，无skip；包括108个新rank
math cases、原projection/shared/down、实际Private channel错误恢复/
晚期complete-GPU fallback、pair/dense/register affine、完整ConvRot
LoRA和实际Public encoder。不是全仓或Public实模型性能矩阵验收。

Private native-only build exit0；Public `TURBOCIDER_BUILD_LIB_ONLY=1`
exit0，另构建adjacent thin CLI，实际release-binary guard通过。两库
各494个source hashes匹配本轮工作树；含原ConvRot drafts，是working-
tree snapshot，不声称clean staged-only provenance。library SHA：

```text
Private 1a9a1b4af05f5c0c17268ffc19766db08e600723a904f144c495b884ee06ab42
Public  7a407c89a48a3a010aee2cd63438bf7b17a10c7defadb4f8e166d1d054adf9aa
```

所有owned jobs终止后，仅清理v1/v2 Private和v2 Public三个build中的
637个可重建`.o`，50,095,536 logical bytes（约47.8MiB），及三个空
module-cache目录。保留库/CLI/probes、logs、PNG、manifest、模型和
adapter，外部进程未触碰。selective commit只包括本轮owned文件，
原ConvRot草稿不夹带。相对路径机器记录见
[本轮证据](../design/validation/local512-bf16-rank-operands-20261008.json)。

接续应按完整请求优化critical path、adapter B及dispatch/fusion边界，
允许更宽松delta的低精度候选继续做实图评估；不为数值逐位等价牺牲
速度，也不因允许误差就把实际更慢的ANE路线默认启用。GGUF提前解码
仍需真实hit/eviction与全层轮转收益；Private/Public杂糅仍按不可变
operation/source/shape配置选executor，不能靠改变并发进程env选层。
