# Qwen LoRA 联合 A/B：小幅 GPU 整图收益，首步混合仍未超过 GPU

2026-10-09，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[首步并行分阶段](local512-qwen-prefill-parallel-2026-10-09.md)。本轮
先实测GPU partial-down舍入候选，再将BF16 A operands/F32 ranks与
已有B epilogue结合，使用真实单/双参考图六步LoRA验证。没有下载、
改写模型/adapter、修改参考工程或生成dense sidecar。完整Z/Qwen
base/LoRA、1–2参考图、encoder、GGUF/ConvRot加速目标仍未完成。

## 先排除较慢的 BF16 partial-down

新 `qwen_down_boundary_probe` 只读本地layer0原BF16 down，physical
pitch12288，合成finite hidden，M1024/2096/3144、K5120/7168/10752、
N4096。eager与compiled的MPP F32、MLX BF16结果再widen到F32共36格，
各3warmup/15循环换序hot；含partial输出cast/eval，无原权重disk副本。

| M / K | compiled MPP F32 ms | compiled MLX BF16→F32 ms |
| --- | ---: | ---: |
| 1024 / 7168 | 4.271833 | 4.385625 |
| 2096 / 7168 | 8.417917 | 8.664709 |
| 3144 / 7168 | 12.533291 | 12.890875 |

BF16 partial的relL2约.00166–.00167（约.17%），但全部tested geometry
略慢约2–3%。**不接入模型，不牺牲精度换一个更慢的consumer。**
这是同物理原source API，不断言MLX内部没有layout copy，也不推断
唯一慢因。host component spans不是device timestamps/完整FFN盈利。

receipt `outputs/local512-qwen-down-boundary-v1-component-20261009.json`。

## 联合 A/B 的实际数据流

新 `TURBOCIDER_QWEN21_LORA_BF16_AB=1` 默认关闭，明确选择已有两个
consumer组合：BF16 X/A→F32 ranks；然后原BF16 B、rank narrowing与
F32 scale/base epilogue。F32 rank存储/alpha与stacked rounding契约
保留，但不是原F32 operand计算逐位一致；不merge/requantize adapter。

仅approximate resident512²六步、一个原inference-time LoRA、GPU或
明确Private固定非零channel runtime；排除FP16 ranks、时序FFN reuse、
streaming/constrained/budget/PE。joint与旧独立A/B flags同时启用拒绝，
旧两个flags互相冲突的行为不改变。base合法flag忽略，非法值仍拒绝。

请求owner snapshot两个effective选择并drain/synchronize后更新Weights；
early/late executor与prefix/calibration key已经绑定两种effective
算术，不临时改process env。新plan/selection使用joint专用marker，
不再把组合冒称“原A-ranks的B-only”。它影响**所有eligible adapter
projections，包括attention和后续GPU步骤**，不是只影响ANE FFN。
原base W8/A8/restore、hidden ABI、memory admission和完整GPU失败重算
均未改变。

24个actual MLX cases覆盖BF16/F32源、rank64/256、M1/128/145、非零
physical column range、stacked ±strength/BF16 alpha、shared ranks和
独立原始F32 alpha oracle。最大delta relL2 .0153679（约1.537%），
低于5%component预算；F32 ranks与immutable master id保持。不将该
预算当最终latent/画质通过，不放松finite/shape/source安全检查。

## 完整在线 A→B 组件，不能借 prepared rank 抹掉成本

原layer0 BF16 A/B、合成输入，四个compiled policies原始/A-only/
B-only/joint；每次计时确实重算A-rank、B、scale和最终delta cast。
3warmup/15循环换序hot，不含base FFN/encoder/整模型。

| M / projection | original ms | B-only ms | joint ms | joint delta relL2 |
| --- | ---: | ---: | ---: | ---: |
| 2096 / gate7168 | 1.625625 | 1.325000 | 1.096708 | .002581 |
| 2096 / gate5120 | 1.341958 | 1.148583 | .900291 | .002572 |
| 2096 / down4096 | 2.434291 | 2.345250 | 1.533583 | .002375 |
| 3144 / gate7168 | 2.248792 | 1.842583 | 1.520583 | .002583 |
| 3144 / down4096 | 3.352459 | 3.116958 | 2.165666 | .002375 |

两component probes均observer errors0、binary unchanged，最大gap约
.274/.271s。二者absolute rpath链接保留phase-v1 Private库，没有
adjacent dylib，不冒称下面joint-v1库/loaded-image/device trace。
receipt `outputs/local512-qwen-lora-ab-v1-component-20261009.json`。

## 实模型：匹配 B-only 控制，保持首步并行/后续完整 GPU

原BF16 Qwen、原Viggle v0.2.1 r256、strength1、六步、seed29、512²，
三个fresh prompts，conditioning全miss。四臂同source retention，
encoder全部GPU；off为已经较快的B-only，on为joint，不用原慢F32 B
当分母。hybrid只在prefill做原channel分工，后五步完整GPU；两臂
同shared gate/up ranks。没有down split/W-code cache/时间reuse/
prefetch/lookahead。每臂独立process一冷两热、100ms memory sampling，
未请求strict load资格。

单图Fa7168/Fg5120/bucket2112、首步2096行；双图Fa5120/Fg7168/
bucket1056、首步3144行。不同workload使用不同分工，不合并比较。

| route | 单图 fresh warm request s | 单图首步 / 后五步 s | 双图 fresh warm request s | 双图首步 / 后五步 s |
| --- | ---: | --- | ---: | --- |
| GPU B-only | 10.885866 | 2.521207 / 6.322087 | 12.982578 | 3.839780 / 6.516941 |
| GPU joint | 10.595190 | 2.430861 / 6.152968 | 12.701378 | 3.751156 / 6.379754 |
| prefill-parallel B-only | 11.549650 | 2.739381 / 6.258976 | 14.177816 | 4.530652 / 6.493195 |
| prefill-parallel joint | 11.357079 | 2.741127 / 6.120500 | 13.863937 | 4.330929 / 6.365146 |

GPU wall减少约2.67%/2.17%，hybrid约1.67%/2.21%，但hybrid joint仍
慢于匹配GPU joint约7.19%/9.15%。单图hybrid首步没有改善；双图首步
约减少4.41%，不足以超过GPU。收益包括普通GPU LoRA工作，不能
将整图差值全部算给首步FFN、ANE或一个component。

单图顺序GPU off→on→hybrid off→on；双图反序hybrid on→off→GPU
on→off。不是同workload双向/ABBA或统计稳定保证，不把新收益与
上一轮B-only收益跨窗口相乘。每个cold/warm样本均保留。

每请求227实际adapter bindings，hybrid只在首步有32成功channel
blocks/32 shared rank sets/64 rank arrays；单/双图32/96实际driver
calls，后五步0。累计calls为32/64/96和96/192/288；failure/fallback/
overflow retry0、headroom1。严格phase validator拒绝未选中阶段的
调用、reset/伪造计数与不完整实际工作，不从marker推断执行。

8个memory reports complete、swap-in/out0，peak约38.3–40.85GB；
scope是进程树load+cold+warm+exit，不归因外部services/driver。
单图GPU on比off高约1.09GB，双图高约.11GB，不能因移除F32 operand
casts就声称whole-process内存下降或把峰值差异归给唯一原因。

两workload的B-only GPU三张PNG与上轮匹配GPU exact，hybrid三张亦与
上轮对应prefill-only exact，验证新flag默认/控制输出没改变这些cases。
joint不是byte-equivalent。目录
`outputs/local512-qwen-edit{1,2}-joint-ab-v1-diagnostic-20261009/`。

## 视觉与接续

已检查case1单图GPU off/on、hybrid off/on的whole/center；双图对应
两组whole与三同坐标原像素crops。壶体、盖/钮、把手、双壶位置、
颜色和阴影很接近，釉面/高光/纹理有小变化，未见新增明显棋盘格、
色块或断裂。其他case/crops没有冒称已逐一验收；agent有限观察不
是多seed/用户批准，automatic manifests保持pending。
工具固定标题GPU reference/ANE candidate；GPU组右侧其实也GPU，
hybrid组左侧其实是Private control，以manifest来源为准。

保持显式实验默认off。下一步GPU channel gate/up的base+LoRA B
epilogue融合值得独立测，保留原base舍入/alpha顺序或明确近似配方，
避免全量delta/intermediate traffic；不能以此未实现方向冒称已盈利。
还需profile实际correction readiness、partial-down、restore/join
及更合适share/bucket，保留最快完整GPU decode。Public/Private按
operation盈利选择、Z/Qwen完整base/LoRA/encoder/GGUF/ConvRot矩阵、
更多scene/seed、strict窗口与物理trace仍未完成。

最终build/tests/cleanup与artifact hashes见
[机器证据](../design/validation/local512-qwen-joint-ab-20261009.json)。
所有native构建仍包含原ConvRot drafts，提交只选owned hunks，不
冒称clean staged-only制品、App发行、原生INT8 MAC或物理overlap。

Private43、Public52项selected regressions通过、无skip，包括新24-case
joint math/plan、原A/B separate gates、phase24-case、stacked/shared/
down-LoRA、MPP/ConvRot、actual Private channel晚期full-GPU fallback、
actual Public encoder/FFN wrapper。另19项host screen contracts通过；
不是全仓或完整Public实模型性能矩阵。native-only builds均exit0，
相邻thin CLI完成，两份495 source hashes分别匹配，actual Public
release class/flags/link guard通过。

```text
Private 1862a3662020839ed6fad6161231b6e750e3427f5024017d402b44c1cc35b86a
Public  64f6eef4f5d672850a569618aff65b490c93103946a8215929e98f0ff392892a
```

所有owned jobs确认terminal后，清理本轮两个build的425个可重建`.o`，
33,214,832 logical bytes（约31.7MiB）及两个空module-cache。库/CLI/
probes、原日志/PNG/manifest和全部原models/adapters保留，没有清理
外部process或用户缓存；没有将保留量测库冒称clean HEAD复现制品。
