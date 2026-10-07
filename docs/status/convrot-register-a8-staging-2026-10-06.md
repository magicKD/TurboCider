# ConvRot：GPU A8 寄存器 staging 与真实轨迹保持

2026-10-06，Asia/Singapore，M4 Max。接续
[GPU/GGUF 有界消费窗口](convrot-register-dense-window-2026-10-06.md) 和
[最终生成质量八格](native-final-generation-quality-2026-10-06.md)。
本轮是保持既有计算结果的局部优化，**不是** W8A8 相对 optimized GPU 的
质量/性能资格。双后端、四格 base ≥1.2×、真实 LoRA 整模型加速、memory
及 physical trace 目标仍未完成；原 N1 门槛和失败记录不变。

## 已集成的 GPU kernel

Private Comfy A8 的两个 Metal staging pass 增加 SIMD-register 实现。
一个 32-lane SIMD group 处理 H256，每 lane 持有八个值；保留原 Comfy
radix-4 加减次序、FP32 中间值以及旋转后一次原 source dtype 舍入，再
计算原 normalized FP16 scales 和 RNE A8 codes。不是 Sylvester，不再
量化原 ConvRot W codes，不删除 BF16 边界，也不改变 MIL 算术配方。

row A8 和显式 group256 A8 均支持。只在已有显式
`TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE=1`、经过验证的 dense Comfy
H256 activation staging 选择新核；开关0保留 generic control。
Sylvester H128/H512、raw GGUF、affine Q4/Q8 及 direct signed/packed
ConvRot W 不选此核。Public 默认路径及发行边界没有改变。
新代码不增加模型权重副本、W/A slot、scale cache 或无限 pipeline cache。
producer ready event、source owners、padding/非有限校验和失败整操作
GPU 重算保持原契约。新增核不调用 ANE，本轮没有物理 INT8 MAC/并发声明。

## 同 binary 的组件测量

`build_convrot_a8_benchmark.sh` 可重建独立 Metal-only probe。BF16 合成
有限输入；每 arm10次 warmup，31个串行交替 hot samples，全量保留。
每格 codes/scales/physical padding 按 bytes 一致，staging 不推进当前
inference shared-event timeline。测量包含 command creation/staging/host
ready wait，不含模型读盘、ANE prediction 或完整 FFN。

| M / K / A8 group | generic median ms | specialized-register median ms | 局部倍率 |
| --- | ---: | ---: | ---: |
| 1056 / 3840 / row | 0.521750 | 0.425209 | 1.22704× |
| 1056 / 3840 / 256 | 0.501792 | 0.366750 | 1.36821× |
| 1056 / 10240 / row | 1.163625 | 0.866375 | 1.34310× |
| 1056 / 10240 / 256 | 1.190167 | 0.797959 | 1.49151× |
| 4224 / 3840 / row | 1.621667 | 1.133917 | 1.43015× |
| 4224 / 3840 / 256 | 1.627334 | 1.064000 | 1.52945× |
| 4224 / 10240 / row | 3.873334 | 2.596625 | 1.49168× |
| 4224 / 10240 / 256 | 4.030916 | 2.558875 | 1.57527× |

这是 generic 与 specialized-register **组合路径** 对比，包含既有 typed
load/function-constant 特化，不单独归因于寄存器旋转。K10240是独立
staging geometry，不表示实际 FFN down 的 hidden A8 已移出 ANE。
100ms host observer15次观察、最大间隔0.1573s，无 observation error；
不是正式 competing-load gate、memory qualification 或 GPU timestamp。
首格仍有频率/调度状态切换，未裁掉任何 hot sample，不给置信区间保证。
先前两轮17样本也保留，不拼接/挑选进此表；首轮曾与 host tests 重叠。

最终组件回执：`outputs/convrot-register-a8-staging-31samples-final-20261006.json`，
SHA256 `a4dd188326e653b078d433f2f41124ffb5311f8e45a0a53e51105d9eb1d473e0`。
standalone binary SHA256
`0aeedd55f0f0cefbbcec5129f949da9d60c1760fcdd3bab8e22305d58f0ff29f`。

## 实模型：不是 GPU-reference 质量对照

最终同一 v2 library，原 ConvRot checkpoint / legacy stored BF16 scales，
fox prompt/seed42、4步、resident、Fa4096/Fg6144、原 FP32 partial join。
512使用1056-row、1024使用4224-row模板。generic/register各独立进程，
两臂均 Private W8A8；唯一实验差异是 stage-specialize0/1。
fixed-async1、scale-cache/launch-fence1，prefetch/lookahead/deferred0，
software BF16 value boundaries和group A8关闭。四个模型进程串行，未与
owned build/test/benchmark重叠。

两分辨率均 conditioning、initial latent、全部4步 latent、final latent
逐位一致，rel L2=0、max abs=0，PNG文件亦一致。每臂128 actual Private
calls /128 channel blocks、device IO calls128，zero failure/fallback/retry、
headroom1，实际 stage-specialized标记分别false/true。
slots/estimated bytes两臂相同：512为108232704/320241920，1024为
144605184/570032384。它们是 runtime slot/heuristic estimate，非 process RAM。

PNG SHA256：512
`da58e54e401566e60297c65f7de31cf74bec5297251abc4ed172e3a3b8409feb`；
1024 `61b26c1aa5c40f97e5360963bde415e5c05dcb5282e62da947dd1d914a57cae2`。
目录 `outputs/convrot-register-a8-model{512,1024}-{generic,register}-20261006/`
保留 request/dumps/PNG/observed receipt；register目录有`staging-parity.json`。
其局部 N1 true 表示两个 **Private** 路径一致，`qualification_passed=false`；
不能替代 [GPU-reference 八格负结果](native-final-generation-quality-2026-10-06.md)。
带dump、无热重复/反序和正式load/memory gate，故不计算整模型倍率。

## 回归发现与修复

最初独立 Comfy tests 通过，但扩展 GGUF/W8 stager 回归失败：新增缓存
字段误将首次H128的TG线程数用于共用 generic pipeline 后续H512。
修复后 pipeline只缓存“是否使用register核”，generic extent仍从每个
operation的spec获取。新增H512→H128→Comfy-H256→H128→H512混用检查，
独立 CPU codes/scales oracle通过，generic仍仅1个pipeline。
没有删除旧测试、接受错误数据或放松padding/finite/reuse检查。

最终98项工具质量/回执/overflow/calibration/screen/load/memory/image
host tests通过；另10项选定Private host/actual-driver tests通过（3 host，
7 hardware），覆盖raw GGUF Q4_0/Q4_K/Q8_0/Q6_K、affine Q4/Q8、三dtype，
Comfy direct37格、row/group各9格、独立offset/pitch的24格及失败/重填，
真实LoRA hidden/down、raw/packed×serial/lookahead、input/hidden group
scope、BF16 carrier guard，以及实际4224-row、多chunk、权重代际切换。
不是全仓/全部Private/Public/Swift suite验收。

最初v1 native library包含上述cache错误，未用于模型验收；其完整构建
误进入额外Swift/App阶段，随后只终止已确认的owned process group，保留
产物作为superseded实验，不碰外部ComfyUI。最终以native-only重建v2，
完整native构建成功，476个source input hash检查通过。
library SHA256
`f0c510e814221a909efb53bbdd3600c224ca71719f97d9dbf77c49378bd78987`；
CLI SHA256 `f09144aed59e0899e0f3cd06ec97e54958ac91094f7b54453aeb84faf62ea02f`。
build ID `tc-runtime-build-v1-0498910c612157f2984f0a94348080b172bdb36a2b48c11b536eba2a0404437f`。
构建保留原native ConvRot drafts；不是clean staged-only build或稳定App发布。

## 回执边界和后续优化

观察器新增CLI/相邻dylib前后hash，artifact发生变化时保留失败回执并
退出失败；薄CLI相同不能再隐去dylib变化。这仍不是实际loaded-image或
physical trace。quality binding拒绝缺失/非法binary identity、部分/改变
的library identity、requested/actual steps不符和Z dumps少于实际步数。
legacy receipts明确 `adjacent_runtime_identity_bound=false`；八格重新
比较到`quality-bound-v2.json`，质量结论不变，原文件不覆盖。

GGUF有界decode/GEMM的既有回本证据仍见
[消费窗口记录](convrot-register-dense-window-2026-10-06.md)：512形状对
GGUF gate需23/28次复用、down需46次；逐层eviction后每步重解码不划算。
本轮只保证raw GGUF stager不被Comfy优化破坏，没有将有界窗口提升为
默认或改写GGUF/model文件。后续按真实命中/eviction统计做提前解码；
ANE优先定位最终误差中的W/A/中间舍入，测GPU敏感激活与Public/Private
projection handoff，而不是加大型软件舍入图或盲扩大offload。
