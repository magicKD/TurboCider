# ConvRot512²：融合 typed decode 的 GPU F32 partial 实际盈利

2026-10-08，Asia/Singapore，M4 Max64GB / macOS26.6.2。接续
[split down ranks](local512-split-down-ranks-2026-10-08.md)及
[原 F32 join](convrot-fp32-join-error-isolation-2026-10-05.md)。只用本地
模型，无下载、GGUF/header改写、dense sidecar或大型 activation dumps。
完整 Z/Qwen base、实LoRA、1–2参考图、encoder、双后端、≥1.2×及严格
窗口目标仍未完成。新核默认关闭，本轮只得到有限诊断盈利。

## 新 GPU kernel 与源码边界

`affine_gpu_mpp.hpp` 消费已经产生的 MLX affine Q4/Q8 planes，不直接读
raw GGUF。每个系数仍以原 FP16/BF16 metadata dtype 解码，不先宽化
scale/bias或改变 Comfy legacy BF16-scale 权重。MPP right-input cooperative
tensor 持有这一 typed tile，`multiply_accumulate` 到 F32 destination；
不建立全矩阵 dense bank，也没有显式 threadgroup W scratch/barriers。
源码的 register/cooperative 表述不保证编译器无spill、占用率或物理MAC。

参考本机26.5 SDK的 `MPPTensorOpsMatMul2d.h` / implementation与官方 MPP
guide。input cooperative tensor需要single-SIMD execution scope；一个
128-thread TG划分四个SIMD group。最终候选每个SIMD处理M64/N32/K32，
TG覆盖M128/N64；直接从原physical pitch/row/column offsets读packed
weights，原最终 BF16 join、Comfy H256 ordering及输入旋转均保留。
没有改变 ANE W8/A8 graph、scale/headroom策略、两套bank或hidden ABI。

最初M16/K64/N64 tile可以编译，但实际FP16 Q4/g32数值relL2≈0.943846、
maxabs≈1.477373。失败log保留，最终明确拒绝该组合，不静默换tile或
放宽3e-6数值门槛；尚未独立定位layout/compiler/resource的唯一原因。
五个准入recipe经过200项实际Metal对照：两dtype、Q4/Q8、g32/64/128、
M33/67/128、physical offsets、strided input、M/N/K480尾块及非法配置。
对独立原dtype解码后F32 GEMM，relL2≤3e-6、maxabs≤2e-5。

## 实权重组件：F32 partial显著改善，不是packed QMM整体提速

同standalone v3 binary，本地Z layer0真权重、合成有限sine hidden、
M1056，每recipe三次warmup/15个循环换序串行样本。时间包含原输出dtype
cast与eval；排除初始文件read/packing、输入旋转、ANE和全模型。prepared
dense是独立控制，单次decode setup约4.29–4.50ms，未摊薄成零成本。

| source / projection | packed QMM ms | prepared GEMM ms | original F32 decode ms | M64 MPP F32 ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF Q4 / gate | 6.089250 | 5.942667 | 26.979500 | 12.907417 |
| GGUF Q8 / gate | 6.107000 | 5.933667 | 29.727958 | 12.957667 |
| ConvRot / down | 6.751458 | 6.194583 | 34.799500 | 13.874458 |

新核对旧F32 consumer约2.09/2.29/2.51×，但仍约为原packed QMM的
2.06–2.12倍时间；不替换纯GPU默认QMM。三个finite输入所有recipe的
原dtype输出对packed按bytes一致，F32 oracle relL2=0；不泛化成任意
输入/trajectory都逐位一致。原/候选三种较小tile及全部样本均保留。
observer complete、最大gap<167ms、binary不变、结束dense ledger claim0；
不是strict competing-load、GPU timestamp或whole-process内存资格。

v3 binary `d8a5bc854c409afbaae9fdd1812bad041a65baa3d4cf1428594ab0b7e953139d`。
它通过absolute rpath用保留的down-rank Private库，wrapper无adjacent
dylib identity；不是loaded-image trace或最终493-source库的测量。
数据 `outputs/local512-affine-mpp-v3-{q4-gate1056,q8-gate1056,convrot-down1056}-component-20261008.json`。

## 接入模型：不可变请求snapshot，默认不变

显式 `TURBOCIDER_Z_CONVROT_FP32_MPP=1` 默认0/unset。要求resident512²、
base、approximation、明确Private固定非零channels、原ConvRot W8A8与
F32 channel join、显式runtime ConvRot source。计划/运行共用gates；
Public、auto channels、非512、LoRA、非F32或错误source route拒绝，合法
flag在普通GPU控制上不起作用，非法值仍拒绝。

在owner thread开始请求时，将选择snapshot到当前`Weights`，不是在
每个projection/并发callback里修改process env。切换先drain/synchronize；
runtime request identity包括新GPU recipe，不能借用另一配置的executor/
calibration状态。现有两F32入口才选新核，普通dense/packedGPU projection
不受影响。失败仍沿用完整GPU FFN回退，不发布半成品。新marker是
选择描述，不冒称physical kernel计数；screen另要求每请求32×steps
成功channel blocks、实际driver calls、zero fallback和指定F32/basis。

## 两方向实模型：原混合约5.70s →4.08s，优化GPU约4.50s

同Private v1库，本地原ConvRot、fox/seed42、四步、512²、Fa4096/Fg6144，
fixed-async/F32 join，stage-specialize/launch-fence1，prefetch/lookahead0，
W-code cache关闭。完整GPU用原compiled-dense与显式retained3GiB allocator
hint，包含现有ConvRot草稿，不以慢legacy GPU作分母。每臂独立进程，
一冷两同prompt热请求，conditioning hits false/true/true，100ms进程树
采样；冷是first request，不称storage/compiler-cache cold。

| order | GPU warm median s | original hybrid s | MPP hybrid s | GPU / MPP |
| --- | ---: | ---: | ---: | ---: |
| GPU→original→MPP | 4.486711 | 5.705788 | 4.081827 | 1.09919× |
| MPP→original→GPU | 4.512016 | 5.695798 | 4.086223 | 1.10420× |

原混合wall减少约28.3–28.5%，比完整优化GPU减少约9.0–9.4%。保留两方向
分别的全部样本，不挑方向/合并分母，也未达到1.2×目标。冷GPU/original/
MPP为正向5.460725/7.016662/5.672137s，反向5.506021/6.940341/5.348021s。
没有以overlapped host timers相减推断各kernel成本或物理GPU/ANE overlap。

六个memory reports complete、system swap-in/out0；两方向MPP
phys-footprint peak约17.37GB，original约17.35GB，GPU约11.64/13.03GB。
GPU hint不是进程RAM cap，Private也不是仅driver的物理分配；scope为
load/cold/warm/exit进程树，排除外部服务/driver归因。没有扩大memory
admission、4GiB system reserve、2GiB optional tier或原load gates。

### 冷溢出：零重试窗口失败，不可追认正式资格

首次默认`cold_retry_cap=0`窗口：GPU三请求完成，original三请求产生
128/256/384成功blocks，但冷请求两次实际A8 overflow，故被拒绝，MPP
未开始，summary incomplete。原stdout/events/memory/PNG全部保留。

另开显式`--cold-retry-cap 2`的两方向 **诊断**，默认仍0。两臂都冷重试
两次、headroom1→4→16，之后三个回执headroom均16、累计retries均2；
每次热请求新增retries0。validator检查完整overflow events及不变的热
recipe，结束再要求两臂resolved headroom相同。不是隐藏重试、放宽旧
窗口、删除busy sample或正式zero-retry qualification。cold/warm输出
略不同也不能忽略；原N1/zero-validation不当验收的政策均不变。

每请求128个successful channel blocks，actual calls累计130/258/386，
多出来的2次正是冷重试；failure/fallback0，不写成retries0或128calls。
新核没有改变ANE数学配方；回执不证明physical INT8 MAC。

目录 `outputs/local512-convrot-mpp-v1-cold-retry2-{forward,reverse}-diagnostic-20261008/`；
被拒绝窗口 `outputs/local512-convrot-mpp-v1-forward-diagnostic-20261008/`。

## 画面和质量范围

两方向每组三张original/MPP PNG对应bytes exact；两方向也一致。冷hash
`1f998d4fc662eea2f10da40263708e646532a6ee1002f214bf8d783546611bef`，热hash
`1c741a984f626983625f7dc54a33acc6f9e6479e2ca70a430bb79f83b4a4070c`。
GPU三请求hash均
`86ed0a9392bc6fab063b849e4c8ec7b7c5a587b74223eeb115c8cfa374e094d2`。

已目视original/MPP热图，以及GPU/MPP冷、热两组，全部whole与三同坐标
裁剪。狐姿态、耳/脸、毛色、眼鼻、尾巴、雪/枝条和构图很接近；
眼周表情、毛纹/雪粒稍变，未见新增断裂或色块。有限one-scene/seed
agent观察，不是用户批准、多prompt/seed或latent gate通过。第一组
sheet固定标题GPU reference，实际左图是original Private，未冒称GPU。
visual manifests保持pending，既有ConvRot source/ANE N1失败不改写。

## 构建、回归和清理

Private实验native-only build、Public普通native-only build均exit0，薄CLI
另编译。Public actual release-binary guard通过，493source inputs在验证
时各无不匹配。含原ConvRot drafts，非clean staged-tree或稳定App发布。
本轮model screen的compiled-dense控制依赖这份含既有草稿的snapshot库；
缺少该route时工具会拒绝，不静默降级为legacy控制或冒称clean HEAD复现。

```text
Private 251195d0c540386d5fb911b23629beb9adcd42180d7196040c6363b9ecf7978c
Public  fce969491c97710ac427e6a07e14e8ff54505585ea311bfddb21aa0dfdbe7fc6
```

Private kernel/default selected10项、最终Private16项、Public26项均pass无
skip，含新200-case核、8-case Weights原/新/原与两F32 API、旧ConvRot
rotation/QMM/physical range、dense-window生命周期、原Qwen LoRA/encoder、
实际Private channel/calibration/failure恢复；不是全仓/完整模型矩阵。
host回执/observer共9项通过，默认零重试与显式冷重试的伪造/丢失/
热变化均拒绝。完成全部owned jobs后清理425个`.o`、33,111,200 logical
bytes和两个空module-cache，可重建；保留库/CLI/probe、logs、manifest、
PNG、模型/adapter。选择性提交owned hunks，原ConvRot草稿仍在工作树。

机器汇总见
[本轮证据](../design/validation/local512-convrot-mpp-partial-20261008.json)。

接续：多scene/seed与有效严格窗口，Z真实LoRA/Qwen实际盈利、GGUF混合
F32 consumer及实际raw decode/retention hit审计、Public实模型与不可变
per-operation backend overrides。Private适合已经测到收益的W8长矩阵，
Public/完整GPU可保留敏感/兼容性层；该Public–Private按层盈利选择尚未
自动接通，不能以构建期capability fallback冒称杂糅调度完成。先测
single-owner handoff，不为同片权重同时常驻两份ANE表示。GGUF容器与
pure GPU QMM暂不改；已有R8负结果仍有效，新F32收益不等于提前解码
回本或compressed全模型默认盈利。
