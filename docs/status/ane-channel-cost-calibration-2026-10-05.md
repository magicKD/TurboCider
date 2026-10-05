# 双后端共享校准计时与实际模型 channel 成本初筛

本轮实验采用 UTC 2026-10-05；机器 Asia/Singapore 日志为
2026-10-06，因此 ignored `outputs/` 的文件名使用 `20261006`。
接续 prepared calibration 和 ConvRot 寄存器/有界解码工作。完整双后端
集成、四格 base ≥1.2×、真实 LoRA 加速、质量/内存/device trace 目标
仍 active；本报告没有宣布这些目标完成。

## 实现与验证

`0d3b903` 接入完整独立 GPU/ANE/Both 校准：每个 cell 两次 warmup、
七次 hot，串行循环换序，保留所有原始 samples。GPU part 包含真实
source W8 staging、A8 pack、family GPU head、独立 frozen restore/join；
prefetch-on 还包含下一组权重。ANE 输入预完成、输出独立，不让 GPU
callback 依赖本次 live ANE output。多个深度 ordinal 0/7/15/23/31，
第五组用于未来权重。以 one/four median 差除以三扣除固定成本。

共享 CPU `ChannelCostModel` 拟合 G/A affine lines 与可行 bandwidth
coupling polygon。非 binding 点不臆造零 bandwidth fraction；使用
保守 uncertainty envelope。只在已测 share 区间选择 512-aligned
candidate，与独立实测 full optimized GPU 比较，要求预测收益 ≥5%，
取最佳预测 1% 范围内最小 ANE share。所有建议仍标记
`requires_candidate_trial=true`，不写入推理策略或 qualification cache。

接续整理将 `GpuCalibrationSamples` / full GPU baseline adapter 移入
`ane_calibration_timing.hpp`，Public/Private 可共用，不依赖 SDK、MLX
或 Private symbols。旧 Private qualified API 用 `using` 保持源码兼容，
one/four 顺序、warmup/repeat 限制、medians 与生产 steady clock 不变。
新增 fake-clock host test，确定性检查 serial drain、exact hot counts、
warmup/preparation 排除、保留 outlier、固定成本消除、非法 span/参数、
reset/partial-submit/finish 异常与 first-error preservation。

最终 host suite **10 项通过**；新 standalone probe 的 `--host-only`
通过。共享计时重构后的真实 ANE fixture 同样通过 ownership、完整
traffic、failure cleanup/refill 与两种 prefetch 检查；这轮 fixture 对
两种 prefetch 都建议 GPU-only，未将 toy 结果当成模型策略。第一次
fake-clock 零差测试用十进制 `.01` 引入浮点差异，已改为精确二进制
`.015625`；生产 policy 未因此放宽或修改。

## 四组实际模型诊断

M4 Max / 64GB。加载完整 transformer 并保持 resident，使用原 BF16
checkpoint 与固定合成 normalized input。这是 **Sylvester H128/H512
W8A8** 校准，不是 Comfy ConvRot source recipe，也不是实际生成的
激活轨迹。1056/4224 是诊断 row geometry，不等于完整 512²/1024²
请求 qualification。

以下四组全部使用 `0d3b903` 时同一 standalone binary，SHA256
`8fb633fa14f4b6fff9c008b3a3bd57f67bf8119d0d5934cd67706884270869da`。
没有与 build/test/其他 owned timed arms 重叠。外部进程没有暂停或
结束；component wrapper 记录进程 comm、CPU/load 与 observer gaps，
但不强制整请求的 competing-load gate，也不证明设备独占或物理重叠。

| 模型 / rows | full GPU ms/layer | Fa 建议 | prefetch-off 预测 ms | GPU / 预测 |
| --- | ---: | ---: | ---: | ---: |
| Z-Image / 1056 | 16.302861 | 4096 | 13.260034 | 1.229474× |
| Z-Image / 4224 | 66.577931 | 4096 | 48.334481 | 1.377442× |
| Qwen 2.1 / 1056 | 22.240125 | 5120 | 17.311425 | 1.284708× |
| Qwen 2.1 / 4224 | 84.674264 | 5120 | 61.417412 | 1.378669× |

更大采样 share 的 ANE-alone 分别约 19.485/64.476/24.046/81.013ms，
比相应较小 share 慢，不能用更多 ANE 通道直接推断加速。
prefetch-on 仍建议相同 Fa；预测分别 13.288559/46.004178/
17.258923/61.146179ms。它们没有完成同候选的独立整窗口 prefetch
对照，**不自动启用预取**。测量范围以外没有外推最优 share。

每组四个 complete-traffic point，每 point 实际累计 90 次 ANE calls
（含 warmup），每个 raw hot cell 七个 samples。观察器无采样错误，
最大 gap 0.534/0.539/0.566/0.536s。探针的 proposal admission 仅限制
不超过已 admitted 的最大采样宽度，不是逐一新 candidate 的完整
graph/surface memory admission；建议仍需实际实例化及完整运行试验。
较小建议恰好是已采样 graph，不能省略实模型 trial/identity 检查。

Z small full GPU 使用原 physical MPP projections + fused SwiGLU；
Qwen full GPU 使用原 fused gate_up 权重及编译后的 SiLU/down；Z large
使用 compiled dense projections。各自分母为同 source/input 的完整
family GPU FFN，不是 extrapolated G(0)。GPU channel head 与当前
runtime callback 相同，Qwen 使用 separate gate/up projections。

## 完整请求试验：两次不合格窗口

新薄 CLI 链接既有 guarded Private library
`b6944e101b64887be16324d81cdf104e0462b77373283aec14c6be1bc9e04ca2`，
本轮没有重新构建整库。CLI SHA256 为
`02c55923be1dff7fc8dfd5681fd26b1cfaf15d8553eec28d1d9ce09b16d11ebb`。
该库是先前保留 ConvRot drafts 的 working-tree build，不能称为本次
shared-timing refactor 后的完整库，也没有新的 Public release build 声明。

Z-Image BF16 base、512²、fox/seed42、8 steps，Fa4096、c1056/v1，
W8A8 GPU I/O、chunks1、fixed-async1、scale-cache/launch-fence/
stage-specialize1，prefetch/A8-lookahead/deferred0。每个 route 一冷
五热；先 GPU→runtime，再 runtime→GPU。

- 正序 GPU arm：83 个 load samples 中两次短时外部 ComfyUI Python
  活动，连续 load verifier 拒绝，未进入 runtime arm。
- 反序 runtime arm：68 个 load samples 中两次同类活动，拒绝，
  未进入 GPU arm。两份原始 summary 都保持 incomplete、零认可 trials。

**不计算这两份窗口的速度倍率，不借用旧库 GPU 分母。** 未向外部
进程发信号或放宽门槛。独立读取反序六个原始 native results：严格
route/IO/data-path/fixed-async/deferred contract 通过，每请求 256 次
Private calls，无失败、retry 或 GPU error fallback；六张 PNG 都与
之前相同 Private recipe 的 hash 一致：
`6d3f01c35392246eedc75fddc76c8dc6d0048ad9ec42e339fbaa7bdc20aaf7cf`。
这是独立生成和回归一致性证据，不是 GPU-vs-W8A8 质量资格。
native quality validation calls 仍为零，不能把其 passed 字段当验收。

## 对 ConvRot / GGUF 的优化结论与接续

- [寄存器核/有界解码](convrot-register-dense-window-2026-10-06.md)：
  exact H256 register kernel 的既有 operator 收益保留；与原 butterfly
  相同不代表已修复 dense-H reduction 的模型 N1 差异。GGUF/ConvRot
  每次重新 decode 后再 GEMM 的成本仍高于 packed QMM；只有真实
  跨调用保留热点、达到回本复用才值得采用，不把 32 层 eviction 轮转
  误算为同矩阵复用。当前两槽是矩阵槽，不是两个完整 FFN。
- [BF16 value-boundary 负验收](convrot-bf16-value-boundaries-2026-10-06.md)
  仍有效；不增加更多软件舍入节点或把 Comfy W8A8 提升默认。下一项
  是 GPU 接手敏感激活/scale restore 与 ANE projection 的完整 handoff
  成本及质量对照；Public/Private/GPU 按 recipe/layer 选择仍需实现。
- channel auto 仍是固定 share 开/关，**尚未接入本次 multi-share fit**。
  还需 native model/runtime constructor/scheduler 集成、真正 candidate
  graph admission 和 trial，再按 source/model hash、rows/geometry、
  encoding/precision、backend/recipe、SoC/OS/build、Metal/graph ABI、
  adapter identity 和 prefetch policy 隔离 cache。Qwen LoRA range
  callbacks 更改 share 时必须安全重建；失败仍 whole-operation GPU
  重算，不发布部分 scratch。正式四格/多提示词/反序/latent/媒体/
  process memory/设备 trace 验收继续，不把本表填成 ≥1.2× E2E。

## 原始证据哈希

原始文件保留在 ignored `outputs/`，模型、adapter、参考实现和设计
原稿不改、不提交。四组 real receipts 的 suffix 是本地 `20261006`。

| 文件 / 路径缩写 | SHA256 |
| --- | --- |
| `channel-calibration-reviewed-z1056-20261006.json` | `87f342c384daa3442e3204e2877cf7b9f6f9f3e450578b955e57c548f79062db` |
| `channel-calibration-reviewed-z4224-20261006.json` | `b183e8fea88b453635de7b9256c527ee7306a1866709f1fa2ca0b912eb9112fc` |
| `channel-calibration-reviewed-q1056-20261006.json` | `6dbc41c306369dc403c50b11bc17bb3e985379d2e9786be20a85a0300faf9f40` |
| `channel-calibration-reviewed-q4224-20261006.json` | `0b5e324b231acab8d1ee93f3e002d286580476bc14100440df9d58a4709008fd` |
| `channel-calibration-shared-timing-fixture-20261006.json` | `ce9f357abbc827c86f7575f2ee73bde1515fbc2ae4552dbd9b826c1eab3b7cb7` |
| forward5 `summary.json` | `2d9abdbcb057aa0b3756c7a23d48b2147c879f181d2904c0fcd9b6185116cf94` |
| forward5 `0-gpu-load.jsonl` | `9469fe60f27fd50f21bd096ecbb2922eae21177da9c63b83dbb7d278b6254efe` |
| forward5 `0-gpu.stdout.jsonl` | `f51e9c552e9d63160ef5851ac0a0558c238b85623e6ea652ab8b96b20ae28cae` |
| reverse5 `summary.json` | `5b436c76931fd37005fb846d75b98839cbbafccc5857a45eab23f8e2ca32f282` |
| reverse5 `0-runtime-load.jsonl` | `58166b856bbfe804d141960198e43d0c096d67dc3699a779b192f01236567721` |
| reverse5 `0-runtime.stdout.jsonl` | `76baf8f63ec25400371fe8ea421789f60d6be074d2be5ccd82c032ba95eac797` |

forward5/reverse5 指
`outputs/channel-calibration-z512-candidate-{forward5,reverse5}-20261006/`。
最终 shared-timing fixture binary 为
`469830e8a9a7d5c4c8fe8bbe10e5a59b4de74a0dd3dc3ffcbab1fcfd8a04580b`；
四格 real 数值在这个整理之前测得，没有冒称用此最终 binary 重测。
本轮所有 owned build/test/calibration/inference handles 已 terminal。
