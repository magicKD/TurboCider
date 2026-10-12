# 原生 LoRA-aware channel auto：真实修正流量、异常排空与未通过的边界

实验日期 UTC 2026-10-06，Asia/Singapore 同日。接续 `b1fa882`。
完整双后端、Z/Qwen 两分辨率 base ≥1.2×、真实 LoRA 加速、质量/
内存/device trace 目标仍 active；本轮不宣布完成或正式性能资格。

## 实现

`HybridFfn::CalibrationWorkload` 新增按层重建的 adapter provider。
Z-Image 与 Qwen 的实际模型构造点提供 full/range gate-up 修正和
**一次完整 hidden 的 down-LoRA**。Qwen 完整 GPU 分母使用原
`Weights::project` compiled LoRA 图，不拿 base-only matmul 代替；
Z 有 adapter 时不再走 base-only MPP channel callback。原权重与
adapter 不合并、不重写；旧显式通道配置与 Public 默认不变。

独立 ANE sampling 的四组 DG/DU inputs 在时钟外由真实 LoRA 生成、
上传至 immutable surfaces。GPU-alone 不读本次 live ANE output，
独立复制 frozen base-down/hidden/scale snapshots，恢复后与 corrected
GPU head 组合完整 hidden，只执行一次 down correction。GPU/Both
时钟内仍实际重新计算 range corrections、上传、W/A staging、head、
restore/join/down，不能以 frozen correction 假装在线 LoRA 成本。

`W8GpuCalibrationWork` 动态 producer/drain 接口在部分提交、producer/
geometry/head/fence/drain/join 异常后，排空所有已尝试 producers，
恰好一次，保留首个错误；没有成功 completion receipt 时不发布
scratch。返回 source owners 保留至全部生产/传输完成，之后释放；
健康 refill 与 destructor 的 correction cleanup 有实际 GPU fixture。
caller-owned heads 仍必须按接口调用 `finish` 排空，析构不是它们的
替代 fence。base/static-correction 路径保持兼容。

独立 trial 使用真实 adapter 的新推理 executor，不将 base trial
套到 LoRA；原 rel L2≤0.03、cosine≥0.999、实际 four-FFN 窗口至少
5% 收益以及无 fallback/retry 门槛不变。缓存 recipe 为独立
`sylvester-dh-b128-b512-rne-norm-f16-v2-lora`，绑定实际 adapter identity；
Qwen 额外绑定 rank precision 配置。缺 identity、graph activation inputs
或完整 callbacks 时仍明确 GPU-only，不消费 base-only samples。

回执新增 `lora`、每点 `correction_computations` / `correction_uploads`。
verifier 要求每点 90 次 tail correction pair 计算与 180 次上传：
one/four × GPU/Both × (2 warmup + 7 hot)。这不包含时钟外 frozen 初始化，
也不把 GPU head 自身的 LoRA 计算混入该计数。native JSON serialization
和非法/缺失 traffic/identity 的 host tests 已覆盖。

## 实机：512² 路径接通，但不提升为整模型资格

同一 v3 CLI/library，原 BF16 transformer，fox/seed42，resident，
inference-time LoRA strength=1。Z 使用 distill patch、8步；Qwen 使用
Viggle v0.2.1 rank256、6步。1056-row/k1024/n512 activation-input 模板，
chunks1、fixed-async1、scale-cache/launch-fence/stage-specialize1，
prefetch/A8-lookahead/deferred0。两个 timed model arms 串行，不与 owned
build/test 重叠；host observer 连续记录。没有物理 engine trace。

| 模型 | actual/bucket rows | trial 采用 Fa | FFN rel L2 / cosine | trial GPU/candidate median s | 冷/热 request wall s |
| --- | ---: | ---: | --- | --- | --- |
| Z | 1056/1056 | 4096 | 0.0129042608 / 0.9999170430 | 0.080218125 / 0.058309792 | 26.614070 / 7.242443 |
| Qwen | 1024/1056 | 5120 | 0.0130570005 / 0.9999149733 | 0.107319833 / 0.100191875 | 24.872146 / 8.474602 |

两点均实际完成 90/180 correction traffic，raw vectors 经独立 verifier
复算。Qwen 两请求各192次 Private calls，无失败/retry/fallback，冷/热
PNG hash 均为
`9d5ac6e3a61123c977d514dd5b9e28f3957ff7fe3a331681ecdf103843ee07c7`。
这是固定样本一致性与功能接线，不是 GPU-vs-W8A8 latent/媒体资格。

**Z 冷请求并非稳定通过**：真实生成发生2次 overflow retry，program
headroom 从1升到16，calls=258；热请求另256次，累计 retry 仍2。
无最终失败/GPU fallback，但冷/热 PNG 不同：
`5d8f7b3574c41fe2fcb0c85468dc7a87118ee89dd5c31c8ed5cbeb605de37006` /
`55d0f457d792087d6d727fe87b65dd8de6d97a0caea333929c4cfe2c234b59ec`。
不能将低幅度合成 input 的 scale-one trial 当成 headroom16 的实请求
质量/性能证据。原 executor 会保留升高的 headroom；这可解释路径变化，
但尚未独立定位两幅图所有差异的来源。

因此 screen 的 accepted-native-auto 路线新增 recipe-drift gate：
实际累计 retry 必须0且 headroom 必须1；缺失、bool、NaN、其它值拒绝。
Z 上述两请求明确被拒绝，Qwen 通过 receipt contract；不改运行时重试、
不放宽数值阈值、不以 warm-only 选择性忽略 cold recipe change。
现有显式数字通道的重试策略不受此 gate 改动。通过 receipt 仍不等于
正式 E2E/质量/内存或物理 overlap 验收。

首版 correction upload 误用无 download 的 transfer，已修正为 DG/DU
validation-only download，未放宽 transfer guard。Qwen v2 又遭真实
growth-limit 拒绝，安全完成 GPU-only；v3 移除上传/验证后不必要保留
的初始 MX correction tensors，保留 immutable surfaces，预算不变。
v2 raw receipt 保留，但其 cold/hot PNG 路径被 v3 重用，不能再当作
v2 图像证据。后续1024与screen使用独立目录。

## 1024² LoRA：完整校准工作区超预算，正确 GPU-only

同一 v3 library、原 FP32 LoRA rank 算术、4224-row/k1024/n512 模板；
其余上述 flags 不变。Qwen 初次缺已有
`TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC=1` 被 request guard 拒绝，raw
receipt 保留；显式 opt-in 后才能继续，不删除/修改生产 guard。

| 模型 | actual/bucket rows | 已测 channel points | Private calls | 冷/热 request wall s |
| --- | ---: | ---: | ---: | --- |
| Z | 4128/4224 | 0 | 0 | 48.808271 / 40.065979 |
| Qwen | 4096/4224 | 0 | 0 | 49.925337 / 34.620188 |

两者完成 full-GPU baseline，随后在第一个 share 的 complete arena
admission 被拒绝，原因 `native calibration complete arena admission denied`。
最终 selected channels=0、complete=false、无 trial，无运行时 failure/
fallback/retry；两次生成都是 GPU-only。verifier 确认这是合法 decline，
不是1024 LoRA ANE 加速。resident hot 不再次校准，不误称 cache hit。

按当前 native surface/pitch/page 算法及 head upper 公式独立复算第一个
share（Z Fa4096、Qwen Fa5120；macOS16KiB pages）：Z arena+GPU upper=
2747.546875MiB；Qwen=3247.921875MiB，均超过现有 optional cap2048MiB。
这是**admission upper 重构**，不是实分配/峰值内存测量；没有把它
冒充库实际返回的分项 memory receipt。需要减少真实 retained scratch/
重复快照并重新审计，或测量 memory-admitted 的独立 point/批次算法。
不提高 cap，不把未测 point 或1024 base 旧结果当作本轮 LoRA 证据。

## 同库 matched screen：外部负载拒绝

Qwen512、同上真实 LoRA/六步/FP32 rank，同一 v3 CLI/library，
GPU→runtime、一冷三热，continuous load 与 process memory sampler
开启。GPU arm 的68个 load samples 中2个记录外部 ComfyUI CPU 活动，
最大46.4%；load verifier 拒绝，没有进入 runtime arm。
`outputs/native-lora-channel-q512-forward3-20261006/summary.json` 保持
incomplete、零认可 trials，raw JSONL/load/memory/PNG 保留。没有向
外部进程发信号、放宽门槛、用上述 diagnostic 或旧GPU分母补齐倍率。
load evidence SHA256：
`16a8c88870e738e17f53c0254e1320a6856f3b080902309af0cd933f3fc14c8c`。
这是未合格窗口，不是加速证据；尚无本轮 matched 整请求倍率。

## 回归、构建与证据

- 89项 shared host/calibration/screen/load/image/release guard 回归通过。
- Public Core ML/MLX/receipt 13项通过；新增 LoRA traffic JSON assertion
  的 receipt 子测试又实际重建并通过。Public actual flags/private class
  bytes/direct-link 发行隔离检查通过。
- Private MLX 三项集成通过：显式 channel LoRA/failure compatibility，
  actual native-auto ±strength rank4 fixture，以及 prepared calibration 的
  dynamic producer 异常/生命周期/refill/destructor 两 share 检查。
- 仅选择本轮 Z/results hunks；原 dirty draft changed-lines 逐项相同。
  staged-only 两个 cpp/mm 的 syntax checks 通过。不是完整 `make test`。

最终 v3 Private library SHA256：
`82b02e6aa639070221a5af9fef6c5fd83855d979d3d4c2a860546c0377115923`；
Public：
`5808823a04d31ba2efd82e79c86cf6f99b83eadb18e4f3634f10ad5dd34e558b`。
CLI：`02c55923be1dff7fc8dfd5681fd26b1cfaf15d8553eec28d1d9ce09b16d11ebb`。
实际 manifests 的所有 native source hashes 对当前 tree 无 mismatch。
这些都是保留旧 ConvRot drafts 的 working-tree builds，不能冒称
clean staged-only 完整重建；测试文件与screen verifier不在该native
source hash范围。model/adapters/references/设计原稿未改。

原始回执（各含 binary identity、观察记录、stdout/results 与 events）：

- `outputs/native-lora-channel-auto-z512-20261006/generation-v3-receipt.json`：
  `7b9e38876c1a770444a1eca72f88c199988955e841e0d3eb71265beed6e66aff`。
- `outputs/native-lora-channel-auto-q512-20261006/generation-v3-receipt.json`：
  `323d3011c6d8ce90831a3a69f959648fbefe30e0174c08dcfe2f339d616f4854`。
- `outputs/native-lora-channel-auto-z1024-20261006/generation-v3-receipt.json`：
  `8f34d02c0c1bc9f392879b3638459ccc964e06c9670d5d5a0f9e763873823651`。
- `outputs/native-lora-channel-auto-q1024-20261006/generation-v3-optin-receipt.json`：
  `1ea58e23b19bec863734d3353daeef271c0ea1db8e8782f4a81732b3b60fab64`。

## GPU、GGUF 和混合后端接续优先级

已有 register H256 与 bounded typed predecode 的实测/边界见
[GPU/有界解码报告](convrot-register-dense-window-2026-10-06.md)；Comfy
software BF16 boundaries 的整模型质量/延迟负结果见
[数值边界报告](convrot-bf16-value-boundaries-2026-10-06.md)。本轮没有
重测它们，也不将其旧 component 收益宣传为新 E2E 成果。

下一步先解决1024校准工作区和 Z LoRA 的 headroom recipe 一致性，
保留原 source precision 与全 operation GPU 重算。GGUF 消费侧应验证
真实跨调用 reuse/miss/eviction，以 `decode/R + dense GEMM < packed QMM`
为条件；两个矩阵槽不是两个完整 FFN 槽，不默认逐层 eviction 后的
提前解码能回本。暂不改磁盘GGUF/model、不生成永久dense sidecar。

Comfy 敏感激活/尺度恢复的 GPU handoff 与 Public/Private/GPU 按
operation/layer 选择仍待实际接入/成本验证。Public默认、Private显式
授权继续保持；同一片权重不同时常驻两套 backend 副本。多 chunk auto、
真实 GGUF auto 与完整四格 matched/reverse/multi-prompt ≥1.2×、最终
latent/感知/语义、process memory 与 physical device trace 均未完成。
