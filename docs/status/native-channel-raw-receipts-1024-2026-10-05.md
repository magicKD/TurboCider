# 原生 raw calibration 回执、cache hit 与 1024² 诊断

实验采用 UTC 2026-10-05，本机 Asia/Singapore 日志为 2026-10-06。
接续 `41d4524` 的原生 channel auto。完整双后端、四格 base≥1.2×、
真实 LoRA 加速和正式质量/内存/device trace 目标仍 active。

## 实现

将 SDK/MLX/Private-independent identity、baseline/point samples、trial
和 report data 统一放入 `native/core/ane_calibration_report.hpp`，原有
qualified Private API 以 alias 保持源码兼容。实际 native result 的
`hybrid.runtime_weight.channel_calibration` 导出：

- 身份、实际/bucket geometry、五个深度、warmup/hot counts及scope。
- full GPU one/four 两组完整 raw vectors 和 median 差导出的 layer time。
- 两个 share 各 GPU/ANE/Both×one/four raw vectors，每 cell 七个 hot
  samples、每 point 90 次 ANE calls（含 warmup）。
- proposal、实际 four-FFN GPU/candidate raw vectors、calls/fallback/
  retry、独立 FFN rel L2/cosine、最终采用/拒绝原因。

accepted cache entry 保留独立 immutable report。命中时复制 report 并
只更改 cache-hit 标识，不改原始 samples，也不 pin 第二份 model。
未知/非有限统计在 JSON 中显式 null，不伪造为 zero/pass。timing scope
仍为完整 FFN host span，不是 E2E 或物理 engine timer。

审查发现原质量聚合的 min/max 可能掩盖 NaN：有限 candidate 对
非有限 reference、或者 FP32 reduction 溢出时，NaN可被旧0/1聚合值
掩盖。现在同时检查 candidate/reference finite，并拒绝非有限delta/
norm/dot或zero energy，失败状态不能被之后的有限样本修复。没有
放宽原 rel L2≤0.03、cosine≥0.999；这些是独立 FFN门槛，不是最终
latent或媒体资格。

screen 新增 `--private-channels auto`，保留显式整数模式。新 verifier
从所有 raw samples 复算 one/four median 差，并检查几何、身份、
实际 trial 的数值/5%窗口收益/calls与采用的physical width。可清楚
记录合法 GPU-only decline；失败 fallback仍拒绝。没有绕开原
competing-load 或 process memory checks，不能用离线 proposal 冒充
实际采用，不能将 per-FFN trial通过提升为 E2E通过。

## 验证与实机

最终 Private library SHA256：
`8adce80393a5062200a339e7816d60d36c822943f2c26c0c71b0f73d3135ad1f`。
Public：
`ae9bf50504c47c427a55707491dd9078ff4b0c581075e2abe5de200b03ef03b0`。
都是保留原有 ConvRot drafts 的 working-tree builds，不是 clean
staged-only重建；所有模型计时 arms 串行、不与 owned build/test 重叠。
Public actual flags/private class bytes/direct-link发行检查通过。

- 11项 shared host、6项新 calibration verifier、60项既有 screen
  host通过；涵盖raw tamper/缺失/非有限、错计时scope、trial不足、
  参数/physical width不匹配与 clean decline。Public 13项
  Core ML/MLX/receipt通过。
- 新 Private auto fixture实机通过，确认raw vectors确实保留而不是
  只报告suggestion。fixture无收益时为GPU-only；不当性能证据。
- serialization新增未知quality→JSON null、raw counts保留检查；初次
  编译因ObjC `auto`推导成id被-Werror拒绝，改用明确NSDictionary类型，
  不改生产计算或验收门槛。

### Actual cache-hit

Z512 base、fox/seed42、8步。一个batch先普通manifest、再同geometry
不同path的副本以强制重建executor、然后保留该executor。没有改原
manifest/model。三次都采用Fa4096，回执cache flags分别false/true/true，
每请求256次Private calls、0 fallback。五层/两点/actual trial raw
vectors保留且verifier复算通过，cache原始sample不变。
独立FFN trial rel L2=0.0128320520、cosine=0.9999180818；four-FFN
GPU/candidate hot medians为0.065567083/0.048206792秒。
这些不是完整请求的速度倍率。强制重建会开启新的executor counter
epoch，不能按普通未重建resident batch的单调累计计数验收。

### 1024² native auto

original BF16 base、fox/seed42；4224-row/k1024/n512模板；chunks1、
fixed-async1、scale-cache/launch-fence/stage-specialize1，prefetch/
A8-lookahead/deferred0；每模型一冷一热，Z8步、Qwen40步。

| 模型 | actual rows | 自动Fa | calls / 请求 | FFN rel L2 | FFN cosine | 冷/热request wall s |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Z | 4128 | 4096 | 256 | 0.0128445491 | 0.9999179063 | 64.651304 / 26.577012 |
| Qwen | 4096 | 5120 | 1280 | 0.0130448740 | 0.9999153056 | 202.803729 / 151.312072 |

两模型均真正采用Private中间通道路径，两次生成无failure/retry/GPU
fallback；原始baseline、两点、实际trial vectors与geometry经独立
verifier复算通过。Qwen four-FFN GPU/candidate trial medians为
0.328915083/0.214994208秒。Native质量validation calls=0仍不作为模型
质量验收。这里没有同库GPU整请求对照、多hot/反序/多提示词或严格
load资格，**不计算整模型倍率，不借用历史GPU分母**。大bucket相对
既有小bucket/多chunk的真实整请求收益还需比较；不能因FFN trial
通过就把它推荐为已经快1.2×。

## 不合格的完整请求窗口

新screen入口，Z512、GPU→runtime、一冷三热、同最终库，连续load和
memory observer开启。GPU arm的55个load samples中有2次外部推理
CPU活动，verifier拒绝，未进入runtime arm；summary保持incomplete、
零认可trials。原始JSONL/load/PNG保留，不放宽门槛或向外部进程发
信号。不能计算这个失败窗口的速度倍率。

## 原始证据

- `outputs/native-channel-report-z512-cache-20261005/generation-receipt.json`：
  `711754a878a09bd0d79cfa889c2ed78a39a9f708654cdcf3d6ada52a6db49806`。
- `outputs/native-channel-report-z1024-20261005/generation-receipt.json`：
  `5f4ad179b0c34b119f84c4f6bee496a8f31f42af9d106169f5f7cc92f6cb4200`。
- `outputs/native-channel-report-q1024-20261005/generation-receipt.json`：
  `3864cae5a3dede6b5a9a0cd18fde6abecd5abbbdfd5279f12e8180209ee01b0f`。
- 不合格screen：`outputs/native-channel-report-z512-forward3-20261005/`。

模型、adapter、参考实现与设计原稿不改，原有未提交drafts保留。
继续工作仍需matched/reverse/multi-prompt四格≥1.2×、最终latent/
感知/语义与process memory/device trace；native LoRA-aware/multi-chunk
calibration、Comfy敏感算术GPU handoff、Public/Private/GPU per-operation
选择、GGUF bounded decode真正复用都不能由上述diagnostics替代。
