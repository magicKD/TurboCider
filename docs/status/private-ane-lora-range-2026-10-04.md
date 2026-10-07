# Private channel LoRA：只生成 ANE 子范围 correction

接续 [dense-load](private-ane-dense-load-2026-10-04.md)。原目标保持 active：
Z/Qwen base 的512²、1024²四格正式 ≥1.2×、完整 LoRA/画质/内存和实际
device overlap 尚未完成。本轮取得新的执行与数值兼容证据，**没有新的
有效端到端倍率**。

## 实现

- 共用 `HybridFfn::Adapter` 增加可选 `gate_up_channels(input, first, count)`。
  Private channel executor 只请求 ANE tail 对应的 gate/up corrections，
  不再先生成完整 intermediate width 再切片。返回 geometry 明确检查。
- Z 使用已有 `Weights::lora_delta_slice`；Qwen 提供 request-local compiled
  range graph，分别映射 gate 和 up 的物理 output rows。GPU partial 继续
  负责自己的 channels；rank dtype、FP32 accumulation、BF16 rounding、
  adapter scale/order 不变。没有合并或修改 checkpoint/adapter weights。
- Public/row executor 仍用完整 callback；没有 range callback 的 caller
  保持原行为。完整 corrected hidden 仍只 join 一次，down-LoRA 只执行
  一次；late-chunk failure 仍完整 GPU 重算，不能发布部分 ANE scratch。
- `TURBOCIDER_RUNTIME_ANE_LORA_CHANNEL_RANGE=0|1` 进入 executor identity，
  未设置时在 callback 可用的 channel 路由启用；0为同库旧路径消融。
  screen 增加 `--private-lora-channel-range`，必须提供 private W8A8 channel
  LoRA，不能用 zero-chunk GPU ablation 伪装验证。
- JSON 增加实际 callback session counters：
  `lora_channel_range_calls_session_total` / `lora_channel_full_calls_session_total`。
  verifier 检查完整、非负、单调和显式选择的实际执行；不能只依赖 env intent。

## 验证

- 13 Private host/hardware/MLX，13 Public Core ML/MLX/receipt，53 screen，
  7 runtime host，6 memory-runner，4 load-observer，8 layout，3 release guard
  通过；Public 新库通过实际 flags/strings/links 发行隔离检查。
- Private fixture 覆盖 base/A/B/base、17/67 valid rows、tail padding、
  narrow/full逐位一致、一次 full-hidden down-LoRA、显式 off、旧 caller
  无 range callback、malformed geometry拒绝、晚 chunk完整 GPU fallback。
  初次 fixture 的错误异常类型捕获已修正；最终13项完整回归通过。
- Z512 distill-patch 与 Qwen512 Viggle r256，fox/seed42、各两个热样本，
  同库关闭/开启后输出均逐像素一致，max uint8 error=0。
  这证明这些固定样本的兼容，不替代多提示词/latent/媒体质量批准，或
  1024² LoRA 的完整模型测试。
- Full correction counters：Z每请求256、Qwen每请求192；narrow开启后
  全部切换到 range counters，另一类为0。程序 self-test 没有代填这些计数。
- `git diff --check` 通过。未在本轮运行全量 make test；已知 Qwen3 stale
  source-string assertion 未修改。Public默认与 private feature gate 保持。

构建目录 `build/{private,public}-ane-lora-range`：

- Private：`0b798ae07d2754f6e6aa892663180a389e4bd49726877a689c746fde34bec408`。
- Public：`36a77e60ea754e578f810808a74fde975f9eb87cbd4d5f9b4b0c1053cfc9ba76`。

## 为什么仍不报告性能提升

四个 arm 均成功完成一冷两热的完整生成，手动用相同 contract helpers
验证了实际 executor/data-path/LoRA/Q-K 选择。但连续 CPU observer 检测
到匹配活动，summary 保持 incomplete：

| arm | samples | busy samples |
| --- | ---: | ---: |
| `private-ane-lora-range-z512-off-20261004` | 52 | 1 |
| `private-ane-lora-range-z512-on-20261004` | 43 | 2 |
| `private-ane-lora-range-qwen512-off-20261004` | 60 | 2 |
| `private-ane-lora-range-qwen512-on-20261004` | 47 | 2 |

CPU heuristic 不证明 GPU 干扰或独占；也不能忽略预先定义的 gate，借旧库
GPU分母、raw wall或子范围输出大小推导本次加速。没有终止外部进程。
所有本轮 inference/test/build handles 已 terminal。

机器记录见 [LoRA range pilot](../design/validation/private-ane-lora-range-pilot-20261004.json)。

## 后续

在有效同库正反序窗口测四格 base 与 LoRA；继续实际 GPU partial/rank
投影、带宽-aware share/prefetch calibration 和 GPU/ANE epilogue overlap。
本次 narrowing 只减少多余的 correction 输出，不等于已经消除了 rank
投影、join、量化或共享内存成本，不能缩小原来的 ≥1.2× 和质量目标。
