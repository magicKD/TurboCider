# Private W8A8 channel split 接续（2026-10-03）

接续 [W8 Executor](private-ane-w8-executor-2026-10-03.md)。原任务仍 active：
两模型 512²/1024² 四格 ≥1.2×、LoRA/画质、层间 double-buffer staging、
带宽校准尚未验收，不以新的分区兼容性或单向 pilot 替代。

## 已实现到模型层

- 共用 `PartitionAxis::{Rows,IntermediateChannels}` 和 whole-block
  scheduler，保留 layer/rows/adapter 隔离、GPU probes、warmup/迟滞。
  Channel 模式不错误调用 row chunk balancer；row path 不删除。
- `DeviceWeightRegion` 为 logical selection + 完整 immutable physical
  source。gate/up 选择尾部 output rows；down 选择尾部 input columns。
  Q4/GGUF 的 stride/metadata/owner 保留原矩阵，不修改或复制 checkpoint。
- Factory 先以完整模型几何校验 template，再生成较小 Fa 图。
  `TURBOCIDER_PRIVATE_ANE_CHANNELS` 须 512 对齐、正数且小于全宽；仅授权
  private W8 backend 接此 candidate。Public 默认、FP16 row path 不变。
- `HybridFfn::run_channels` 对全部 token 并行 GPU [0,Fg)/ANE [Fg,F)。
  非整 bucket 只 GPU pad activation/corrections，输出裁回真实行数。
  output/hidden/padding/join scratch 分配前做机会性 memory admission。
  任意 chunk 失败后重算整个 FFN，不消费部分 scratch。
- `ChannelGpu` 返回 base-down partial 和 corrected hidden。base partial
  FP32 add → 原 dtype。两段 hidden 合成**完整 F hidden**后，原 down-LoRA
  callback 仅执行一次，不按 channel/chunk 分别舍入、不合入 base weights。
- `Weights::project_base_slice` 拆出 checkpoint-only 算术；原
  `project_slice` adapter/bias 顺序不变。Z/Qwen base 与 LoRA 调用已接入；
  Z GGUF 可走 affine source（实模型 GGUF channel qualification 尚未执行）。
- Z MPP `projection_range` 直接读取原 physical pitch/row-col begin；down
  不复制 compact dense W。`swiglu_gemm_range` 保留原 BF16 epilogue。
  旧 GPU baseline 算术不改变，模型不接 private API。
- Backend/data-path/GPU-I/O/channel 配置进入共享 executor identity，避免
  resident 会话复用错 graph。JSON 包含 axis/Fg/Fa/channel blocks；screen
  `--private-channels` 校验实际分区，纯 GPU 分母清除所有 private 环境。

## 已执行验证

M4 Max 64GB：13 private host/hardware/MLX（含新增 channel test）、13
Public Core ML/MLX/receipt、7 host、47 screen tests 通过。完整两类 build
成功，Public 无 private class strings/PrivateFrameworks link。

BF16/FP16 MPP wide pitch、非零 row/col begin 与 tails 对 compact reference
同内核逐位一致（不是与另一套 GEMM 顺序 bit-exact）。Hybrid channel 的
17/67 行、pad/多 chunk、immutable source、base/A/B/base、单次 full-hidden
down-LoRA、late-chunk whole-GPU fallback 通过。

相消 fixture 原始输出 relative L2=0.0668762；partial + adapter 能量尺度
L2=0.029908，门槛0.04。原始0.06总输出门槛未通过；保留原误差，不以
cancellation-aware operator check 代替实模型画质验收。fixture 的空
`apply_loras` 不是 detach API，按模型流程 clear/rebind 后 base 返回逐位一致。

较早 channel build `506e54e50ce990319d5c7f49fcabfee7ff5ff23f1e8ae1bd43353d62ecf916db`
已过 source/join regression。保留原 Z GPU epilogue 的 final private pilot：
`ccea59611ff7f0a456f425b18075dc48dee6ae61a99556dd8279498154bdfb21`。
Public regression：`144c9597e69be5e7832f7f6fc7dd77ccd83438a51df5ba92e5ed07f9fb2180ea`。

## 初筛与后续

Z512/8步/fox/seed42、Fa1536/chunks=1、GPU→Private、一冷两热；固定分区
强制每个 block 做全 token channel split。目录
`outputs/private-ane-channel-z512-a1536-20261003/` 完整回执通过，768 calls
每请求；GPU hot median=6.993819 s，Private=7.742688 s，约0.903×。
无 error fallback，但仍负收益；不是正式 performance/quality qualification。

同一 ccea 库、同提示词与工作量，改用完整1056行 bucket：

| Fa / bucket | GPU hot median | Private hot median | 倍率 | 每请求 calls |
| --- | ---: | ---: | ---: | --- |
| 1536 / 1056 | 6.989436 s | 6.884262 s | 1.0153× | 256 |
| 3072 / 1056 | 6.987885 s | 6.605137 s | 1.0579× | 256 |

完整回执通过、无 error fallback，均实际运行全部 channel blocks；没有
把 GPU decline 当 ANE 加速。整行 bucket 消除了多个 chunk 串行交接的
部分暴露成本，仍未达1.2×。原始目录分别为
`outputs/private-ane-channel-z512-a1536-c1056-20261003/` 与
`outputs/private-ane-channel-z512-a3072-c1056-20261003/`。

Qwen512/40步、同样 Q/K norm-RoPE 与 Fa3072/c1056、GPU→Private/两热，
以及现有 Viggle r256/6步/512² LoRA 的同库对照已串行启动。目录
`outputs/private-ane-channel-qwen512-a3072-c1056-20261003/` 和
`outputs/private-ane-channel-qwen512-lora-a3072-c1056-20261003/`。未完成
summary 不填性能。此两组随后已完成：base GPU=41.716553 s、Private=
47.805042 s，0.87264×；r256 LoRA GPU=8.137791 s、Private=10.665939 s，
0.76297×。两组均无 error fallback，仍负收益，不提升默认。
下一层 bank pipeline 接续见 [prefetch](private-ane-prefetch-2026-10-03.md)。

仍须两模型/分辨率/现有 LoRA 的 fixed/auto channel 对照、share/bucket
搜索、实际 latent/media 误差、source leases/scale cache、next-layer
prefetch/reuse fences、实际 GPU-stage/ANE overlap、带宽/内存/热态校准，
以及四格 ≥1.2× 正反序正式验收。两个 bank 目前仍只轮换，无层间重叠资格。

```sh
TURBOCIDER_TEST_PRIVATE_ANE=1 TURBOCIDER_TEST_PRIVATE_CHANNEL_MLX=1 \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/private-ane-channel-final \
  .venv/bin/python -m unittest discover -s tests/native -p test_private_ane.py -v
```

未改发行 build/native、模型、参考仓库、设计稿，未 stage/commit。
