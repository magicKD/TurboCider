# Private W8 layer-ahead prefetch（2026-10-03）

接续 [channel split](private-ane-channel-split-2026-10-03.md)。目标仍 active：
两模型 512²/1024² 四格 ≥1.2×、完整 LoRA/画质、带宽与内存 qualification
未完成。不用预取 hits、组件通过或测试数替代整请求收益。

## 已进入实现层与模型路径

- 共用 Executor 增加 future-bank prefetch、按完整 source/selection identity
  激活与 discard 合同。Public 默认不支持、不编译 private API。
- Private W8 恰有两套 W banks；当前 worker 只读 current bank，独立
  staging worker 写 current^1，不复制完整模型 W8、不修改 base/LoRA。
  仅一个 future reservation，第三套请求拒绝。
- 激活必须 join 当前 ANE/epilogue consumer 与未来 GPU producer。source
  buffer/offset/full physical pitch/encoding/metadata/slice/recipe 都匹配
  才消费预取；不匹配丢弃并普通 restage。失败 producer 不发布 ready，不
  复用旧 weights。source allocation leases 保留到完成与激活/丢弃。
- 未来 weights 使用独立 GPU queue/event；A8 使用另一 queue，避免在 GPU
  ordering 上直接排到未来 W 后面。二者不推进 current ANE timeline。
  disabled 健康状态为 atomic；独立结果/ready slot 避免 current/future 数据竞争。
- `RowScheduler::peek_plan` 预测而不推进访问次数。HybridFfn 仅预取预计
  split 的下一层；同请求 plan/stage drain 不丢 future；adapter/request/
  cancellation/memory 边界显式 drain/discard。不消费部分失败输出。
- Z/Qwen 都提供已经 materialized 的 immutable 下一层 projections；只
  模型数学/weight callback，private API 不进入模型。Z 的 dense/affine
  source 提取合并为一个 helper，保留 physical layout 与 transform gate。
- JSON 包含 enabled/submissions/hits/discards/failures/exposed wait；screen
  增加 `--private-prefetch 0|1`，纯 GPU 分母清理 private 环境；校验完整且
  有界的 counters，不允许 failed/incomplete 伪装成加速成功。

## 关键验证

13 private host/hardware/MLX、13 Public Core ML/MLX/receipt、7 host、48
screen tests 通过；完整两类 build 与 Public class/link 隔离通过。

- 当前 A 在 flight 时准备 B，完整 A 输出仍匹配 A oracle；激活 B 后匹配 B。
- 未来 identity/recipe 不匹配拒绝，第三 reservation 拒绝；source weak lease
  在 producer/current completion 后、激活前仍 live，激活后释放。
- failed future producer 不破坏 current B；失败激活不能读旧 bank；普通
  stage 可恢复。显式 discard 后未来 key 不可激活。
- 真实 MLX/HybridFfn 两层 callback 路径有一次 submission 与 source-matched
  hit，source-identical 输出逐位一致；原 base/A/B/base、full hidden单次
  down-LoRA、tail padding、late-chunk whole GPU recomputation仍通过。

没有采集连续 device overlap trace，**不据两个 worker、queue 或 hit 数
声称物理 ANE/GPU/staging overlap 已完成 qualification**。

## 同库 Z512 消融：机制成功但性能负收益

Private pilot 库：`448c814d7d33579708ef65bdeb5c44ba0d821b5fd4b0d0388491b6c2de934482`。
Public regression：`2a1433339c3ba1b16ada6e0e120f1d0e952e53698c2799fb224bc1a25e3ac67f`。
Z512/8 steps/fox/seed42，Fa3072/c1056/chunks=1，GPU→Private，一冷两热，
相同 GPU 算术，关闭 profile。完整 request wall 含 VAE/PNG、不含冷请求。

| 配置 | GPU hot median | Private hot median | 相对GPU |
| --- | ---: | ---: | ---: |
| prefetch=0 | 6.998585 s | 6.603004 s | 1.05991× |
| prefetch=1 | 6.988936 s | 6.984510 s | 1.00063× |

开启组累计744 submissions、744 hits、0 discards/失败；exposed prefetch
wait=0.000110 s，但 shared GPU work/memory 竞争抵消收益。不能把 hits 当
speedup；不拼接旧库分母、不认为达到1.2×。原始目录：
`outputs/private-ane-prefetch-z512-{off,on}-20261003/`。

实现与开关保留；因已测负收益，后续 source 默认值恢复 explicit opt-in：
`TURBOCIDER_PRIVATE_ANE_PREFETCH=1` / `--private-prefetch 1`。默认 Public、
普通 private row/channel 都不隐式采用该负收益 policy。上述448c库是默认值
修正前的消融身份，最终 build 不借用其性能身份。

继续：按 stage/current-GPU/ANE/UM 带宽竞争做校准、scale cache 与 source
lease integration、实际重叠 trace、1024²/更多 LoRA 与媒体/latent资格，
四格 matched 正反序 ≥1.2×；没有缩小或宣布完成原目标。

未改发行 build/native、用户模型、参考仓库/设计稿，未 stage/commit。

## 跨日接续：2026-10-04（Asia/Singapore）

Opt-in 默认值修正后的 Private library：
`da684a3082a7556c4cc434a934f70c2f8e7f24a972800f6bb7c58b23fcac05af`；
Public：`e6c307c2f04f4b47f44d1e096c391d9bb4bab6a90cacca15a8c4b6c50970bff0`。
最新两类 full build、13 private（显式prefetch=1）、13 public、7 host、48
screen 和 Public class/link 隔离通过。不是完整 make test/目标验收。

新库 Z1024/8步/Fa3072/c1056/chunks=1、prefetch off/on、同库 GPU 分母
对照已完成，目录为 `outputs/private-ane-prefetch-z1024-{off,on}-20261004/`。
off：GPU median=31.258481 s、Private=34.351492 s；on：GPU=31.250903 s、
Private=34.230986 s。两者仍负收益，未达到1.2×；完整回执与无 error
fallback 不等于性能资格。此两组不能替代 Qwen1024/LoRA/画质验收。
本组 bench 句柄已终止，没有仍在运行的本组推理。
