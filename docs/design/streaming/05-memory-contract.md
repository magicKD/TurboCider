# 05 · 给定布局的内存合同与可选 guard

[目录](README.md) · [布局](02-configuration.md) · [验收](07-tooling-and-validation.md)

## 1. 不从预算倒推，不代表不估内存

manual 模式不搜索 K/G/P，只计算用户布局需要多少资源。编译分两步：

```text
describe + manual layout -> 确定性 ResourcePlan（不需要 Y）
ResourcePlan + baseline + 可选 Y/X/S -> admission（不修改布局）
```

只指定 K 能约束池数量，不能保证 activation、text encoder、VAE、export、framework temporary 也放得下。
例如两槽每槽 0.5 GiB，却有 8 GiB weights prefix 和 12 GiB activation；“两槽低内存”并不成立。

layout-only report 可以显示未知资源和预测区间，但不得显示 bounded=true。bounded 请求有 required unknown upper 就拒绝。

## 2. 正向容量计算

对 group g，adapter 给出真实格式/布局所需的唯一 backing bytes `W(g)`，包含 scales、padding、alignment。
静态分配 `slot(g)=ordinal(g) mod K`。pool slot s 的容量为：

```text
capacity(s) = aligned_capacity(max requirement of compatible groups mapped to s)
pool_bytes  = sum(capacity(s))
```

如果 group layout 不兼容，不能只取总 bytes 最大值混用：必须为每个 tensor field 做可证明的兼容 layout，
或拆 pool/class、在确定 barrier 重新分配；pool 重建时的临时重叠进入 live intervals。
上述 requirement 是 storage bundle 的布局上界，不是不同 group 总 bytes 简单取 max；固定字段 backing 必须逐字段取 max 后求和。
`(320,64)` 与 `(64,320)` MiB 的槽容量反例见 [09](09-layout-compiler-spec.md)。首版 class barrier 要求旧池先释放再建新池。

简单同构场景只可用下式解释大致组成：

```text
weights = resident_prefix_unique_bytes + pool_bytes + mandatory_small_weights
peak = max over epochs(
    residual_process_baseline + unique live model backings
    + reserved but unallocated capacity + pending releases
    + staging + activations + conditioning + outputs
    + control/worker/mailbox/trace + framework_unattributed_upper
)
```

real compiler 按唯一 storage/alias 和半开 live interval 求峰值，不把每个阶段独立最大值直接相加。
也不能既把整个 “activation reserve” 算一次，又把其中的 scratch 每项再算一次；manifest 标明 envelope 包含范围。

## 3. 可核算例子（合成，不是任何模型实测）

假设同构 N=12 blocks、每块 256 MiB、P=2、G=2、K=3：

- prefix=512 MiB；suffix 10 blocks 组成 5 groups；每 group 512 MiB。
- pool=3×512=1536 MiB；mandatory constants=128 MiB。
- 某峰值时 activation=1024 MiB、staging=128 MiB、control=32 MiB。
- baseline+framework=512 MiB。

该峰值上界为 3872 MiB；后续 VAE 若是 5000 MiB，则 whole-request 上界至少为 5000 MiB，不能只报 denoiser 3872。
减少 K 能减少 pool，但不会降低独立 VAE 的 5000 MiB 峰值。Y=6 GiB、X=15% 的 B 约 5222.4 MiB，
是否放得下仍取决于 VAE 的 5000 是否已包含其 baseline/outputs 等全部同时存活项。

## 4. Bounded-manual 的预算规则

保留已有语义：`B=floor(Y*(100-X)/100)`，X 是保留比例，不是允许使用比例；整数计算 checked arithmetic。
S=`min_free_bytes` 是系统余量约束，不能再作为 Y 的一部分反复扣减，也不能把 swap.used 当 RAM。

admission 至少满足：

```text
compiled_request_upper <= B
additional_increment_upper <= reliable_system_available - S
all required sites covered and qualified
layout identity and memory evidence match exact execution
```

baseline 已包含的 backing 不得再双算：首版 clean session baseline 与之后新增模型 storage 分开；
未来 retained cache 采用 session-root claims + residual baseline + request increments，每个 storage 只计一次。
共享映射/child helper 的统计口径要明确，不盲目把 parent RSS 与 child RSS 当作真实唯一物理总量。

认证决定能否执行；guard 决定该次基线是否允许执行。请求不能注入 `framework_upper=0`、manifest 或认证 JSON。

## 5. Guard 不是系统硬分区

纳管分配前 reservation 可阻止超额 pool/tensor 分配。OS、driver、MLX/MPSGraph 等未显式纳管部分，
依赖已验证 envelope 和观测；不能从低频采样证明连续时间峰值从未超过 B。
结果区分 `planned_upper`、`managed_peak`、`sampled_process_peak`、`framework_upper`、`observation_interval`。

未覆盖的 required sites/envelope 表示不能认证 bounded，而不是“多留点 buffer 就一定安全”。
系统其他进程仍能制造 swap；zero system swapout 是保守 acceptance 信号，不是此框架可独占保证的 OS 行为。

guard 开启时全局 swapout 增长导致请求 fail-closed；离线 campaign 若确认外部噪声，标记 inconclusive，不能改写 PASS。
layout-only 的诊断观察到 swap 仅说明该布局/机器组合不适合 bounded 发布，不假称自己违反未承诺的用户 Y 上限。

## 6. 手动布局与 capability 的绑定

拟议 execution identity 必须包括：

```text
checkpoint + model variant + shape/token bucket + operation + dtype
backend/device/runtime/adapter/descriptor revision
every stage residency, groups, K/G/P/D/Q, slot assignment and pool layout
retention policy + schedule revision + required-site closure
framework envelope revision (bounded only)
```

资格分成两条独立审核：

| 执行类型 | 必需证据 | 不承诺 |
|---|---|---|
| layout_validated | exact layout、质量、真实 fence/slot reuse、取消/cleanup、默认零回归 | Y 上限、零 swap |
| bounded_certified | 以上 + required-site closure、whole-request upper、L2/L3、观测可信 | 所有其他设备/shape 自动可用 |

首发 exact tested tuple，不做未经证明的范围认证；推荐工具只枚举已发布的 tuple。
未认证组合可以 plan/simulate，不可普通 generate；实验 build 必须显式标 experimental，不能写进生产 registry。
新 framework 的 layout-only 入口不调用旧 legacy 实现伪装验证通过，guard=false 不构成逃生通道。

## 7. 过大配置的反馈

```text
layout_rejected:
  stage=video_vae, epoch=decode_tile_0
  required_bytes=..., available_bytes=..., shortfall_bytes=...
  adjustable_weight_bytes=...  # 若峰值不是 weights 主导可能为 0
  hints=["选另一个已认证 tile preset"]  # 只建议，不执行
```

unsupported layout、unknown envelope、budget不足、实际分配超限、swap activity、cleanup failure 用不同错误类别。
不要对所有失败都提示“减少 slot”：如果 peak 来自 VAE，减少 DiT slot 可能完全无效。

## 8. 大内存与 retained session

legacy 默认不启用新 guard，保留原缓存行为。显式 resident+guard 仍走所有必要校验，不因 RAM 大偷偷关闭 enforcement。

首发 request retention 释放本请求 backing，可能比默认 retained resident 多加载；性能报告必须说明这种差异。
未来 session retention 要有 root ledger、storage identity、cache eviction、generation 和 owner handoff，
terminal 判定“request-local resources zero + validated session claims retained”，不能直接删除现有 storage_count==0 gate。
该扩展单独 PR/认证，在此之前 UI 不展示为可执行模式。
