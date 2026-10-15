# 低内存协作：阶段预算、固定子集和防抖反馈

[设计目录](README.md) · [本地策略](../labs/contracts.py) · [区间预算工具](../scripts/interval_peak.py)

## 从 vpipe 吸收的关键区分

内存优化不是只算压缩权重大小。计划应先让 encoder/DiT/VAE/媒体消费者全部 declare，再统一 decide。否则先初始化的模型可能把后续激活或输出所需空间当自己的可增长余量。一个 allocation 的 alias 只计一次；跨阶段存活取区间并集，不把同一 latent 的不同名字当两份，也不能提前释放仍被异步消费者借用的 backing。

本地 [memory-plan.json](../examples/memory-plan.json)与 `interval_peak.py`实现了 phase 区间、alias backing 和 target 预算；[slot state machine](../labs/systems.py)补充实际完成前不可复用的约束。预算是声明下界，不是 OS RSS/phys_footprint 的承诺。

## 为什么循环扫描不宜用 LRU

N 个 block 按 `0..N-1`循环，容量 C<N，纯 LRU 每次把下一次将要访问的 block 逐走；warm 轮可零命中。固定保留 C 个 block，不为 admission 抢别人的座位，每轮稳定命中 C 个。`residency-policy`用 N=8/C=3 的三轮确定性 trace 同时运行两者，具体展示收益来源，未声称任何 SSD 速度。

## 可复用策略的最小算法

`FixedResidency`不持有真实模型权重，只管理策略记录：

1. reserve 未申报时不增长；已申报 0 与“未知”不是一回事。
2. 令 `remaining_reserve=max(declared_reserve-already_allocated_reserve,0)`。free/headroom 已扣掉现有 allocation，不能再保护同一 scratch 两次。
3. admission 要求足够容纳 candidate、remaining reserve、额外一个 block 的 slack；不 evict 旧 block 来保新 block。
4. 实际压力或权重已被压缩/换出时，请模型在 last-reader completion 后真实释放。确认后调用 `release_completed`，ceiling 降到剩余大小，阻止立即长回。
5. 数个完整健康 forward 后，adapter 可显式调用 `healthy_probe`小幅重开；不得每个 block 都重置 ceiling。

样本不读取系统 free，不实施系统 wired cap，也不声称测到了真实 resident pages。生产适配要补 pressure、实际 resident/compressed、peer parking 的观测与安全释放接口。只有 file-backed 变多不证明有更多余量：那些页可能正是自己刚 streaming 的 checkpoint。

## 与编译、ANE 和媒体协同

ANE program、IOSurface、双 W bank、GPU partial、dequant scratch、VAE activation、输出 frame queue 都进同一 ledger。卸载 program 不自动释放自己创建的 surface；模型 park 也不自动取消已提交的命令。Conditioning 可以在 encoder 释放后继续存活，因为消费者可能是每个 DiT step。

对每个 phase 记录：fixed weights、resident subset、streaming slots、activations、跨阶段 payload、library/service unknown、safety reserve。声明、决策、真实分配、峰值采样、真实 release 是不同证据，不能互相替代。

```sh
python3 -B techniques/labs/run.py residency-policy
python3 -B techniques/scripts/interval_peak.py techniques/examples/memory-plan.json
python3 -B techniques/labs/run.py slot-ownership
```

反例：LRU 循环 thrash、reserve 重复扣除、admit 导致 evict、evict 后立即长回、只凭 available/filecache 推断驻留。新 RAM、phase overlap、实际低内存设备、block bytes、编译数量或输出队列改变后重新验收。出处见 [吸收账本](../references/absorbed-designs.json)。
