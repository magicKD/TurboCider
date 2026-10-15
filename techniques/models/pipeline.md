# 整体模型加速：先保语义，再消除阶段瓶颈

[阶段地图](stage-map.md) · [驻留设计](../designs/residency-and-phases.md) · [本地测量样本](../labs/media.py)

## 分阶段计账，不把重叠时间加两遍

Cold 请求包括 source 验证、权重映射/安装、graph compile/load、first prediction；warm 还可能有 encoder、denoising、VAE、upscale、编码/写盘。Span 嵌套时不能直接相加；并发阶段应看区间并集和关键路径。[profiling-scopes](../labs/media.py)给出 union 与错误求和对照，[compare_runs](../scripts/compare_runs.py)拒绝非匹配输入/失败样本的伪倍率。

假设串行 wall time 的 DiT 占 70%，其余占 30%，即便 DiT 无限快整体也不超过 `1/0.3≈3.33x`；这个是假设性的 Amdahl 下界提醒，不是本项目成绩。优化后必须重新定位 floor，encoder/媒体/VAE 可能成为新主导。并发的真实图要按关键路径算，不能套串行占比公式。

## 按投入顺序建立候选

1. 减少重复 cold 工作：verified source cache、shape/recipe 编译缓存、phase-scoped program，分别记录命中与首次税。不要缓存不完整输出或以路径替代 source 内容身份。
2. 保持单阶段 GPU 数据流：pack/consumer、fusion、online attention、少提交，先有优化 GPU 控制。
3. 收紧 phase peak：encoder 完成后释放无需的 weights，保留 conditioning；DiT resident subset 与 VAE reserve 协作；bounded writer 不保留全视频 RGBA。
4. GPU/ANE split 只占合法独立边界，完整 layer 和完整 request 都必须盈利；短 shape 可回退 GPU。
5. 最后在明确质量许可下搜索量化、sparse、residual reuse、token merge/较低 step 等方案，并单列改变工作负载的候选。

## 什么时候可以跨阶段并行

独立请求 A 的 VAE 与 B 的 encoder 可在容量允许时 pipeline；同请求若 DiT 仍需要 encoder 未完成的 conditioning 则不能提前执行。一个可按帧消费的 VAE 也要先证明时间感受野和 output readiness，不能任意按未来帧切开。后台 prefetch 只隐藏独立 IO，不抹去 source、layout 和 last-reader 依赖。

统一内存使 CPU/GPU/ANE 可访问共享 backing，不等于没有 layout pack、cache coherence、同步和带宽争抢成本。更多并行可能提高功率同时增加能耗或 wall time；统计整请求能量，而非只看某阶段瞬时功率。

## 可复用的阶段验收清单

| 边界 | 必须固定 | 必须观测 |
| --- | --- | --- |
| Encoder -> conditioning | tokens/图片内容、mask、positions、取层/归一化、source/adapter | 完整 conditioning 输出和 cache invalidation |
| Prefill -> KV/decode | causal policy、global positions、KV head/dtype、sampler | TTFT、KV 完整性、后续 decode 连续性 |
| DiT steps -> latent | seed/noise、steps、sampler/CFG、source/LoRA、cache policy | 中间误差、最终 latent、整图/视频质量 |
| VAE -> frames | latent scaling、norm/halo/phase、range、alpha、FPS | seam/时间连续、峰值、所有帧实际完成 |
| Writer -> 成品 | codec/尺寸/帧数/音轨/文件范围 | flush/close 成功、可解码、取消不发布半成品 |

```sh
python3 -B techniques/labs/run.py startup-amortization
python3 -B techniques/labs/run.py profiling-scopes
python3 -B techniques/labs/run.py media-backpressure
python3 -B techniques/scripts/interval_peak.py techniques/examples/memory-plan.json
```

小型样本用于证明机制，不代替真实模型上下文、低 RAM 实机、artifact size、driver/service footprint、功率和任务质量。重新探索条件是瓶颈、phase graph、硬件、shape、source 或质量预算改变；每次都保留失败和 fallback 证据。
