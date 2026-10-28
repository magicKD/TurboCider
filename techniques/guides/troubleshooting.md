# 症状 → 证据 → 可能的解决方案

<!-- knowledge
id: technique-guides-troubleshooting
type: guide
-->

[总目录](../README.md) · [先确认计时口径](../profiling/measurement.md)

以下是排查顺序，不是看到症状就直接改默认。先保留失败输入/日志和构建身份，再选单个变量实验。

| 症状 | 先检查的证据 | 下一步 / 不该做什么 |
| --- | --- | --- |
| hot快、第一次很慢 | compile/hash/load/binding/warmup/first predict分别计时 | [load-only与Session复用](../ane/lifecycle.md)；不隐藏prepare时间 |
| 512突然比以前慢很多 | actual rows/bucket、1024→512切换、caption长度 | [最小桶](../hardware/apple-silicon.md)；不要先加ANE通道 |
| ANE calls=0但label称hybrid | actual receipt、load/admission/self-test decline | 修真实选路标签；[GPU完整回退](../ane/private-ane.md) |
| calls远超预期 | warmup vs runtime、Session累计值、chunks/prefill/steps | 做增量，核对选中phase；不按layers×steps瞎估 |
| GPU/ANE两支alone都快，合跑慢 | concurrent完整traffic、bandwidth、staging queue | [三臂校准](../gpu-ane/calibration.md)，更小share或关预取 |
| prefetch全命中却更慢 | current GPU前沿、W/A stage竞争、额外bank | [预取消融](../gpu-ane/scheduling.md)，hits不是speedup |
| join主机时间变短，E2E不变 | deferred实际consumer、嵌套spans | [owned deferred](../gpu-ane/shared-memory.md)，工作可能只是后移 |
| 单FFN快，整模型不快 | attention/encoder/VAE占比、完整block计时 | [Amdahl和依赖](optimization-workflow.md)，别跨构建借分母 |
| base快、六步LoRA慢 | correction readiness/down hidden、首步/后续rows | [LoRA](../gpu-ane/lora.md)、phase路由；不借base倍率 |
| 缓存后最优share变了 | conditioning/prefix hit、actual rows | 重新按phase校准，不把cache×ANE倍率相乘 |
| 两个相同seed跨框架漂移 | initial noise、tokenizer/resize/scheduler/MRoPE | 注入同noise/conditioning，再逐步定位 |
| NaN出现在投影/激活后 | BF16→FP16范围、headroom、scale/zero/tail | [量化](../ane/quantization.md)；不clamp掉错误/放松finite |
| 最终MP4/AAC编码报错 | preencoder RGB/PCM finite、frame/FPS完整性 | [媒体IO](../media/video-and-audio.md)，不先归咎codec |
| tile编译成功但部分输出NaN | 尾K/M/N、cooperative API约束、accumulator | [kernel验收](../metal/kernel-tuning.md)，非有限timing不入榜 |
| hybrid看起来像但latent gate失败 | gate/up、hidden、partial/join、每step误差 | 保留failed gate，单独做感知诊断，不改写pass |
| 切adapter后仍像旧adapter | compiled capture、generation、rank/scale cache | A→B→base；[LoRA缓存合同](../gpu-ane/lora.md) |
| 相同shape/指针却读旧权重 | allocation地址复用、immutable promise错误 | [source generation](../metal/command-graphs.md)，禁pointer-only cache |
| outputBackings请求了仍有copy | actual pointer/stride/shape | bounded stride-correct copy与计数；不强读假定backing |
| 小history view却保留很多RAM | backing owner、temporal concat、lazy graph | [VAE history](../media/vae.md)，用独立compact owner |
| MLX active下降但footprint上涨 | IOSurface/CoreML/retained bank/filecache | [分口径记账](../memory/residency.md)，禁止两者相加 |
| DiT内存降了全请求peak不降 | 重新定位phase高水位 | VAE/history/output head可能成新瓶颈 |
| streaming比resident慢 | bytes/read wait/slots/pinned subset | [固定subset+typed refill](../memory/streaming.md)，这是可接受内存tradeoff |
| 加slots反而更慢/挤爆内存 | pool/class总槽数、decode scratch、UMA | 有界少量lookahead，不能只测SSD带宽 |
| 每step权重cache都miss | 循环scan+LRU、C<N | fixed resident prefix/subset，不按recency反复逐出 |
| free看似很多却swap | filecache自生成headroom、reserve漏/重计 | vpipe式residency反馈；不调系统swap掩盖问题 |
| F_NOCACHE仍增长filecache | 实际fileoffset/page对齐、读窗口 | [存储对齐](../memory/storage.md)，不声称cold IO已证明 |
| 取消IO后还成功采用参数 | future.wait未传播reader exception | [显式error latch](../memory/streaming.md)，cast/GPU前拒绝partial |
| timeout后重试偶发崩溃 | 物理completion、latecallback、owner提前释放 | quarantine/drain；signal不是重用许可 |
| VAE tiles有接缝/视频闪烁 | global stats/attention、halo、temporal phase | exact stage tiling与approx crops分开验收 |
| kernel GPU时间变化不大、wall大变 | queue wait/compile/host/frequency/其他负载 | 分层trace；不擅自重启/关用户服务 |
| 测试全绿但实际App没变 | installed binary/library identity、actual请求 | [产品清单](../templates/new-model-checklist.md)，build不是部署 |

## 当还没定位时

减少实验变量，而不是减少原任务：保留相同模型/geometry/steps，只在一个实际边界捕获输入。做GPU原路径、同边界无ANE拆分、实际hybrid三臂，检查额外materialize/owner/finite。若环境负载不合格，保留incomplete记录并换授权窗口，不选有利样本“修正”统计。
