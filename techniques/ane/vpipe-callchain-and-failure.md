# vpipe源码纵切：runtime权重、持久worker、headroom与失败传播

<!-- knowledge
id: technique-ane-vpipe-callchain-and-failure
type: method
operator: gemm, layout, scheduling
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: precision-loss, lifetime, measurement
semantics: contract-only
representation: fp16, fp32, iosurface
boundary: producer-consumer
api: reference-only
-->

[ANE目录](README.md) · [既有图内tiling导读](../sources/repos/vpipe-runtime-tiling.md) · [本轮来源](sources.md#implementation-sources-20261016)

固定 `tgo-app-dev/vpipe@c2989a76841888e4d8d208fad4f476c487710962`，选定源码只读。与此前只看emitter不同，本轮追到了模型调用、weight slot、worker、Core ML wrapper和实际失败返回。没有运行完整vpipe模型或其Core ML FFN。

后续独立完成了[原emitter图+受控headroom输入实验](headroom-bias-and-bridge.md)：实际执行生成的Core ML小图，数值观察支持下述条件bias反例；仍未执行原AneFeedForward自动retry或完整模型，不回填成本篇历史轮次已验证。

## 1. 调用链与真正的边界

```text
模型stage(layer sources) → AneWorker.dispatch：decode/LoRA/填weight slots
  → begin：选择整row chunks，转换输入，调用AneModule.run
  → 同时模型线程运行GPU剩余rows
  → GPU完成 → finish/join：检查成功、转换/拼接结果、更新策略
  → 首个完整输出consumer
```

源码入口：[ane-ffn.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc)、[ane-module.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-module.cc)、[ane-worker.cc](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-worker.cc)、[Core ML manager](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/coreml-model-manager.cc)。

这里按row分担，GPU prefix和ANE tail对FFN数学独立；不是把一个线性K归约分给两设备。所有row仍必须完成，不能因为GPU完成就发布整个输出。begin捕获的是原buffer pointer，caller必须保证producer完成、输入/目的buffer保活到finish。

## 2. 为什么有两个pool而不是每层新建thread

AneWorker有一个dispatch线程和独立的最多4线程convert pool，避免dispatch job调用parallel_for时等自己。single outstanding job提供backpressure；join只等待本worker任务，不证明其他设备/系统全局drain。低于65536个元素inline是该实现的政策，不是CPU/ANE硬件阈值。

`parallel_for`以generation唤醒worker并等待left归零；调用者的body在等待返回前仍有效，不允许reentrant。FeedForward析构先join再还wired/scratch资源，是这里最重要的生命周期顺序。不能由“一个worker”推出所有模型或所有Core ML调用在硬件上必定串行。

原创CPU client实际编译原worker：65535/65536/131077个元素，各4代，全部元素验证0错误；129×513的row-granular分片每row恰好一次。只链接C++线程代码，不链接Core ML/Metal、不执行ANE；它证明此有界dispatch/convert/归属协议，不是ANE并行性能。

## 3. runtime weight slots与LoRA staging

AneModule runtime图只按shape编译，权重以CVPixelBuffer slot作为输入。每次fill获得真实bytes-per-row，不能用cols×2忽略padding；run还检查每个slot rows/cols与图的weight feature匹配。

stage从具体source格式解码，保持source stride/offset；LoRA不再每row做rank次axpy，而是按最多1024row band构建B选择，与A做BLAS乘积后加到base再cast FP16。这样可利用矩阵级实现、减少调度，但不是证明BLAS必在AMX或band1024对所有模型最优。adapter的group/parts、B stride和offset都是语义，不得只按rank匹配。

runtime图摊销编译，却每层仍有decode、adapter合成、转换和slot写入；常量图保留快路径又增加checkpoint绑定与加载成本。`ane-module.h`中的M4 Pro速度、8–10MB拐点、免费crossing等注释都只作为作者指定实验观察，**不写为M4 Max或所有ANE的规格**。

stage里的4-bit lookup是CPU为affine codes的16种可能值构建FP16转换表，减少逐元素scale/bias算术；它不是Core ML权重palettization的同一种码本合同，更不证明ANE在执行INT4 MAC。需要适配器或headroom scale时仍转入FP32合成路径，不能沿用未缩放的lookup值。

## 4. row chunk与内部K/N tile是两层粒度

module固定M为chunk，host按所分配row数多次调用；因此只保留一个chunk大小的input/output scratch，避免视频长序列时全tail常驻。内部K/N tile仍在同一图归约，是另一层。host chunk解决资源峰值和bucket复用，却增加prediction固定税。

auto share初始约半，按整chunk分配且GPU至少留一row；随后依据GPU每row与ANE每chunk的EMA预测，只有明显改善才移动，减少一块/两块间振荡。经验窗口、5%hysteresis和warmup排除都是软件启发式，不是获胜保证。数值retry即便排除在稳态控制器统计之外，也仍应计入用户真实请求时间。

## 5. headroom补偿必须穿过最终bias

源码对SwiGLU的up权重/其bias做缩小，输出再乘回scale。实数关系为：

`H = SiLU(G) * U`，`Y = H*Wd + bd`。

若 `U→U/s` 而保持bd不变，再恢复整个输出，则得到 `Y' = H*Wd + s*bd`。**bd非零时多出(s−1)bd**。已读emitter确实在down之后加bd，stage/retry对up支路缩放，未见对应down-bias缩放。

独立CPU witness用s=4、bd=0.5验证偏差1.5；将bd也除以s后恢复误差小于1e−12。它证明这一条件下的数学风险，**没有运行上游带bias的真实FFN，不称本机Core ML bug复现或所有模型都受影响**。bd=0或另有补偿时需另判；GELU路线缩的是线性down支路，不能与SwiGLU机制混为一谈。

finite输出不保证语义正确，这也是为什么溢出重试必须携带完整bias/LoRA/restore reference。不能为保finite直接clip原BF16残差，或把转换误差当输入源精度相同。

## 6. output backing：请求与实际采用之间仍有验证点

AneModule借用shared buffer创建输入、给输出提供backing；manager在设置backing时标记bound，并在prediction后跳过对应copy-out。未绑定时则查看实际返回dtype/strides，copy或逐元素gather。

已读bound路径没有对实际返回pointer/stride重新核对。如果某OS/图未按预期采用backing，是否读到旧scratch应做poison和返回身份对照；**本轮未执行此wrapper，不能宣布它在当前macOS错误**。单次数据格式自测也不替代每次真实输出的generation/ownership。

## 7. 失败必须追到模型consumer

FeedForward只在全部chunk成功且finite后设_ok；晚chunk失败时，前面chunk可能已写入tail。finish返回false并告警“rows未完成”，本类不自动整FFN GPU重算。

所读Qwen-Image21调用点会检查false并终止forward；Krea2的指定调用点将finish返回值转void后继续gated consumer。后者是需要进一步故障注入的集成风险：若该次失败确实可到达，就可能消费partial/stale rows；本轮没有运行该模型或断言已触发。不能只看wrapper里有bool就声称全应用有可靠fallback。

迁移到TurboCider应沿用[整operation GPU重算/不发布半输出](private-ane.md)合同，或明确整请求失败；不能把成功prefix和未完成tail拼成结果。失败后的同步、资源隔离和原输入保留必须同时设计。

## 本地学习与继续复核

状态/索引/headroom CPU合同与原worker显式集成可重放；`python3 -B techniques/labs/run.py ane-implementation-contract`不触发设备。source、graph/bias/LoRA、shape/chunk/tile、SoC/OS、backing或模型consumer改变后重验。

尚未验证：真实vpipe带bias overflow重试、晚chunk失败注入、backing未采用控制、完整GPU/ANE并发请求和低内存/长时。保留这些具体重开入口，不用CPU线程通过替完整引擎取得资格。
