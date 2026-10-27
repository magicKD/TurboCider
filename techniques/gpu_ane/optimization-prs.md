# 优化 PR 读本：收益、正确性与兼容性分开

<!-- knowledge
id: technique-gpu-ane-optimization-prs
type: method
-->

[专题导航](README.md) · [GEMM](gpu-gemm-quantization.md) · [Attention](gpu-attention-runtime.md) · [来源清单](sources.md#survey)

状态核查时间：2026-10-16。PR 状态会变化；以下是当日网页观察，不是持续更新服务。patch 下载只用于阅读，没有执行或合并到产品。完整 snapshot SHA-256 记录在来源清单；短 commit 只是阅读定位，不能替代 full source/binary 身份。

## 六项值得精读的改动

| PR / 当日状态 | 机制与已读范围 | 迁移前的验证 |
| --- | --- | --- |
| [MLX #4516](https://github.com/ml-explore/mlx/pull/4516)，已于 2026-10-04 合并，merge `1eb18d7` | PR 描述+patch 开头：先拓宽再求输入和，减少 affine bias contribution 的 half 舍入；分批 QMM decode 复用等改动 | 独立 decoder/FP64 oracle、bias 贡献、不同 bit/group/M、split/tail。它既是性能改进也是数值修正 |
| [MLX #4682](https://github.com/ml-explore/mlx/pull/4682)，Open | 两个 patch 的说明和关键 hunks：mid-M NAX split-K、FP32 partial、16-row simdgroups、packed words/activation prefetch、双 buffer | 特定 N/K/group 对齐、输入 offset、batch fallback、FP32/TF32、寄存器压力与 reduction 成本 |
| [MLX #4596](https://github.com/ml-explore/mlx/pull/4596)，Open | 描述+patch 1–60：generic vector SDPA 以四个 key unroll/prefetch，提高 memory-level parallelism，保留 scalar tail | 实际 generic/GQA dispatch、D/context/tail、online-softmax 数值；不是所有 attention 核都变更 |
| [MLX #4594](https://github.com/ml-explore/mlx/pull/4594)，Open | 描述+patch 1–60：gated-delta NAX cooperative 类型通过 operation helper 推导 | 新 SDK 编译/旧 SDK 回退与数值；这是兼容性修复，没有据此声称加速倍率 |
| [llama.cpp #20962](https://github.com/ggml-org/llama.cpp/pull/20962)，已于 2026-03-25 合并，merge `adbadfaf` | 描述+patch 1–60：分离 tensor matmul2d 与 legacy SIMD-group 核，独立 layout/threadgroup/dispatch | Metal family/SDK、quant decode、非整除输出；保留 legacy 路线。不能只替换 MMA 而保留旧核全部数据流 |
| [llama.cpp #27461](https://github.com/ggml-org/llama.cpp/pull/27461)，已于 2026-09-02 合并，merge `30e53b91` | PR 讨论+patch 开头：Tensor API 需要正确 Metal language、库编译和 shader 路径；讨论后采用分开 tensor kernel library 的解决方向 | embedded/source/external library 各配置、MPP header/compiler 可用性。硬件支持不等于 build artifact 包含功能 |

上述上游 benchmark 均为 **source-reported**。本机安装 MLX 0.32.0，不把它冒充这些 PR 的 parent/head A/B；M4 Max 也不能验证 M5 NAX kernel 的性能。主线源码含某项修复不证明某个发布 wheel 是该 commit 构建。

## #4682 的两个不同实验变量

第一 patch 增加 split-K 路由，让 mid-M、输出 tile 太少的场景有更多 threadgroups；改变了归约顺序和 scratch。第二 patch 改 tile/加载 schedule，在当前矩阵计算前先取下一 K step 的 packed 数据，再解码；它尝试保持**同一个 split 配置下**的累加顺序。

不能把作者“同 split bit-identical”的说明扩大为与原 unsplit 路线逐位相同，也不能把最初 patch 的 split 上限当成最终 patch 的参数。下载的系列包含 `6b2af68…`、`6e7ab39…`；第二说明已把 split 限制改为最多 4、每段至少 256 列。移植时读最终组合后的文件，而非只拼第一版文字。

## 一个优化 PR 的合格学习记录

第二十六轮另读[ggml-ane固定audit commit](case-studies/ggml-ane-backend-cache-and-failure.md)的两文件patch：double-check duplicate free、FP16 denormal及shape下限修正有明确代码身份，但不是一份已核状态的PR，也不等于完整backend合同已解决。对应CPU负控/ABI拒绝另列，不把commit标题或作者TFLOPS升级成本机正确性和速度资格。

必须保存 parent/head、当日状态、changed paths、patch hash、原问题和适用条件；然后把“实现了什么”“作者测了什么”“我们测了什么”分开。合并只是代码治理状态，不是本机加速证据。

复现实验固定 GPU、OS/SDK、模型/quant source，构建 parent/head 两份，先过相同 oracle，再交替计时。报告所有合法 shape 与失败，不只选表里最大的倍率。吞吐提升同时记录额外 resident/scratch；新 GPU 优化上线后重新校准 ANE share。

本专题没有导入上游 shader/patch；代码许可与 third-party notices 如将来移植需逐文件核对。当前只有原创机制解释、链接和内容身份。

## 第三轮新增：Uzu #973

[当前实现与两段patch对照](case-studies/uzu-metal4-and-routing.md)记录了#973从临时短KV旁路到撤销KV门槛的演进；[第三轮清单](sources.md#compiler-packets)保存patch hash与commit身份。PR网页状态未成功取得，保持unknown；不把commit标题中的“2–3x”当本机或所有shape收益，也不将浅克隆整仓库stat当PR改动范围。

## 第六轮新增：图委托与共享资产

下表状态来自2026-10-16的GitHub API快照；metadata、patch及文件历史分别绑定，见[第六轮来源](sources.md#coreml-delegates)。本轮未运行三个PR的上游测试或benchmark。

| PR / API观察 | 值得学习的机制 | 不能继承的资格 |
| --- | --- | --- |
| [ORT #28293](https://github.com/microsoft/onnxruntime/pull/28293)，2026-05-20合并 | 新小op避免分区断点，同时丢弃纯trivial分区；4段patch中的后续改动把marker移至builder | M3 Max作者结果不等于本机/任意shape收益；不是所有Tile都便宜 |
| [ExecuTorch #16552](https://github.com/pytorch/executorch/pull/16552)，closed、merged=false | 多方法multifunction资产、CMJR/NamedDataStore引用、function与method分离；当前源码另有完整实现 | 提交标题含PR号、非空merge_commit_sha或feature已存在，不足以声称GitHub PR已合并；磁盘共享不证明resident/state共享 |
| [ExecuTorch #21969](https://github.com/pytorch/executorch/pull/21969)，2026-08-28合并 | tied embedding的gather与linear共用const时统一压缩策略；3段patch最终形成私有free helper | regression只测lowering/program生成，不证明推理质量、ANE整数MAC或性能 |

对应完整调用链见[ORT](case-studies/onnxruntime-coreml-partition-cache.md)与[ExecuTorch](case-studies/executorch-coreml-delegation.md)。这些优化改变图边界、表示和资产生命周期，不能只抽一条kernel或一句“支持CoreML”学习。

## 第九轮新增：LiteRT #9762生命周期修复

[LiteRT #9762](https://github.com/google-ai-edge/LiteRT/pull/9762)在2026-10-16的GitHub API观察为closed、merged=true，2026-09-04合并，merge/patch `cad523c157861c4256f845d7f47e4bff308138ad`。读取四文件、152行patch，身份见[第九轮清单](sources.md#litert-coreml)。它给模型/路径property明确owner，提前记录temp路径，检查cleanup目标并在kernel析构捕获Foundation异常。

学习价值是失败编译、autorelease与C++析构边界的可靠性，不是速度倍率；本机没有崩溃复现、性能A/B或上游测试运行。[调用链与不能继承的资格](case-studies/litert-coreml-handoff-and-lifecycle.md)单列。旧delegate的修复也不能证明V2 Compiled Model API已经接通ANE。

## 第十三轮新增：GGML few-row与residual身份修复

[五份官方PR状态与固定源码](sources.md#ggml-fewrow)区分#25377（closed、merged=false，即使merge_commit_sha非空）、#29110（open）与#29869/#30100/#30065（分别10月5日、7日、7日合并）。三份长patch只归档，#30065/#30100共234行完整读；不把提案、PR正文和最终merge代码混用。

[few-row机制](case-studies/ggml-fewrow-mma-and-reuse.md)追到解码复用/shape gates，另实际执行[#30100原始回归](case-studies/ggml-residual-identity-regression.md)：修复和关闭融合各28/28，旧encoder负对照27/28失败。它是correctness，不复现作者性能或真实Clef模型；没有合并patch到产品，也不对上游提交PR/评论。

## 第十七轮续读：MLX #4682最终组合源码

2026-10-17的[API/固定文件证据](sources.md#mlx-nax-dispatch)确认#4682仍open、merged=false，精确base`06eb7483…`和head`6e7ab39…`、两commit ancestry与tree分别绑定。新下载patch与第一轮字节相同，不重复计作独立PR；本轮进一步读[最终host/kernel/测试实现](case-studies/mlx-nax-mid-m-and-prefetch.md)，不是把两段hunk机械拼接。两个原始pure CPU helper的3430点通过不等于M5 kernel或作者benchmark复现，也不将API的非空merge预览SHA当合并证据。

## 第二十七轮新增：tinygrad #18762的ICB统一分配

[固定来源](sources.md#tinygrad-metal-replay)核#18762在2026-10-12T11:30:49Z合并，merge`b41c51306bb1f4228e5ecb5ba78b838363442724`，base/head及两文件返回patch全文另存。它把LinkCtx传入bufferize，使ICB绑定统一分配的buffer/offset，移除专用nolru分配；应结合ring、JIT保留链接与owner阅读。[当前原runtime/受控执行](case-studies/tinygrad-metal-hcq-and-replay.md)不是PR parent/head性能A/B，不能把8kernel eager/JIT倍率归因给该PR；profiling、Python scalar与bare1D extent边界也不能由“支持ICB”一词替代验证。

## 第二十九轮新增：MLX #3635的首次trace失败事务

[固定API/源码](sources.md#mlx-compiled-ffn)核#3635在2026-06-09T23:56:04Z合并，merge`01368d8e7888d6989969aa82bc36f2ba09dc5ced`；两文件patch全文读。`entry.empty=false`移到trace/simplify/fuse全部完成后；原回归是nullary首次throw，不是随意一个非空输入重试。[安全Python analogue](case-studies/mlx-compile-and-cross-runtime-ffn.md)两次失败均重trace、第三次3，但不是原C++套件/parent-head执行、GPU失败恢复或性能倍率验证。版本匹配wheel指纹与完整build来源证明仍分开。

## 第三十轮新增：anemll-flash-mlx #1的标题与实际改动

2026-10-17[官方API/固定compare](sources.md)核[#1](https://github.com/Anemll/anemll-flash-mlx/pull/1)仍open、merged=false；base`f86c0682…`、head`3ff58277…`，非空merge预览不是合并。标题称fix native regression，但7个changed paths仅ignore/CLAUDE/autoresearch脚本/npz/TSV，未改`csrc/`或`flash_moe_mlx/` runtime。两次patch失败/partial保留，不冒充完整patch；固定head结果表Git blob另核。M5 Max作者native/Python速度、长短生成warmup、LFU与N→N+1 predictor负结果见[报告](case-studies/anemll-flash-mlx-banked-streaming.md)，均未本机复现。原始CPU I/O短读边界是另一组组件实测，不是该PR的parent/head A/B或已实施修复。

## 第三十二轮新增：whisper.cpp #566的stage替换接缝

2026-10-17[官方API快照](sources.md)核[#566](https://github.com/ggml-org/whisper.cpp/pull/566)已于2023-04-15T10:21:28Z合并，merge`5e47e223bd74af8a16cd4f2900cfbcfb92cd1f18`，base/head及15路径绑定。encoder bridge、whisper.cpp和generate脚本三份返回hunk全文读，其他不称全读。初始固定80×3000输入和独立encoder资产/调用接缝，不能直接套成今天显式All配置或GPU/ANE并行。当前ref的[真实tiny.en/强Metal基线](case-studies/whisper-coreml-metal-stages-and-real-asr.md)不是PR parent/head或README的CPU倍率复现；旧compiled资产、init/first/warm和原timer分母分别记账，未给物理ANE/overlap或通用模型排名资格。
