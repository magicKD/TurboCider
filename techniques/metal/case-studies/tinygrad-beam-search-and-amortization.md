# tinygrad BEAM 真正选核：调度变化、搜索成本、cache 范围与失败策略

<!-- knowledge
id: technique-metal-case-studies-tinygrad-beam-search-and-amortization
type: case-study
stage: ffn, pipeline
operator: gemm, norm, swiglu, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: compile-bound, launch-bound, measurement, wrong-layout, stale-cache
semantics: numerically-equivalent, contract-only
representation: fp32
boundary: producer-consumer
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU 导航](../README.md) · [前轮 TinyJit 实现](../tinygrad-metal-jit.md) · [来源/PR 身份](../sources.md#beam-sources-20261016) · 全部候选/正式样本 · [本轮核查（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，第八轮。前轮只运行 BEAM=0 的 TinyJit，本轮**真实执行原始 Metal BEAM search**：两个 seed、12次 cold kernel search、12次自己 SQLite 的 cache 命中、1,400次候选编译尝试、1,296次候选设备计时；另有256个正式 default/BEAM TinyJit 回放输出全量对独立 CPU FP64，全部通过。

结论不是“打开 BEAM 即加速”：本矩阵热态 paired median约 **1.05–1.13×**，搜索却约2.5–24秒/工作负载。按本次小型组件的 replay 差估算，常需十万到数十万次相同调用才能抵消 setup；一个点没有正的 median latency 差，故没有有限回本值。cache hit 能跳过搜索，不等于免掉其它编译/提交准备，更不是新设备/输入的性能或正确性资格。

## 1. 执行身份与预算：原始算法、自己的观察器

固定 `tinygrad/tinygrad@3d057f05df532deca7661bfb2400c6564f96a04a`；18个选定文件前后都匹配 commit blob/SHA，reference tracked tree 采集时 clean。环境为同一 M4 Max 主机、Python3.11.16/NumPy2.0.2，tinygrad 报 `Apple9`（Metal family 字符串，不是 MLX architecture 编号）。没有安装 LLVM、改系统/产品默认、加载 PCI kernel module 或下载模型。

运行后工具链复核为 Apple clang21.0.0、SDK26.5、OS build25G83，作为post-run环境快照，不追认为每次私有Metal compiler service的独立构建attestation。

显式 integration加载固定 checkout，**调用原始 BEAM 函数，不改变候选选择代码**。Python wrappers 只记录合法动作数、编译返回、源/二进制 hash、时间与返回 opts；因此 search wall 是带观察器的成本，不称无扰动原生时间。

| 设置 | 本轮值 | 范围 |
| --- | --- | --- |
| host runtime | `CPU:CLANG` | 原本默认 LLVM 在本机不可用；仅任务子进程选择 |
| BEAM | width2 | bounded search，不是全局最优证明 |
| `BEAM_ESTIMATE` | 0 | 候选测完整 grid，不用缩 grid 外推 |
| upcast/local/UOp cap | 128 / 512 / 3000 | 显式预算；与无限搜索或所有上游默认不等同 |
| compile timeout | 5秒 | 原始 `_try_compile` 的预算；不等于所有 GPU fault 可恢复 |
| PARALLEL | 0 | 编译/计时串行，可追踪 observer，不同时并跑正式 GPU workload |
| CACHELEVEL | 2 | DB/XDG 路径只指向新任务目录，不清/读用户旧 cache |
| VIZ/PROFILE/DEBUG | 0 | 正式回放关闭观察型 profile/server；search 仍使用显式 timer |

关键 compiler settings 使用固定读本的 factory defaults（例如 TC=1、TC_OPT=0、ALLOW_TF32=0），实际 context 保存在 capsule。`TC_OPT` 的 helper 默认0，与 BEAM action 构造的 `getenv(...,2)` 是不同位置的规则；没有把两者硬改成同一个值再称“上游默认”。

## 2. 哪些实现值得逐行跟

- [opt/search.py](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/search.py)：动作、资源过滤、binary 去重、少量 min 计时、beam 退出与 SQLite hint。
- [postrange.py](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/postrange.py)：`Scheduler`、TC padding/mask、dummy buffers/变量中点、default/BEAM 分流。
- [codegen](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/__init__.py)与[realize](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/engine/realize.py)：beam 标记、KernelInfo/ProgramInfo、program cache、HCQ lowering、候选 timer。
- [Metal renderer](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/renderer/cstyle.py)、[TC 描述](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/renderer/tc.py)、[Metal runtime](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/runtime/ops_metal.py)：实际 `simdgroup_*8x8`/MMA、threadgroup 限制、ICB/队列/时间读取。
- [helpers](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/helpers.py)：ContextVar 与 SQLite schema24；[worker](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/engine/worker.py)与[viz README](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/viz/README.md)：编译子进程和观测边界，不照搬 AMD SQTT/PMC 到 Metal。

编译/源阅读范围在 manifest，18个执行输入身份不等于整库逐行审计。MIT，Copyright (c) 2024 the tiny corp；没有将上游源码 body vendored 到 tracked Wiki。

BEAM=0 是**hand-coded heuristic 优化基线**，不是未优化 scalar。BEAM>=1 走另一条 schedule 搜索，不保证包含/选中 heuristic 完整方案。每条路径在相同 Tensor 图/输入/权重、相同 TinyJit 与同步边界比较，不拿 eager 的 host 开销冒充 BEAM 收益。

## 3. 选核分数是真设备报告，但不是 correctness certificate

候选 buffer 按 AST 尺寸单独分配，symbolic values 按范围中点取代表；不是用户真实 tensor 数据。搜索不逐个对 CPU oracle 检查输出。这里选固定索引算术图，未用数据依赖 gather 或故障注入，不能将1,296个计时候选称作1,296个已验证正确的 kernel。

本 Metal 路径的候选分数经 `time_call`→HCQ profile slots→完成后 GPUStartTime/GPUEndTime 换算，属于**该后端报告的 command-buffer 时间**，不是 gputrace 按名字赋常数。CPU submitter 分值由该入口排除；仍不等于 shader 指令耗时、DRAM bytes 或硬件 occupancy counter。FLOPs/loads/unique bytes 来自 `Estimates`，另属静态推算。

候选通常取至多3次的最小值，并可 early-stop，不能当正式独立 A/B。默认源码还可缩 global grid 再乘 factor；本轮所有 timed records 的 `allow_test_size=false`，没有缩 workload。CPU proxy 模型演示算术外推及 min 对偶然快值敏感，**不是本机验证了缩 grid 线性**。

编译返回“无 program”共4次，只说明原函数过滤/捕获后无返回，未取得具体原因；不自动标成 timeout/GPU fault。其余重复 binary/计算量过滤也与真正设备计时分开计数。候选源/binary通常名为 `test`，最终 program 会重新命名/编译；hash 可能不同，用 opts/AST 关联，不因 hash 不同臆断算法改变。

## 4. 代码层变化：tail 的 TC padding 真的执行且数值通过

四种 FP32 workload，各有两套输入，权重作为显式参数：regular GEMM128³、tail GEMM M33/N127/K129、norm+tanh128×1024、小 SwiGLU FFN M16/H128/F256。正式每点8个ABBA block，单独 cold ignore/capture/replay，保持 source/binary、最后 schedule opts、global/local dims。

tail 点的 default program 无 MMA 调用；两个 seed 的 BEAM 最终 program 都有 `simdgroup_multiply_accumulate`，opts 包含 TC opt-level2 padding。原 Scheduler 只在该条件允许相应维度 padding；完整33×127输出通过 CPU FP64，不能只跑整除128案例后宣称 tail 合法。

regular 点两臂都有 MMA，但布局不同；seed17 的 heuristic grid/local 为 `[4,1,1]/[32,4,1]`，搜索后为 `[16,16,1]/[32,1,1]`。这是并行布局空间变化，不是已经测得 occupancy 根因。norm/FFN 的 local/upcast/TC 选择也会变，kernel 数没有因此自动减少：本矩阵 GEMM1个、norm2个、FFN2个最终 Metal program。

FFN 仍可能混用含/不含 MMA 的程序，不能把一次 `TC` 动作称为整个模型 tensor 化；tinygrad 内部 tensor-core 名称指本 renderer 的 SIMD-group matrix 路径，**不是 M5 NAX/ANE**。没有独立指令反汇编或 PMU 证明。

256个正式输出最大 relative L2约 `4.37e-7`，门限 `1e-4`；所有 cold stages、同 shape 新 weight、cache 重建也全输出检查。两臂都拒绝 fixed TinyJit 的新 shape。未测 FP16/BF16、极值、更多 stride、symbolic binding、真实 checkpoint/LoRA 或模型质量。

## 5. 热态小收益与秒级搜索：必须把 setup 放回账本

表中时间是两个 run 各自中位数范围；paired倍率按每 block 内 mean 相除，再取8个比值 median，不是两列 median 相除。计时是 TinyJit call→Metal synchronize host wall，source/input准备、CPU FP64/readback在外，验证后及下一次计时前 drain。

| workload | default replay ms | BEAM replay ms | paired default/BEAM | instrumented search wall |
| --- | ---: | ---: | ---: | ---: |
| GEMM128³ | 0.573–0.598 | 0.565–0.575 | 约1.05–1.06× | 6.22–7.42 s |
| GEMM33×127×129 | 0.250–0.252 | 约0.237 | 约1.06–1.07× | 5.31–11.29 s |
| norm128×1024+tanh | 0.642–0.672 | 0.615–0.649 | 约1.05–1.11× | 2.49–3.09 s |
| SwiGLU16/128/256 | 约0.754–0.757 | 0.692–0.714 | 约1.11–1.13× | 23.16–23.60 s |

这些小合成 replay 很大部分可以受 host/submit/同步影响。两 seed 各自独立 search 会选不同布局与分值；未做后台 GPU 静默/热状态控制，不把几个百分点推广为模型收益或置信区间，也不与 MLX/MPS 跨框架排名。

回本模型：`ceil(额外首次成本 / (default热态中位数 - candidate热态中位数))`；差值非正则 null。这里额外首次成本是 BEAM/default 首个 ignore 调用之差，含 instrumented搜索、编译/初始化，不是纯 kernel 时间；baseline先构建，缓存/顺序可能影响该估计。

本次 tail约37万–86万调用、norm约10.7万–11.1万、FFN约35.8万–59.2万。regular点一个 seed无正中位差所以null，另一个约21.6万；paired ratio略大于1与unpaired median差非正可同时出现，**不能拼成一个保证会回本的结论**。该模型不是实跑这么多次后的真实回本验收，也未完整纳入冷模型加载/JIT capture、cache验证/重新搜索与请求长度分布。

## 6. 四层 cache 的职责不能互相替代

| 层 | 保留什么 | 本轮观察 / 资格边界 |
| --- | --- | --- |
| BEAM SQLite | 已选 opts hint | key 显式字段为 AST、width、是否缩 grid、device、renderer suffix；Metal观察到 device=`METAL`、suffix为空 |
| in-process program cache | AST/renderer target/compiler context→PROGRAM | 含 target 与 codegen设置，不等于 BEAM hint 自带性能环境闭包 |
| compiler binary cache | source→编译结果 | binary相同不证明新输入数值/新设备性能通过 |
| TinyJit graph | captured调用/内存/输入view契约 | 相同 shape新 weight仍需传入；本轮 fresh weight checks通过，不授予所有LoRA/动态模型资格 |

重复构建前只清**本进程** program dictionary，让原始 `beam_search` 实际查自己的 SQLite。12次均命中、零新候选编译/计时；整个rebuild仍约15–153 ms，包含其它lower/编译/准备，不是“cache hit启动0成本”。没有删用户 DB，或把别人的 SQLite/pickle 当可信输入。

该 BEAM key 没有显式 SoC/SDK/arch/resource-cap 字段；有些差异可能间接进入AST/suffix，但不能假定全部都会。SQLite schema24是格式隔离，不自动等于所有源码/驱动语义隔离。实际候选与最终 program 身份都保存；应用应额外绑定精确环境、policy/shape/layout/变量、质量输入与消费者边界后重验，而不是把 hint 当编译/正确性/性能资格。

## 7. 两个关闭未合并 PR：想解决的问题仍值得学

2026-10-16 REST身份与完整diff已读取，base/head/diff SHA在manifest；两者**closed、merged=false、merged_at=null**，本地都未打patch：

| PR | 改动与当前代码关系 | 学习边界 |
| --- | --- | --- |
| [#15057](https://github.com/tinygrad/tinygrad/pull/15057) | 给 BEAM key 加具体优化env；维护者认为方案不够通用而关闭；当前读本仍是五个显式字段 | 不应假定小名单覆盖所有影响选核的环境；不能把未合并修复当有效主线 |
| [#18436](https://github.com/tinygrad/tinygrad/pull/18436) | timing阶段异常在非strict时跳过，strict时抛出；新增 monkeypatch测试 | 不能从模拟异常/继续选核推出真实设备恢复安全 |

#18436 REST的 `merge_commit_sha` 非空，是API返回值，**不是合并证明**；manifest单独保存该值，实际 merge_commit保持null，不用临时merge SHA覆盖`merged=false`。

为避免只读diff猜语义，CPU显式 source client执行当前和固定PR head的**原始 beam_search 函数 body**，只把allocator/compiler/device/timer替换为CPU doubles（不 import Tinygrad设备，不注入GPU fault）：

| timing异常 | 当前读本 strict0 / strict1 | closed PR head strict0 / strict1 |
| --- | --- | --- |
| RuntimeError | 跳过 / 跳过 | 跳过 / 抛出 |
| KeyError | 抛出 / 抛出 | 跳过 / 抛出 |
| TimeoutError | 抛出 / 抛出 | 跳过 / 抛出 |

12行均实际验证各source body的一次timing调用。“跳过”在本CPU场景所有候选失败后返回原scheduler，**不是数值/性能胜利，也不等于自动采用 heuristic**。严格模式名字不能代替逐阶段异常分支；本表只验证timing分支，不推广成所有compile路径。

没有真实 fault/reset/recovery 资格。仓库的外部timeout recovery例子面向AMD AM设备，不能拿到Metal默认运行；本轮未跑该fuzzer、未证明Metal错误后的上下文可安全复用。宽泛catch也可能隐藏客户端KeyError等程序错误，需要保留失败类型/候选账本，而非只把search继续执行当成功。

## 8. 复现与后续

CPU controls与公共capsule检查不加载外部引擎/设备：

显式GPU运行需审核固定checkout、已有NumPy/Python/CLT；所有输出新建。CLI只在任务子进程设置factory关键变量与budget，不改global defaults、源checkout、系统保护或包：

两个seed串行，search不与自己其它GPUbenchmark并跑。VIZ=0不会自动起server；原始cache/日志/生成物在ignored任务目录，源码不分发。只读明确final run与CPU policy结果，候选以列式完整保存，不用最优几个替代全部计时。

首次smoke因客户端错误地在ProgramInfo读取`estimates`而失败，正确归属是KernelInfo；只修自己的collector，没有改optimizer。第二个width1小smoke数值通过但没有明显回放收益，两份都保留并排除于最终比较。9项离线tests重算正式配对/回本模型、全部候选最小值、cache/新权重/shape控制、PR与CPU异常策略，不在普通unittest里启动GPU。

[版本复查](../../guides/knowledge-base.md)绑定本轮source/evidence SHA与记录环境，换SoC/SDK/编译器或30天到期会要求重审；与cache hint一样，检索命中不授予性能/兼容资格。

后续应继续到更大真实FFN/Conv/模型、cold/request与质量门禁，独立重测selected kernel/device span与完整consumer；如果研究proxy/parallel/more widths/symbolic输入，也必须另立候选/数值/预算。MLC/TVM实际模型、其它引擎、Metal4/NAX与更强profiling仍有工作；本轮只补实BEAM，不以小收益关闭持续研究目标。
