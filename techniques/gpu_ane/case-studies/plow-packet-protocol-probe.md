# plow 原始 Apple shader：有界 packet、跨组可见性与逻辑 fault

<!-- knowledge
id: technique-gpu-ane-case-studies-plow-packet-protocol-probe
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [第三轮架构读本](plow-packets-and-offload.md) · 实际数据 · 编译预检 · [源码范围](../sources.md#plow-packet-runtime)

2026-10-16，M4 Max、macOS26.6.2/25G83、SDK26.5。与此前CPU合同不同，本轮真正调用固定 [`infervisor/plow@35dfa22ed319ece06255e10e4d76996f7a48977b`](https://github.com/infervisor/plow/tree/35dfa22ed319ece06255e10e4d76996f7a48977b) 的**未修改interp.metal**，通过本库小客户端构造有界packet。没有构建/运行完整plow引擎、模型、ANE lane或serving。

## 1. 先复用已有工具，不直接运行无界visibility模式

已阅读上游AGENTS及500行agent-tools指引，检查现有Apple probes。其`probe.m`有大轮数counter/residency模式，visibility shader还含无上限flag等待；本轮不直接运行这些模式，不做最大co-resident grid搜索。

先使用原始`runtime/apple/probe/mslcheck.m`创建library/pipeline：实际返回plow_interp、maxThreads1024、static threadgroup memory32768。该工具不提交应用GPU dispatch，且未显式指定language version，不能把“pipeline创建成功”当数值通过。

随后使用原创host客户端和驱动/整理器，显式MSL3.2、safe math；保持原shader/header字节不变，并复制LICENSE/NOTICE到ignored运行快照。代码在TurboCider内，不修改参考源码或新增PLOW knob/recipe；未执行上游远程GPU租赁、Nix/ROCm/CUDA serving campaign，也不申请其最终性能胜出资格。

## 2. ABI需要核对wire格式，不是Rust builder对象

| 结构 | 实际大小/位置 | 本次用法 |
| --- | --- | --- |
| PlowDevInst / DevInst64 | 64 bytes，t[8]在offset16，tensor handle为u16 | host直接包含原C header，并通过其static assertions |
| PlowStreamEnt | 24 bytes | 每virtual-CU stream保存inst/slice及wait/successor范围 |
| PlowWait | 8 bytes | counter ID与threshold |
| Apple Params | 20 bytes | segment、logical group数、spin_max和instruction窗口 |

Rust的builder-side DevInst仍有宽handle与依赖列表，不能直接当64-byte wire record上传。本轮核对了Rust定义，但没有运行Rust ABI tests；C header和Metal shader也不是同一个语言的自动共享类型，仍需逐字段一致。

payload地址通过GPU pointer table间接访问。客户端另外对六个buffer声明resource usage，并保留owner直到command完成；只绑定pointer-table本身不应替代资源驻留/寿命安排。

## 3. 不是只测counter：先预读、再由另一组写、最后消费

两个logical threadgroup，每组1024线程，原kernel分配32KiB threadgroup scratch。positive spin_max固定1,000,000，不能由CLI放大。

四条原始instruction只使用FP32 ReLU/ScaledAdd：

```text
W = ReLU(poisoned temporary)        # 强制在写入前预读temporary，预期W=0
T = X + 0.5 * Y
R = ReLU(T)
Z = R - 0.25 * X

logical TG0: pre-read slice1 → produce slice0 → consume slice1 → final slice0
logical TG1: pre-read slice0 → produce slice1 → consume slice0 → final slice1
```

每条instruction的两个slice各发布一次counter；下一阶段等待threshold2。预读与消费是同一组访问的半区，producer来自另一组；最后阶段又读取另一组产出的R。N为1、257、65539，包含空slice和非整齐尾部：**空slice也必须完成协议贡献，不能只按非空计算量降低threshold**。

对照调用原始plow_single，同一command buffer中四次逐instruction dispatch、每次两个slice；GPU kernel边界提供诊断隔离。它不执行persistent内的counter协议，不是另一个优化引擎。

## 4. fence观察与进展边界

原kernel在gate后做device-scope atomic_thread_fence，op完成后有threadgroup device barrier及device-scope fence，再由lid0用relaxed atomic增加successor。本文记录这个组合及实测结果，不把relaxed counter独自当payload发布保证。

本轮没有删fence、构造数据竞态或测试弱内存语义的所有排列。两个logical group通过，只表明这个小程序在本机这些运行中前进并读到正确值；不证明general forward progress、最大驻留容量或每个threadgroup固定占某个物理GPU core。

这里的协议是GPU内跨threadgroup，不是CPU↔GPU或GPU↔ANE的live flag许可；异构row/channel路径仍有host segment、设备完成和外部join，不能把本次device atomic观察升级为跨ANE交接证明。

公共header里的grid==CU、CDNA/XCD等说明包含AMD执行假设，不能原封不动成为Metal调度合同。CPU抽象actor模型只说明：若一个actor永远不被调度，threshold2会等不到peer；它不计算Apple occupancy、preemption或weak-memory行为。

本次FP32 op不使用GEMM/attention的tile scratch，不能由32KiB分配成功推断所有scratch别名/全部opcode都正确。

## 5. 实际完整数值与counter检查

两独立进程seed1/2；每N、每arm两次warmup+六次正式，先后次序逐轮反转。合计96个计时记录，72个正式记录，另外10个协议对照。

这里seed只是确定性输入phase的起点，不是两套独立随机模型样本；每轮改变phase用来排除旧代次输出，不能借“两个seed”扩大统计或模型质量范围。

positive完整检查最终Z **2,105,504个元素**，六个payload数组共12,633,024个元素，以及18,432个前后guard元素，全部finite、exact、mismatch0；persistent四个counter均为2，对照均保持0。输入为小型精确dyadic值，不构成一般FP32 fidelity、模型质量或跨dtype资格。

预读数组W也被完整检查，不能仅靠最后checksum看起来正确就省略前面的消费者。Python另独立重算每个phase的最终checksum，防止整理时混入旧输入/输出代次。

## 6. Completed、fault clear、整packet数学完成是三层

每个进程另运行五个单threadgroup、spin_max16的有界控制。没有非法地址、删除同步或硬件device-error注入。

| 控制 | Metal command | fault / counter结果 | 应如何解释 |
| --- | --- | --- | --- |
| 未知opcode65534 | Completed，无Metal error | fault0x40000000，未发布successor | 应拒绝数学结果；定义好的解释器错误路径 |
| 不可能满足的wait | Completed，无Metal error | fault0x80000000，未发布successor | 有界逻辑超时，不是GPU硬件timeout |
| segment不匹配 | Completed | fault0、counter0、payload未动 | 选择范围内空执行，不代表原数学图已完成 |
| 空instruction窗口 | Completed | fault0、counter0、payload未动 | host必须知道实际选择/覆盖范围 |
| NOP带successors[0,0,1] | Completed | counters[2,1,0] | 重复ID必须按multiplicity累加，不能set去重 |

10个控制的payload/guard都维持预期。错误控制只用一个threadgroup，避免多个fault writer争写同一word；不证明多组同时失败时有确定的first-error记录。

原Rust host也有独立check_fault，不能只看command状态。此处是实际原shader错误协议验证，**不是复现生产模型故障、Rust异常回收或row/ANE早退drain**。所有“控制正确”与“可发布tensor”字段分开记录。

## 7. 计时负结果与准备成本

下表是每组六个正式host计时的中位，单位µs；两支同一个command buffer completion边界，初始化/CPU检查/编译在计时外。shared host写入后所需的coherency/driver工作仍可能在提交边界暴露；不是GPU-ready-only或完整请求计时。

| seed / N | persistent | 四次single诊断dispatch | 逐对single/persistent中位 |
| --- | ---: | ---: | ---: |
| 1 / 1 | 191.688 | 185.562 | 0.950× |
| 1 / 257 | 184.812 | 185.500 | 1.004× |
| 1 / 65539 | 813.917 | 794.250 | 1.019× |
| 2 / 1 | 183.416 | 180.396 | 0.991× |
| 2 / 257 | 173.166 | 170.041 | 0.989× |
| 2 / 65539 | 778.750 | 815.896 | 1.054× |

逐对比值不是两组中位数之比。小N的GPU区间中位persistent约28–31µs、single约24–27µs；大N的host波动明显，不能用最大正向样本宣布胜出。**减少dispatch次数不自动比counter/fence协议更便宜。**

这还是有意加入pre-read的两组小packet，未充分利用全GPU；更强基线可以把这些逐元素运算直接融合。不能把它当plow-vs-MLX/vLLM或持久化方案的全局性能排名。

显式3.2的shader+两pipeline准备耗时，两个进程约28.84秒与6.07毫秒；新进程不等于cold-cache。本轮没有清系统cache，巨大准备差异不能藏进一个warm速度数字，也不能解释成所有部署每次都需29秒。

## 8. 安全准入与重放

编译预检和两个执行进程各先取得五次占用0%样本；during仍可能漏掉短GPU burst，不证明绝对无干扰/锁频。没有终止其他应用、改系统wired limit、功耗策略或安装上游依赖。

参考源码和编译物只在ignored outputs，Wiki仅保存原创client/CPU合同、来源与整理数据。以后若更换shader、ABI、compiler/math mode、SoC或geometry，都要重新验证；本轮成功不授予大grid、其他op、ANE、M5或完整请求资格。
