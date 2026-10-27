# Stage支持不等于dispatch支持：实际QKV primitive与完整模型的证据边界

<!-- knowledge
id: technique-metal-case-studies-mlx-stage-sampling-capability-and-primitive-boundaries
type: case-study
stage: prefill, decode, pipeline
operator: attention, gemm, norm, rope, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: approximate, contract-only
representation: bf16, fp32
boundary: cpu-gpu, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [并发encoder机制与校准](mlx-concurrent-encoder-stage-counters.md) · [既有GQA/score研究](mlx-sdpa-gqa-fallback-and-score-precision.md) · [来源](../sources.md#mlx-stage-counter-sources-20261017) · 完整输出指标 · 计数器原值 · 全部事件 · [native控制与审计（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十五轮。目标是核查在现有未修改MLX wheel上，公共counter能观察到哪层边界。结果不是“每个kernel都有时间”：当前M4 Max/25G83/SDK26.5仅支持encoder-stage，明确dispatch请求拒绝；两个受控attention primitive各落在单direct编码encoder，但完整Model主要仍是多dispatch聚合。

## 1. 先拒绝不支持点，再选择独立stage实验

实际capability probe与helper均见仅`timestamp/GPUTimestamp`、stage=true、dispatch=false。独立`request-dispatch`调用只加载原创helper并枚举公开能力，报告`unsupported-no-fallback`、exit3、`MLX_imported=false`、`submitted_GPU_work=false`。v1和v2的两份失败receipt均保留；这不是成功0耗时，也没有为满足请求暗中切到stage。

随后stage实验是**另行明确选择的工作负载**。遵循[Apple API/SDK来源](../../sources/official/metal-counters.md)，不能从某个M4 Max的记录归纳所有M4/OS/驱动都如此；存在stage API更不推出statistics/带宽/ALU利用率集合。复查时必须重查能力，不能只按设备名决定。

## 2. 用真实操作数控制单编码域

输入不是随机小矩阵，也不是本轮Model32-token的中间张量。复用前轮原BF16 Qwen、7-token前缀、第0层已捕获的完整Q/K/V，SHA绑定原report和三份原文件：Q`[1,14,7,64]`，K/V`[1,2,7,64]`。FP32臂仅把**这些BF16实际值**提升表示，不冒称FP32完整模型的QKV。

| primitive | 物理输入/输出需求 | mask与实际入口 |
| --- | --- | --- |
| vector，scope401 | 最后一个Q、K/V长7；完整`[1,14,1,64]`共896值 | 无mask、`force_fused=True`；实见`sdpa_vector_{bfloat16_t,float}_64_64_nomask...` |
| full，scope411 | Q/K/V左补2为物理长9，只请求最后7行`[1,14,7,64]`共6272值 | 预eval的typed additive causal+padding mask；实见`steel_attention_{bfloat16,float32}...has_mask_t_do_causal_n...` |

full的mask条件为`key_index>=2 && query_index>=key_index`，允许区写0，禁止区写负无穷；预先eval输入/mask，与primitive输出eval、readback/cast/file分scope。前两行padding query全部被mask，**未保存/未验收这两行**，它们可能非有限；通过的完整数组指全部requested valid区域，不能写成物理9行输出全通过。

GQA=14/2=7，scale=.125。vector与full的query数、物理K、mask和输出工作量不同，不构成同工作负载算法A/B。`force_fused`在这些特定shape实际成功，不代表任意Q/GQA/dtype支持；既有7/8-token Model fallback结论不被该物理Q9的控制推翻。

## 3. 条件数学通过不修复完整模型

对同一真实操作数，以FP64计算GQA展开的`softmax(QKᵀ*.125+mask)V`；full参考在7行有效逻辑域独立算causal数学，再按输出dtype单次舍入。它是**same-actual-input operator参考**，不是完整模型原数学或低dtype分步执行的逐位承诺。

四模式×两dtype×两重复，32个完整requested primitive数组全部通过原数值预算；相同dtype/operation跨模式/重复逐位一致。与它并列的64份完整Model logits对新算的4个24层参考，FP32 32通过、BF16 32拒绝。局部算子通过、观察保值、top1一致、完整模型预算和任务质量是不同命题。

BF16 full例的6272值对单舍入条件参考max_abs约6.10e-5、relative-L2约5.62e-5；这说明当前控制域的误差小，不证明原BF16模型累计误差已消失，也不定位哪条机器指令/归并导致差异。

## 4. 单dispatch encoder能把时间归到哪

四个stage进程的8个primitive scope各有一个encoder、一次direct编码、一个已知函数入口，stage原值通过同进程映射和父CB包含检查。以下仅为**被观察的encoder边界跨度**，不是benchmark中位数：

| dtype | 重复 | vector stage µs | full stage µs |
| --- | --- | --- | --- |
| BF16 | 17 | 5.416 | 16.083 |
| BF16 | 29 | 8.500 | 21.791 |
| FP32 | 17 | 5.750 | 16.125 |
| FP32 | 29 | 5.583 | 16.583 |

每格只是一个诊断样本，cold首用/缓存、日志/counter、descriptor、driver时序均可能参与；两个重复不是充分统计。vector/full工作量不同，BF16/FP32不构成任务质量等价的速度资格，不能由表给dtype/引擎排名。

即使单direct编码，也没有读取通用dispatch边界：encoder可能有本观察器未覆盖的命令，stage span不是纯shader busy、独立device occupancy或ISA周期。所有`per_dispatch_GPU_seconds=null`保持原样。完整模型的51/52-call encoder更不能均分、按函数计数加权或整段归给attention；[机制页](mlx-concurrent-encoder-stage-counters.md)保存聚合直方图与校准策略。

## 5. 观察器覆盖与扰动控制

stock不显式加载helper；off仅发现具体类/公开能力，不安装hook；meta安装原工厂/PSO/direct/CB日志；stage再换descriptor并附counter。24组Model/primitive有序函数/API/grid/group签名相同，全部96个payload保值；但没有descriptor-only控制，也不授予不可见屏障/生命周期/性能透明性。

hook只覆盖owned child发现的device/queue/CB/compute encoder具体类上9个选定公开selector；对该类所有实例生效，不只一个对象。没有async工厂、indirect/ICB、Metal4或任意运行类的通用覆盖。恢复原IMP经检查成功；为继承方法新增的slot可能留至进程退出。queue/PSO强持有、ARC、锁和IO会扰动资源存活，不能据此做原模型峰值内存资格。

函数名来自PSO创建时`MTLFunction.name`而非metallib清单或空label；不同库可能同名，不是完整specialization/编译闭包/IR层所有权。源树commit、安装core/libmlx文件hash与native binary身份分别记录，不能写成整个wheel可复现构建或已读GPU机器码。

## 6. 可重放资产与修订保留

labs/CPU合同和完整测试不启动设备。新原生client/helper ABI要显式原始固定资产、编译好的helper、新ignored目录、owned supervisor与GPU串行准入；原formal client/plan不就地重写。需显式原件，只核对hash/terminal与重放，非新的GPU或完整数学重跑。

预正式v1 stdout曾把Model+primitive总数命名为`complete_vocab_outputs`。完整已执行v1 client/plan/smoke保留；正式前改为两个独立计数，使用fresh v2 plan/root，复制字节相同的既有binary。原件审计证明源码差异**恰好只有这一处stdout表达式**，native代码/命令/数据shape未改；本轮不是静默用新代码覆盖旧实验。自检/smoke不计正式64/32输出或1370对counter。

复查条件：采样能力或counter ABI变化、SDK/OS/SoC/runtime/source/client变化、真实QKV来源/shape/dtype/mask/有效区改变、输出需求/consumer/scope改变，以及需要物理kernel/能耗/长期请求证据时。广泛研究继续active；当前不支持dispatch点是本配置结果，不是整体阻塞，也不把单primitive通过当完整模型采用依据。
