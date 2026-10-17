# MLX完整Qwen的公开CB窗口：进程内类级观察、queue绑定与覆盖边界

<!-- knowledge
id: technique-metal-case-studies-mlx-command-buffer-windows-and-queue-scope
type: case-study
stage: prefill, decode, pipeline
operator: scheduling, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, launch-bound
semantics: contract-only
representation: bf16, fp32
boundary: cpu-gpu, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [profiling手册](../profiling-playbook.md) · [计时helper/stream反证](mlx-benchmark-evaluation-and-stream-completion.md) · [固定来源](../sources.md#mlx-cb-profile-sources-20261017) · 完整记录 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十三轮。在未修改的MLX0.32.3 wheel上运行原MLX-LM Qwen完整24层，取得公开`MTLCommandBuffer.GPUStartTime/GPUEndTime`完成回调窗口。12个stock/off/on×BF16/FP32×两重复进程保留144个完整词表向量，6个prefix重新独立计算；同scope输出跨三模式、两轮、两重复全部逐位相同。on取得1299条真实CB记录，其中582条属于48次模型forward。

这是**当前子进程、当前具体CB类和绑定queue的窗口证据**，不是逐kernel、shader busy、全设备timeline或无扰动profile。FP32的72个raw输出过原数学门槛；BF16对应72个仍拒绝，不因观察前后一致而取得质量或性能资格。

## 1. 不改wheel，但观察并非无状态

[原MLX提交链](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)的每个stream持有自己的command queue/encoder/CB；`CommandEncoder::commit`装原completion handler，再调用公开`commit`并创建下一CB。安装包不导出本研究需要的CB对象。本轮因此采用原创ObjC++ helper：用公共Metal接口发现具体类，仅在实验子进程内给该类的公开commit方法加完成回调，再且只调用一次原IMP。

这与[llama.cpp第十八轮13个显式站点插桩](llama-command-buffer-timing.md)不同：**本轮是类级拦截，影响该类在当前进程的全部实例/queue，不是仅改一个对象或一个MLX站点。** 不注入其他进程，不改上游源码、磁盘wheel或系统配置；当前类名`AGXG16XFamilyCommandBuffer`不是稳定Apple ABI承诺。debug/proxy或其他CB类可能不被覆盖，不能推荐成任意App的通用profiler。

helper为seen queue赋连续进程局部ID，内部暂时强持有queue至结束，以避免地址重用混淆；公开记录不含原指针/地址。退出前排空回调并恢复原IMP；若原方法来自父类，新增的方法表槽仍留到子进程退出，因此“恢复IMP”不等于完全恢复所有runtime metadata。queue持有和回调/锁/IO本身也可能改变资源生命周期与CPU成本，未给零开销、内存峰值或完整透明性资格。

stock不显式加载helper；off加载并做类发现但不装hook；on开启。三者调用同一模型/权重/输出需求。源码、原core扩展及安装libmlx文件SHA固定，但本轮未逐一枚举完整dyld/PSO闭包，不把这些文件身份说成可复现wheel构建或所有实际加载镜像证明。

## 2. 秒、Mach ticks与完成后的可用性

SDK26.5的`MTLCommandBuffer.h`规定GPUStartTime/EndTime是**host seconds**；零值是不可用，不是零耗时。只在completion handler读取，要求status4、正有限端点、end≥start。CPU独立使用`mach_absolute_time`与timebase；本机125/3，只用于CPU ticks，不能再乘到SDK秒上。SDK秒仅×1e9。

每条记录保留commit前Mach ns、callback内Mach ns、queue/scope/CB序号和SDK端点。复用既有CPU时钟合同的2µs括号余量；它不是本轮测得的绝对精度或置信区间。全部1299条窗口有限、时钟括号通过；模型582条也在各自host-ready窗口内。

源码CB完成和自有日志写完不同：host结束点先记录，随后才等该scope的回调日志排空。回调仍可能在host结束前竞争CPU，窗外drain不消除观察成本。错误/缺失/反向/零端点的CPU反例仍返回null；不因本批通过就删除拒绝合同，也不把新SDK时间回填到历史synthetic gputrace。

## 3. 实际测的模型程序

原MLX-LM `93e6d4a6447b3080861985947f0820e76fcb957d`、core读本`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`；M4 Max、macOS26.6.2/build25G83、CLT SDK26.5。复用同一官方BF16 checkpoint，每进程290个实际参数全量SHA匹配独立位展开；FP32控制只提升原值。

每进程2文章×2轮×3次forward：完整32-token prefill，再消费两个相同真实续文token；每轮fresh原KVCache，不是原`generate_step`、greedy采样或HTTP请求。scope计入Python输入数组构造、原模型调用和`mx.eval(logits, all KV states)`至output/cache-ready；不含后续BF16→FP32验证cast、完整D2H/文件/hash或observer日志drain。model load/参数核查、stream绑定、验证readback与idle另分scope。

144份151936维输出全部独立对6份完整24层FP64-reduction/FP32-stage参考检查，不借引擎CPU forward或历史同payload来冒充新CPU计算。原`.005+.005*abs(ref)`且relative-L2≤.005门槛：FP3272/72通过，BF160/72；它是本研究预算，不是“所有BF16模型不可用”的结论。

12个dtype×article×step完整payload身份组各只有一个SHA；round0/1、stock/off/on、17/29都相同。重复性/观察保值、数学通过、任务质量和性能采用仍是独立状态。

## 4. queue归属与CB数量

先对默认stream和新建foreign stream分别运行有界校准操作，on的四进程各得到不同queue ID1/2。模型forward明确放在foreign stream，582条模型CB都属于其绑定queue2。ID是本进程的软件编号，不是GPU硬件queue/引擎编号，不跨进程合并。

| 记录 | 当前完整数量 | 不表示什么 |
| --- | --- | --- |
| on全部四进程 | 1299 CB | 不是1299 kernel；包含startup/核查/绑定/验证/stream探针 |
| 模型forward | 582 CB / 48调用 | 不是24层各一个CB |
| BF16 prefill / decode | 每调用14 / 11 CB | 不授权其他shape/dtype/编译模式固定数量 |
| FP32 prefill / decode | 每调用14–15 / 11 CB | 同输出也可有不同打包，不能从CB数反推数学工作量 |
| on的8个100ms idle窗口 | 观测集合为空 | 不是整台GPU或所有类/queue都空闲 |

CB可能包含多个kernel、blit、event与内部等待；source以ops/涉及buffer sizes等决定提交，未测具体14/15差异的唯一根因。强持有/allocator历史、首次准备也不能仅凭该差异就归因某项优化。

## 5. 诊断量：sum / union / span

只列on、round1的中位数，prefill每cell4样本、decode8样本。原完整逐条记录、round0和stock/off都保存；数值不作为引擎或dtype排名。

| dtype /阶段 | host-ready ms | CB sum=union ms | CB span ms |
| --- | --- | --- | --- |
| BF16 /prefill | 7.369 | 6.610 | 6.793 |
| BF16 /decode | 3.527 | 2.741 | 2.930 |
| FP32 /prefill | 8.968 | 8.210 | 8.405 |
| FP32 /decode | 5.477 | 4.717 | 4.913 |

本批模型queue串行所以sum=union；span包含CB之间的空隙。union仍含CB内部等待，不是shader busy。`host−span`或`span−union`不能直接叫纯CPU成本、有效带宽或可消除瓶颈；其他queue/进程的活动未知。

FP32 stock/off/on decode host中位数约5.443/5.478/5.477ms，少量进程和历史状态不足以验证稳定observer开销。on的SDK窗口也不能作为未观察stock的无扰动替身。即使FP32数值通过，所有qualified performance仍null；没有新增Instruments、per-kernel counter、M5/NAX或功耗资格。

## 6. 可执行合同与复查

第二十四轮已继续补[实际dispatch/PSO/函数和父CB的可重放关联](mlx-dispatch-pso-and-function-association.md)，并观察[attention/wide/split-K](mlx-attention-routes-and-wide-gemv-observed.md)。这是新实验，不回写本轮旧scope/582模型CB/144向量，也没有把CB时间变成逐kernel时间。

CPU合同从全部原始SDK行重算scope/queue/时钟/并集，普通tests不加载helper或启动GPU。显式客户端必须新ignored目录与supervisor、固定dylib/源代码/资产并串行GPU；只读原件。64-byte blit自检和一个模型smoke另外归档，不计144输出/1299记录。

可选独立builder只编译原创helper，不导入MLX或启动GPU。复现要选新的ignored parent，保留collector读取的build receipt basename，不能覆盖本轮sealed binary/plan；例如：

随后原client的prepare/run显式传该profile root、固定core/LM源码与已有model/token资产。新builder已在另一个fresh parent成功编译；这不把新binary冒充已重跑整个144向量矩阵，也不授予可复现wheel构建资格。

复查条件：具体类/IMP/hook、SDK/OS/SoC、source/wheel/native文件、stream/queue、output demand/KV/shape/precision、collector与consumer改变。未覆盖类、callback缺失或queue绑定不唯一应拒绝覆盖结论，不通过“文件非空/exit0/有时间数字”兜底。下一步继续更长请求、可信counter/源到pipeline关联和代表性消费者，不能由本轮CB窗口推算ISA或全设备利用率。
