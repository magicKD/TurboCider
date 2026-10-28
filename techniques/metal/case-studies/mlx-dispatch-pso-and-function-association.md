# MLX dispatch到PSO、函数与CB：可重放关联，不用symbol清单代替执行

<!-- knowledge
id: technique-metal-case-studies-mlx-dispatch-pso-and-function-association
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

[GPU导航](../README.md) · [此前CB窗口](mlx-command-buffer-windows-and-queue-scope.md) · [实际attention/wide路由](mlx-attention-routes-and-wide-gemv-observed.md) · [来源](../sources.md#mlx-dispatch-sources-20261017) · 汇总 · BF16事件 · FP32事件 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十四轮。在未修改的MLX0.32.3 wheel和原MLX-LM Qwen上，进一步关联实际公开API的dispatch编码、encoder最新PSO绑定、创建时函数名与父CB完成窗口。12个stock/off/on×BF16/FP32×两重复进程保存144份完整151936维logits；12个prefix重新独立计算。24个完整payload身份组跨模式/重复逐位相同。

四个on进程取得56845条metadata事件；25520次direct dispatch编码中24336次属于48次模型forward，全部有已捕获PSO函数名及status4父CB。**仍没有取得逐dispatch GPU起止时间或ISA**。source family汇总只辅助导航，不是IR节点/层编号的所有权证明。

## 1. 五种不同状态不能合并

```text
PSO创建返回：函数名 + 对象token + 资源上限
  → encoder绑定PSO（可以被下一次绑定覆盖）
  → direct dispatch编码：当前PSO + grid/group
  → 父CB提交
  → 父CB完成：公开SDK窗口
```

创建过不等于绑定过，绑定过不等于dispatch过；编码后也不等于已提交/完成。本轮独立原生自检使用三个自有函数，两个输出各128个scalar由原生端完整核对：

| 控制函数 | 创建 | 绑定 | direct编码 | 父CB完成 |
| --- | --- | --- | --- | --- |
| `tc_created_only` | 是 | 否 | 0 | 不给它捏造GPU执行 |
| `tc_bound_only` | 是 | 是，随后覆盖 | 0 | 不给它捏造GPU执行 |
| `tc_used` | 是 | 是 | 2：threads/groups各一 | 同一个CB完成 |

这两次dispatch的grid含义不同：threads=`[128,1,1]`与threadgroups=`[4,1,1]`、group=`[32,1,1]`都请求128个逻辑线程；把二者同乘group会误算。公开selftest仅保存native全值检查bool与完整API记录，不含输出bytes；CPU离线独立重放的是三状态/关联，不假称重新验算了未保存的输出。

## 2. 为什么必须在PSO创建时取得函数名

SDK的pipeline label可为空：本轮Model捕获的126个PSO对象全部label=null，自检三个也如此。若只看label，就会丢失真实函数关联；若扫描整个metallib的symbol，又会把没有dispatch的函数混进来。

[MLX `Device::get_kernel_`](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)持有`MTLFunction`后调用公开pipeline工厂。本轮原创ObjC++观察器在返回时记录`MTLFunction.name`和PSO对象token，再在实际encoder的`setComputePipelineState`、direct dispatch处关联。

PSO资源属性包括threadExecutionWidth、maxTotalThreadsPerThreadgroup、staticThreadgroupMemoryLength。它们不是寄存器数、实际occupancy、总动态TG内存或带宽。函数名也不是shader机器码hash、完整specialization key或逐层模型标签；不同库/版本可能同名，不能跨进程把数值token当稳定二进制身份。

创建返回记录的host区间包含原工厂调用以及随后的getter/日志开销；API内部可能命中driver cache，也可能嵌套另一工厂调用，**不叫纯编译时间**。本轮Model实际捕获的是function:error路径；另外两个同步工厂selector虽安装了观察代码，但不据此声称当前消费者执行过它们。

## 3. 覆盖范围与生命周期

当前进程发现四个具体类：device、queue、command buffer、compute encoder。选择9个公开selector：三个同步PSO工厂、unretained CB工厂、concurrent encoder工厂、commit、setPSO及两种direct dispatch。这个类级观察影响当前子进程该类的全部实例，不是只改一个对象；不注入其他进程，不改磁盘wheel/上游源码。

当前运行类为`AGXG16CDevice`、`AGXG16XFamilyCommandQueue/CommandBuffer/ComputeContext`，不是稳定Apple诊断ABI承诺。异步工厂、indirect/ICB、Metal4或其他类没有由本方案自动取得完整覆盖。普通JSON中没有导出buffer内容或对象地址，内部opaque地址映射仅用于暂存关联。

queue/PSO被强持有至观察结束；encoder/CB不被长期强持有，由已捕获工厂给新对象分配新token，避免只靠旧地址判断身份。源码logger、锁、对象持有、ARC边界都可能扰动CPU/资源生命周期。退出恢复原IMP，但新增的继承方法槽可能保留到子进程退出；不授予透明性/内存峰值或生产可部署性。

## 4. CB创建scope不是dispatch所有权

[原提交链](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)每次commit后立即创建下一个空CB；它随后才装入下一次请求的工作。实测每个Model scope都至少有一次dispatch，其父CB早在另一个scope被创建。若按创建时scope分摊工作，就会给上一阶段多算。

本轮分别保存创建scope、编码scope、commit scope；当前所有编码和父commit scope匹配，不用创建scope覆盖它们。1267个完成CB中608属于Model；捕获结束还有8个已创建但没有观测到提交的空CB（每on进程2个），不是8次执行或泄漏证明，框架后续析构处理不属于该观察窗口。

encoder状态按序重放：每条dispatch必须使用该encoder最近绑定的PSO，encoder/CB外键必须存在，group乘积必须不超过**该PSO**的上限，父提交/完成必须唯一且status4。缺失、错误绑定、超限group、未知record kind等负控会拒绝，不以“有名字/日志非空”兜底。

## 5. 观察保值与数学验收仍分开

原MLX-LM `93e6d4a6447b3080861985947f0820e76fcb957d`、core读本`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`，M4 Max/macOS26.6.2/build25G83/SDK26.5。复用原官方BF16文件，每进程290个实际参数完整SHA匹配独立位展开；FP32只提升这些原值。

两文章、7/9/32-token fresh scalar KV prefill，再各消费一个固定真实续文token；不是greedy、原generate_step或serving。stock不显式加载helper，off只发现类，on装观察。原模型和所有KV状态eval至ready，readback/cast/文件另分scope。

144份完整词表输出对12个新算的完整24层FP64-reduction/FP32-stage参考：FP3272/72过原门槛，BF160/72。预算仍是`.005+.005*abs(ref)`且relative-L2≤.005，观察输出一致不替换该目标，也不是所有BF16不可用的结论。

全部1267个完成CB窗口正有限、时钟括号通过，方法沿既有SDK时钟合同。只有CB级GPUStartTime/EndTime；每条dispatch的GPU时间仍null。不能把父CB时间均分、整段归给最后一个PSO，或按函数计数排序成GPU热点。

## 6. 可离线重放与复查

标准库replay/tests逐条重放两份完整metadata capsule，不导入MLX/加载native库。重算完整原模型参考需显式原件和已有NumPy，仍不执行GPU；native客户端需新ignored目录、原始固定资产、编译好的原创helper及supervisor/GPU串行准入。

可选独立builder只编译helper；新parent内保留collector的receipt basename即可避免覆盖旧binary/plan，例如：

然后prepare/run显式传该profile root和固定源码/既有模型资产。编译成功不等于新binary已重跑本页144向量矩阵；原正式binary身份与新构建应分开。

CPU分类器v1只认BF16生成名字`CV`，把实际FP32 `Cf`保守放进unknown；v1代码/结果已归档，v2仅补该前缀。两份原事件bytes、全部函数名/数学/其他metadata完全相同；不是GPU或上游修复。自检和4向量smoke不计正式144输出/56845事件。

复查条件：具体类/selector/ABI、object生命周期、source/wheel/native/SDK/OS、factory/encoder/dispatch形式、shape/dtype/输出需求、scope或consumer变化。原始大向量/权重/native binary保留ignored；公开只是完整metadata/hash。源名与固定代码相符仍不是完整dyld/编译闭包或ISA证明，更不能继承到M5/NAX、其他引擎或逐kernel计时。
