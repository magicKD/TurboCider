# ExecuTorch MLX：bytecode、slot复用与lazy图物化

<!-- knowledge
id: technique-gpu-ane-case-studies-executorch-mlx-bytecode-slots-and-materialization
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [本轮摘要](../../MAINTENANCE.md) · [来源/行范围](../sources.md#executorch-mlx) · CPU记录 · 计划 · [验证（过程原件已移出）](../../MAINTENANCE.md) · [Core ML delegate对照](executorch-coreml-delegation.md)

2026-10-17。本轮将ExecuTorch从既有Core ML读本扩到独立MLX delegate：导出器产生自定义FlatBuffer bytecode，C++解释器调用MLX GPU primitive。它不是Core ML，也不自动调用ANE；同仓库另有`backends/apple/metal`，其路径存在不代表本轮已审阅或运行那个backend。

两个关键区别：slot数量减少不等于Metal物理buffer数量减少；输入/常量采用借用式包装，也不代表完整delegate无copy。所读正常输出路径仍做contiguous/可能dtype转换、GPU等待和宿主memcpy。另有lazy图中途eval阈值，但它不是硬内存预算。以下C++结论为源码证据；本轮只运行原始Python的纯元数据片段，没有运行导出、MLX runtime、模型或GPU/ANE。

## 1. 固定版本与真正的MLX依赖

固定`pytorch/executorch@3f15285f3b668b0ecdac6fa985bc207edeb50efc`，tree `05bb5b71a1264c083566c606231d17cfe521a16e`，commit日期2026-10-16。与第六轮Core ML是同一仓库/ref的新读本，不计新增独立引擎实测。26份选定文本/patch按原Git blob核验，实际行范围见[来源](../sources.md#executorch-mlx)；不是整个仓库的人工审计。原root AGENTS为指向CLAUDE.md的symlink，两者均读取；自有下载目录将link文本单独保存，不改只读参考checkout。BSD许可与原文件仅留ignored `outputs/gpu-ane-executorch-mlx-round35-20261017`。

[CMake](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/mlx/CMakeLists.txt)从`backends/mlx/third-party/mlx`构建静态MLX，父Git tree固定gitlink `1f8e74e3f12f31365464a6867c6579f0e9b29d85`。这不是本机既有MLX wheel的完整构建身份，不能借用前轮wheel实验为此delegate背书。

构建链还先应用两份本地patch：Metal SDK/deployment flag随macOS/iOS目标选择，以及旧Metal frontend的`remove_addrspace_t`兼容分支。`apply.sh`通过reverse-check跳过已应用patch，context漂移则失败；外部构建以Metal ON、CPU/CUDA/Python bindings OFF配置。**依赖gitlink、patch集合、构建配置都属于资产身份。** 本轮仅核声明/patch正文，未下载该MLX依赖、未apply、未configure/build，也未执行文档的安装/切换Xcode命令。

固定官方文档将该delegate标为experimental；iOS设备执行也有未覆盖CI的限定。源文有iPhone/Whisper性能表，属于源报告，不作为本机测量，更不能推出同模型一定快于Core ML或独立ANE。

## 2. 分区：支持一个op名称不等于支持每个调用

[partitioner.py](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/mlx/partitioner.py)利用同一`MLXProgramBuilder`检查实际node/handler的支持状况，而非只看静态op表。具有专用support-check的pattern handler可只检查；其他所选路径仍可能做更多lowering工作，不能把check-only理解为绝无权重准备成本。

`ops_to_not_decompose`只保留支持的目标；同一target有任一调用被拒绝，就从保留列表删除整个target。源码明确选择较粗的target级策略，未启用逐node filter，以避开另一条edge路径的兼容问题。好处是避免留下既不分解、又不能lower的调用；代价是一个被拒绝调用可能让其他合法调用也失去高层融合机会。

支持检查不调用最终`build()`来具体化SymInt，避免提前固化动态shape；分区还将被内部用户使用的`sym_size`吸入partition。它是保留符号关系的工程措施，不证明任意动态shape/范围都有效。该ref的MLX partitioner对旧`to_edge()+to_backend()`工作流直接报错，要求`to_edge_transform_and_lower()`；与第六轮Core ML的deprecated描述不能互换。

未委托node仍由外层ExecuTorch执行计划负责；不能把“MLX以外”自动标成优化Metal、Core ML或ANE。导出、分区、每node support、实际C++运行及物理设备放置是五份不同证据。本轮没有执行这条完整导出链。

## 3. bytecode和常量：名称空间不等于内容资格

`preprocess.py`将图交给builder，serialize为带24-byte header和`MLX0` magic的payload；权重经NamedDataStore另存，不嵌入这条delegate payload。schema将Constant、Input、Output、MutableBuffer、Temp分区计数，另有main/init chain、I/O map、named slots与tensor metadata。

preprocess用图字符串SHA-256的前4个hex字符作named-data命名prefix，意图减少多method常量同名冲突。它不是完整内容寻址、无碰撞证明或checkpoint/LoRA质量key。`get_named_data_store()`逐份取得CPU contiguous常量、加入16-byte对齐的数据条目并删除原EP引用，以降低AOT峰值；它会改变导出对象持有的权重引用，不是运行时“动态权重无成本替换”。

源码另有serializer字符串去重回归：相同Metal source/header只写一次，loader可进一步intern。我们只读该测试的断言，没有运行serializer/loader，也没有生成或衡量.pte体积；去重字符串不等于常量物理页、MLX数组或所有method的state自动共享。

## 4. 真正执行的CPU部分：身份、局部ID和全局ID

[slot_manager.py](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/mlx/builder/slot_manager.py)将tensor与scalar value分开管理，SymInt/SymBool共享value ID池，各IdSpace还有各自池。`Slot`是冻结、按对象身份相等/哈希的记录：两个不同生命周期即使字段/局部idx相同，也不应在Python集合里提前合并。

`tmp_scope`把临时分配登记到最内层，在finally归还ID；禁止在scope外创建temp，并在`set_slot`处检查不能把仍归scope管理的temp绑定成持久node输出。后一个guard本轮只读，不构造假的FX node去冒充框架测试。builder的`_mark_read`另把原地alias的生命周期转给consumer，避免旧producer最后一次读后就归还仍活跃的输出ID；这也是源码证据，不是完整图运行。

本库probe按AST选出原IdType/IdSpace/Slot/IdManager、SlotManager的7个纯方法及builder的两个原始映射方法。所选函数体不改写；不导入torch/MLX/ExecuTorch/NumPy，不伪造Tensor、FX graph、export或GPU API。映射引用只用stdlib `SimpleNamespace(idx)`作可变整数记录；没有运行完整SlotManager/ProgramBuilder构造或lowering。

实际结果见全记录：

- 32组`width1…8 × generations1…4`，3600个被引用的Slot对象身份；另128个未引用constant/temp被过滤。原`_collect_used_slots`按tensor/value池、space、idx计数；`_create_slot_mappings`分别coalesce两类ID并回写所有引用，完整匹配独立参考。
- 最宽4代用例有320个被引用Slot对象、额外4个未引用记录，却只有40个tensor全局ID和40个value全局ID。是bytecode寄存器/逻辑slot数，不是80个MTLBuffer、峰值/RSS或320→80物理分配实测。
- 32组嵌套/异常scope，合计3000个Slot对象分配；64轮循环时每类池的high-water ID仍等于depth1…4，正常和自有CPU异常均finally归还。异常是host contextmanager控制，不是设备fault恢复。
- 独立新进程重跑的完整JSON逐字节一致；没有把两轮重复当两份独立硬件证据。

另保留一个容易误用的liveness观察：旧Slot在scope退出后`is_alive=false`，下一代复用同一idx时，对旧对象查询可变回true；两对象仍不相等、set中保留2份。该函数观察的是整数池ID当前是否被占用，不携带代际证明。使用规则已经要求scope退出后旧temp失效；**这不是上游执行器实际发生stale read的复现**，也不能据此撤销本来合法的离线ID coalescing。

## 5. runtime还有两层不同的内存压力控制

`ExecutionState`的per-execution容器不包含constant或mutable buffer。bind时计算temp最后使用点：无嵌套chain中只被该chain命名的temp可以在最后使用后reset；跨chain共享temp保留，含SCAN/IF的chain整体放弃这套局部释放，避免错释放分支/carry结果。即使容器引用被释放，lazy graph的下游节点仍可能持有计算依赖，所以不能直接从slot归还推算GPU峰值。

[Interpreter](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/mlx/runtime/MLXInterpreter.h)的`eval_threshold_bytes`为另一层：默认0，关闭时跳过byte accounting；开启后每条普通instruction取其触及的、仍存在的per-execution tensor中最大nbytes记账，排除constant/mutable。先计账，再释放死temp，再在pending≥threshold时eval所有live per-execution roots。加法饱和，不允许溢出绕回阈值以下。

这不是live字节精确求和：大输入只读也可能被计账，内部eval不一定冲减估计，read/shape等指令也不同于真实分配。SCAN/IF嵌套chain只累积，不触发此阈值eval；回外层时SCAN的collected输出已经stack进state，还会计入stack的新输出。推迟检查是为让应保留的roots可达，代价是可能越过阈值很多。

因此不要将512MiB设置写成512MiB硬上限、RSS预算或已测收益。中途eval还改变同步/融合机会和数值路线，源码的语义意图不取代目标模型回归。本轮未执行Interpreter、其原始测试或任何阈值性能实验。

`clear_cache_interval`又是不同机制：per-handle计数，但调用的MLX cache是process-global；默认不启用，flush之间仍可增长且后续重新分配。不能与逻辑slot复用、最后引用释放、eval阈值、session KV预算混成同一“显存优化”。本轮没有调用clear_cache或清理系统/用户缓存。

## 6. 完整输入/输出交接：正常返回仍有等待和copy

所读`tensor_to_mlx`对非空输入要求contiguous，核dtype及shape维度int范围，非空指针不能为null；empty输入单独构造zeros保留shape/dtype。非空输入通过no-op deleter包装外部指针。源码注释依赖输入有效至async_eval编码返回；本轮未审计所锁MLX构造/Metal buffer实现，也不证明任意指针、alignment或异步owner都满足物理零拷贝。

常量zero-copy compile flag在此header默认1；`FreeableBuffer`被handle保留，数组借用其指针并立即eval作初始化准备。关闭flag走copy构造。所读`load_constants`没有独立展示一个“任何zero-copy失败都在此处自动copy重试”的catch；不能仅照搬header概括性注释就授予所有失败回退资格。

正常execute顺序是：全局mutex内reset本次临时状态、绑定输入/选择session、解释构图、为输出加contiguous/dtype转换、async_eval → 解锁 → 各输出wait → 必要时resize宿主输出、核字节数 → memcpy到ExecuTorch输出 → reset本次临时状态。输出等待/copy在mutex外，为不同handle的提交和等待留下重叠机会，但没有物理trace就不称已观测GPU并发，更不是GPU+ANE overlap。源码同时要求上层不并发执行同一handle。

`write_output`是复制式交接，不是Core ML那种outputBackings请求/实际采纳判定。dynamic resize仍受宿主输出storage/ABI限制，不代表任意strided view都可直接写。异常返回Internal不自动变成CPU fallback；所选catch里没有完整state snapshot/rollback，也未做故障注入，不能称已证明设备failure/drain缺陷或恢复。

## 7. session与cache：reset本次图不等于重置请求KV

`ExecutionState::reset()`清inputs/outputs/temps/values，不清constants、mutable buffers或off-graph cache。多session manager按`session → handle → MutableBufferData`分配，首次执行相应组合才初始化buffer，执行前换指针；权重仍共享。`skip_mutable_buffer_init`只在active multi-session load owner存在、且init chain不引用mutable buffer时允许，避免留空默认state或重复默认副本。

有session但未选择active token、context不匹配、未知token等会拒绝，不悄悄换用default state。registry mutex只保护查表；execute取得的raw pointer之后仍依赖caller保证active session不被并发destroy，并串行化同一loaded program执行。session隔离不是自动并发许可，独立handle的state也不自动成为跨prefill/decode共享KV。

另一个off-graph KV入口由cache key注册对象，handle持有shared owner、state持有typed view；本轮只读绑定与Pool的growth/read/write入口，未深读所有cache controller实现或运行多method消费者。因此不推断beam/fork、回退、ring复用或跨session任务都已正确。

## 8. 与Core ML delegate怎样公平对照

| 边界 | 本轮MLX源码 | 第六轮Core ML源码 |
| --- | --- | --- |
| 导出资产 | 自定义bytecode＋NamedDataStore常量 | MLProgram/package或compiled asset，可多function共享 |
| 执行入口 | 解释器构造lazy MLX GPU图 | MLModel prediction与compute-unit允许策略 |
| 输出交接 | contiguous/cast、wait、host memcpy | 请求backing，实际未采纳时copy；特定失败忽略backing重试 |
| state | mutable容器/session pointer与另设cache owner | MLState、模型/function与caller状态合同 |
| 内存控制 | slot ID、last-use、eval阈值、全局MLX cache各自分层 | compiled/loaded asset cache、MultiArray lease与模型生命周期 |

两列都是固定源码对照，不是速度榜。今后若做比较，应冻结checkpoint/量化、shape、state代际及完整consumer，分别计AOT、load/init、首次、warm、转换/readback、KV驻留和请求质量；全程仍须空闲准入。不能把另一个MLX wheel的FFN计时、CoreML All policy或作者iPhone数据拼成本次异构加速结果。

本轮因空间仍低于历史10GiB大动作储备，只进行少量源码下载、文档与CPU metadata工作；执行上限为raw32MiB、下载8MiB、curated JSON2MiB、隔离snapshot64MiB、空闲≥2GiB。未降低设备/build准入门，未新增GPU/ANE/native build、依赖安装或模型权重，未清用户文件/cache。24项可移植测试与最终隔离范围见[验证（过程原件已移出）](../../MAINTENANCE.md)。
