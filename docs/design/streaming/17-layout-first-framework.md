# 17 · 布局优先框架：统一合同与端到端执行

[目录](README.md) · [配置唯一规范](02-configuration.md) · [实现矩阵](18-code-change-matrix.md) · [当前证据](13-implementation-progress.md)

日期：2026-09-16。状态：**待实施的整合合同**，不是新一轮运行/性能成绩。本文收敛框架边界；
02/09/10/12仍分别拥有参数、编译算法、执行协议、验收阈值的解释权，不再引入第二套配置或调度器。
后续实现应先完成一个LTX完整请求，再扩模型；本轮只修改文档，不放开生产资格。

后续实施展开见[19](19-layout-first-architecture.md)（对象、算法、时序和峰值算例）与
[20](20-layout-first-implementation-and-acceptance.md)（当前v2接缝及逐PR验收）。最新构建/host/Metal记录见13第11节。

## 1. 架构结论与不做的事情

统一的是“资源怎样排列、何时允许读写、怎样验收”，不是把所有模型重写成同一个forward。
框架分为控制面、数据面和证据面：

```text
控制面（每请求一次；explicit route only）
  intent/profile -> metadata snapshot -> layout compiler -> sealed plan
                                         |                    |
                                  source/binding tables    eligibility
                                                              |
                                           optional same-layout admission
                                                              |
数据面                                              request coordinator
                                   text -> denoiser -> VAE -> export
                                               |
                                    model sampler / pass inputs
                                               |
                                StageExecutor + SlotSafetyTracker
                                  /                         \
                         bounded I/O workers          model adapter
                         pread/convert/fill         prepare/encode/fence
                                  \                         /
                                   fixed slot backing bundle

证据面：plan/actual计数 + 可选trace -> 独立verifier -> 人工review资格记录

未启用新streaming -> 现有legacy planner/session/loader（不绕进上述数据面）
```

“GPU-only”指模型计算只选GPU；CPU读文件/转换和owner提交仍是必要辅助工作，不代表禁止CPU线程。
以下方案明确排除：运行时预算搜索、通用tensor算子DAG调度、自动改变精度/shape、全量CPU权重缓存、
每模型自建预取线程池、为统一接口强迫大内存resident经过逐block pump。

| 决策 | 原因 | 代价/边界 |
|---|---|---|
| 显式K/G/P/D/Q，先布局后admission | 可复现；预算变化不会改变计算/布局 | UI需提供完整preset，不能只把K传给下层猜参 |
| 固定slot backing，循环更换content | 将分配从refill移出；容量可解释 | 小块也占对应field的最大capacity |
| 单owner + 有界fill workers | 所有权/取消/账本可审计 | owner过重会限速，需测prepare与唤醒延迟 |
| kernel不动，adapter只管binding和边界 | 降低数值与默认性能风险 | 融合跨block时可能暂不支持该布局 |
| request retention先行 | 失败清理与identity最容易闭环 | 不继承legacy跨请求权重cache性能 |
| qualification与内存guard分离 | 功能验证与全请求预算证据不同 | layout-only不能宣传Y上限/零swap |

## 2. 四种“布局”必须在数据结构中分清

1. **执行布局**：真实block访问顺序、prefix、group、pass及合法边界。group不改变block计算顺序。
2. **存储布局**：每slot各field的backing容量、对齐、storage mode、可复用兼容类。
3. **读取布局**：从哪个文件的哪些range，经过什么转换，填到哪些destination spans。
4. **生命周期布局**：text/conditioning/upsample/activation/VAE/output何时产生、何时结束最后使用。

只把K写进config是意图，不是可执行计划；只算出`K * block_bytes`是局部容量，不是完整内存模式。
`LayoutCompiler`不应变成新的模型loader；模型描述负责语义，compiler负责确定性展开和验证。

### 2.1 Source身份与backing身份不可复用一个概念

当前`layout.hpp::FieldSpec.storage_id`已明确为destination backing身份，materialization另含source/派生关系；
新descriptor扩展的实现范围见13第9节，整体关系仍须明确区分：

| 概念 | 示例 | 计量规则 |
|---|---|---|
| SourceRangeId | checkpoint某F32表的offset/length | 唯一文件内容与实际读取次数分开 |
| DestinationStorageId | CPU BF16 base、GPU row各一份 | 独立物理分配各计一次，不能按source名字去重 |
| BindingId | block/field到slot内span的引用 | 别名共享同一backing时只计storage一次 |
| ContentGeneration | slot 0本次装block 4 | 增加generation，不增加backing分配次数 |

LTX的同一源表可以产生独立base/row/conditioned副本；必须保留这些真实destination。
source去重、destination别名和重复pass读取三种账分别做。改变现有canonical字段语义必须升级内部格式revision和golden，不能同名静默换义。

### 2.2 拟议的sealed plan组成

```text
ResolvedStreamingPlan
  format_revision / model_descriptor_digest / layout_digest
  workload_identity: operation, format, shape, pass/step schedule, numerical flags
  source_table: artifact/range/format/reader revision
  backing_table: storage/field/capacity/alignment/owner/resource-domain
  binding_table: group/block/field -> slot/span/conversion recipe
  access_table: ordered blocks, prefix, pass transitions, last-reader contract
  resource_plan: non-slot lifetimes, known uppers, explicitly unknown resources
  construction_views: immutable adapter-specific projections of the above
```

这不是新增公共C ABI的逐字定义。C++首次可用有上限的vector/index；C adapter通过有owner的construction view借用，
不把Foundation对象、`std::string`或裸临时数组跨C传递。热路径只索引，不查JSON/字符串map。
执行期变量、wall、probe结果和资格签核不写回plan。Y/X/S改变只改变admission，不改变layout digest。

workload在构造前固定：输入尺寸归一化、frame/latent shape、实际sampler pass表、text padding/token bucket、
audio/I2V分支都要进入身份。若token数需先tokenize，允许受控CPU预处理，并记录其资源/耗时；
不先运行GPU encoder再把超出的token或shape偷偷改进plan。固定bucket内的实际长度在执行时核对，
超出描述范围拒绝或要求重新提交；layout-only可显示非slot upper未知，bounded不能忽略该未知项。

### 2.3 Checkpoint快照与缓存

metadata preflight取得header、受控的小型quant metadata、文件handle和identity；期间不materialize整块权重。
按range做checked arithmetic，验证dtype/geometry/packing、跨field引用和读取上界；source bytes与destination bytes分别保留。
describe不能标榜绝对“零文件I/O”：LTX已有header和小型metadata读取，需报告这些成本。

执行绑定同一个打开的文件与immutable metadata快照，不按路径重新打开另一个同名文件。路径替换、文件截断、
相同大小改内容都是测试用例。打开fd避免路径被替换后换读另一文件，但不防止原inode被原地改写。
首发要求checkpoint在请求期间不可修改；运行前后stat检查只提供变化检测，不是内容不可变的证明。
资格身份用受信artifact manifest/content digest；未验证内容身份不得沿用认证。完整hash在显式注册/校验流程执行，
其有效性/失效规则入报告，不能用文件名+mtime冒充强hash，也不能每请求默认重hash几十GiB隐藏启动成本。
无法建立该信任条件时只允许实验/plan-only；对抗并发写入的快照隔离属于独立工作项。

先不做全局descriptor cache。后续cache必须以artifact、shape、格式、adapter/backend revision为key，
有容量上限、失效规则和owner；request cache命中也不能授权不匹配的运行tuple。

## 3. 公共API与session的接线合同

当前`generate/prepare`统一拒绝active manual，`results.mm`也只返回plan-only；正常LTX session仍使用legacy创建/销毁。
内部`ltx_native_create_streamed_v1()`有tiny实模型证据，v2借用snapshot入口现有K1/K2/K3及真实connector的native smoke；
这些不意味着公共入口接通。下一步不应直接删除拒绝语句。

### 3.1 建议内部接口，默认实现必须惰性

在`ModelSession`上提供metadata probe和显式binding钩子；以下为设计草案，不是已存在API：

```cpp
// 返回自持有的metadata快照；不分配GPU，不卸载现存模型。
StreamingPreflight describe_streaming(const ExecutionPlan&);

// 仅explicit路线调用。默认实现拒绝；不改变旧generate签名。
void bind_streaming_request(StreamingRequestContext&);
void unbind_streaming_request() noexcept;

// 状态返回不能只表示“已发送cancel”。
StreamingDrainResult drain_streaming_request();
```

`StreamingPreflight`包含descriptor和adapter construction view，owned snapshot存活至所有fill退出。
`StreamingRequestContext`是**待新增外层对象**；`context.*::StageExecutor`继续仅管理单stage，不重新包装成第二个executor。
`ExecutionPlan`可增加nullable immutable plan引用；`RunResult`可增加optional真实streaming报告。
legacy不调用describe/bind、不建立空plan/无限预算guard，默认JSON不凭空增加streaming字段。

### 3.2 新路线顺序与事务边界

```text
现有engine/global GPU互斥
 -> parse/make_plan（语法与路由）
 -> explicit时describe + compile
 -> adapter/backend能力检查 + production registry资格检查
 -> bounded时完整内存认证检查
 -> 必要时清旧缓存 + 同布局admission（这里之后才允许mutate session）
 -> 构建request context + 显式binding
 -> 原model sampler执行阶段，逐pass调用同一个StageExecutor
 -> drain/资源回收/输出确认
 -> actual验证 + 成功报告 + unbind
```

请求解析错误、unsupported、未认证、unknown upper均在卸载旧session前返回。
admission阶段若已清理旧cache后预算仍不足，明确报告这一副作用；不承诺GPU分配失败可以恢复旧cache。
构造失败逆序释放已成功创建的资源；不自动fallback legacy、不修改P/K、不重试可能导致另一种布局的分配方案。
`prepare(load-only)`、`prepare(warmup)`、`generate`分别描述访问模板；generate通过不能自动授权前两者。
不支持的prepare明确拒绝。prepare不能提前执行/消耗未来generate的pass或把request pool暗中保留成session cache。

### 3.3 实验与发布不能使用同一个自签开关

建议内部test build允许从受控test harness调用真实C API/session exact分支；普通release build只认registry。
用构建选项与独立测试产物表达实验权限，不增加用户JSON `unsafe=true`、不读取任意环境变量绕过资格。
实验和production使用同一compiler/adapter/executor实现；差异只在资格来源及结果标签，报告必须含`experimental`。
测试构建不能安装成默认service。正式性能测量用release优化、trace off的实验候选，不能拿debug/sanitizer对比release。
资格记录绑定build/adapter/descriptor/workload范围和证据；profile只能选布局，不能签发资格。

## 4. 所有权：生命周期必须比callback长

| 资源 | Owner | 借用者 | 可释放的必要条件 |
|---|---|---|---|
| descriptor/header/open file | request metadata snapshot | fill workers、adapter | 所有fill已退出，最后借用已结束 |
| prefix/slot backing/每K份scratch | model adapter state | CPU fill、GPU readers | 写者退出且所有读者完成 |
| tickets/mailbox/control | StageExecutor | workers、GPU callback | producer关闭、回调全退、owner已消费/清理 |
| rope/conditioning/activation | model pass scope | encode以及实际GPU读者 | pass drain完成；失败时也不能随栈展开释放 |
| upsample handoff | request/model sampler | 下个stage | producer完成且consumer最后使用结束 |
| optional guard | request context | owner-side allocation/release桥 | 对应资源终态已入账，不能先销毁回调上下文 |

LTX当前`denoiser_`使用`unique_ptr<..., ltx_native_free>`；该void deleter不能表达新exact destroy失败后handle仍须保留。
首批应新增独立request-scoped exact handle owner，legacy成员不改；所有exact run/connector/upsample通过明确的本次handle访问。
调用status-returning `ltx_native_streaming_destroy`，成功后才置空。不能为了复用旧ScopeExit而丢失非空quarantine handle。

### 4.1 Request状态机

```text
Created -> Resolved -> Eligible -> [Admitted] -> Constructed -> Running
                                                               |
                                               success/cancel/error
                                                               v
                                                            Draining
                                                          /          \
                                                 safe terminal     Quarantined
                                              success/cancel/error   |
                                                                owner retry
```

取消是终止意图，不是free许可证。sticky primary error保留；cleanup错误作为secondary记录，不覆盖真实失败原因。
quarantine必须保留adapter、sampler workspace、file、mailbox、worker可见状态和guard，不只保留一个MTLBuffer。
quarantined engine不回idle，不接后续请求；只有owner证明安全后能释放。不得detach线程后让外层对象析构。

当前`shutdown_and_join()`仍可能阻塞，`drain()`也没有硬deadline。stall timeout仅是无进展检测，
不能写“请求在N秒内一定退出”。首发本机实验需报告此限制；要有强制服务恢复SLO，应单列受监督worker进程隔离，
不在本批假装可安全强杀同进程I/O/GPU线程。嵌入式C API的engine_free遇到quarantine时也须有明确保留owner方案；
若无完整安全销毁设计，服务集成门不通过，不能依赖C++析构terminate做正常错误处理。

## 5. Multi-slot执行：先有可证明的重用，再谈overlap

### 5.1 Slot不是block，也不是线程

K是固定backing bundle数量；G决定一次装多少连续兼容block；P是独立常驻prefix；D是lookahead；Q是fill并发。
同一slot会服务多个group；新ticket绑定request/pool/pass/step/group/slot/generation。
当前需求j是下一尚未成功提交compute的group，允许派发[j,j+D]，静态`slot=j mod K`。

```text
backing:  Allocated ---------------------------------------- Released
content:  Vacant -> Loading -> Ready -> InUse -> Vacant -> ...
permission:  fill写     owner派生写     GPU只读    再授权下一次fill
```

只收到一个GPU queue完成不足以复写。全部reader已登记(sealed)且全部实际完成，才能复用；
“已提交/已scheduled”不是“已完成”。CPU/GPU共享地址也需要这个读写先后合同，不能用shared storage免除同步。
平台同步语义参考文末R1；具体是否已经同步必须以模型backend实现和GPU试验核实。

### 5.2 Owner pump每轮的固定工作

1. 检查取消/overflow/主错误，消费有界completion；校验ticket和generation。
2. 对需求优先的合法窗口派发fill，消耗空slot与队列credit；不阻塞owner做整块同步磁盘读。
3. prefix每pass算一次，首次suffix读取可与prefix compute重叠；这项startup策略要进入P1语义对照。
4. 需求group Ready后执行owner prepare/bind，提交原有compute，封存reader集合；不能跳过未ready的前块。
5. 无进展时等待事件；不busy spin，不把现有5ms最大wait误读成每块固定sleep。
6. pass末按声明边界drain；所有pass完成才finish并join。失败走同一个安全清理入口。

worker只负责预分配span内的read/convert，不能分配GPU对象、修改账本或自行启动下一次预取。
read可乱序完成，compute仍遵循模型顺序。错误的部分fill永远不能成为Ready。
当前LTX prepare复用每slot scratch：fill完成后才能prepare，GPU读期间也不能下一次fill。
不能机械把所有scratch改成Q份；每K/每Q/owner共享的数量由互斥使用证明决定并写入backing表。

### 5.3 双槽、三槽的收益与边界

以同构groups、Q=1、无额外竞争为示意：

| 时段 | slot 0 | slot 1 | slot 2（若K=3） |
|---|---|---|---|
| 启动 | 读g0 | 空闲 | 空闲 |
| compute g0 | GPU读g0 | 读g1 | 有D/credit时可排队g2 |
| compute g1 | K2时重填g2；K3时可提前填g3 | GPU读g1 | K3时g2可已Ready |
| compute g2 | K2时GPU读g2；K3时g3可已Ready | g1读者结束后可填g3（K2）/g4（K3） | K3时GPU读g2 |

表中先后依赖仍受Q和最后reader约束，不表示Q=1可同时读两块。K3提供额外就绪缓冲，不自动使磁盘或GPU快50%。
K1没有同槽读写重叠；K2通常已能形成稳态流水；K3是否改善取决于读入长尾、prepare、reader滞后和内存压力。
在同构、无争用、owner开销可忽略的理想模型中，n组每组load=L、compute=C：
K1约为`n*(L+C)`，K2稳态下界为`L+C+(n-1)*max(L,C)`。这些是仿真oracle，不是实测预测；
额外槽不能突破长期读取/计算吞吐瓶颈。实际模型还必须加prefix、prepare、pass barrier、竞争与完整外层阶段。
当前LTX边界helper报告已经完成的reader，不新增每组全局wait；这是该kernel现有执行语义，不可据此假设所有模型都异步提交。
H3/MLX必须各自证明completion，不能复制一个“completed=true”适配器。

### 5.4 Pass、class和两stage边界

首发同一兼容denoiser域：pool和Q个worker跨所有diffusion pass保留；内容按当前规则每pass重读。
LTX upsample前drain pass readers，保留weight pool，单独交接stage1/stage2 workspace；shape变化不代表weight布局一定变化。
每步派生conditioning重算，不能把同地址当同内容。最后一个pass后释放denoiser域，再进入VAE。

异构class扩展暂不在当前executor支持集。首个实现按09的class barrier先释放旧池、再建新池；
每个pool generation独立ticket和容量，不能宣称多class全请求只有一次建池。若以后保留多个class池以减重建，
必须显式改资源live intervals/retention revision和资格，不能暗中增加常驻峰值。
跨pass预取同理是后续显式访问模板扩展，不能藏在H3 adapter里绕过pass drain。

## 6. 内存与性能是两张约束表，不互相替代

固定field布局采用“每field最大值后求和”，不是“group总bytes取最大”；prefix、pool、staging、
activation/conditioning/VAE、control和framework envelope按实际同时存活计入ResourcePlan，详见05/09。
managed allocation gate可约束纳管资源；process采样不证明连续时间硬上限，MLX active memory也不是全进程RSS（R3）。
unknown非slot upper必须显示，不得填0。K降低而VAE峰值不变时，工具应解释“本峰值与slot无关”。

性能保护分三层：

- **结构层**：默认不实例化新框架；new steady-state不new/delete/pthread_create/parse/hash；kernel batch不顺手改。
- **测量层**：P0默认、P1同布局、P2守卫、P3低内存、P4改策略各自测；trace与release timing分离。
- **发布层**：任何默认回归阻断对应发布；没有P1证据的新adapter留experimental，低内存更快不能抵消P0。

不要追求最高overlap百分比；目标是合格质量下完整请求wall、尾延迟、峰值和完成率的Pareto最优。
增加Q/K可能消耗共享带宽与file cache余量；只能作为实验假设，不能按RAM档位无证据自动设置。
是否快于OS swap按16的P3实测；物理内存充足时不预期streaming必胜resident，因此默认路径永久保留。

## 7. 模型统一到什么程度

| 模型 | 共用框架部分 | 模型专属部分 | 第一个接入门 |
|---|---|---|---|
| LTX native | compiler、C bridge、pool、I/O、generation、报告 | 48-block metadata、ConvRot fill、双stage/upsample、video/audio reader | 完整prompt→视频，exact session cleanup，P0/P1 |
| H3 native | 同上，不再维护new-route私有pager | active block映射、norm/AdaLN、fusion读闭包、跨forward语义 | 先K2/G1，同义startup/跨pass策略 |
| Flux | 外层component生命周期与报告先共用 | 双流/单流class、量化reader、MLX compiled binding | 组件输出materialize及引用释放；block暂不放行 |
| Z-Image | 同上 | 当前格式reader、weight map、图引用、VAE策略 | 组件级质量/峰值/默认性能 |

给新模型实现descriptor、construction/fill/bind、last-reader三类能力；不要以新增第五套scheduler作为接入方式。
MLX compile对闭包捕获和输入变化有专门语义（R2）；mutable slot内容与compiled graph的关系必须有测试，
不能“字典换成新array”就认定旧weights已释放。backend无法证明安全重用时保留component-staged或plan-only。

## 8. 配置、结果与错误的框架合同

沿用02及examples，不添加`slot_num`别名、新的隐式auto或嵌套预算方言。preset工具输出展开的完整参数，
hardware/profile只是用户显式选用的固定配置。RAM档位筛选候选，不参与compiler执行时改K/P。
内存充足用户不配新字段；精确指定streamed者即使RAM充足也不被自动改resident。

报告至少区分intent、resolved、execution authority、admission和actual。建议新增字段由18独立PR定schema：

```text
resolution_state: requires_checkpoint_metadata | resolved | rejected
execution_authority: production_record | experimental_test_build | none
actual: null（未执行）或实测的pool/slot/fill/byte/terminal数据
memory_enforcement: none | bounded_request
quality/performance evidence: 仅引用外部验收记录，不在每次请求自签
```

`actual`不能用resolved复制；descriptor/read错误、capability不支持、资格未覆盖、预算不足、内容identity变化、
reader失败、取消与unsafe cleanup分别报告。错误包含stage/pass/group/slot/generation（适用时），
不要泄露用户prompt或机器绝对路径到可公开bundle。fill计数不是物理磁盘I/O，slot bundle数不是MTLBuffer数。

## 9. 外部语义参考（不是本项目已实现的证据）

以下为实施时应核对的官方资料入口，不是本轮获得的测试证据；本轮检索未返回可引用正文，未据此新增平台版本承诺。
实际测试必须记录本机构建依赖版本，不自动套用网页最新版本；本文读写同步和统计口径要求以本地实现审计、实测验证为准。

- R1 Apple，Synchronizing CPU and GPU work：`https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work`
- R2 MLX，Compilation（闭包/输入/编译缓存相关约束）：`https://ml-explore.github.io/mlx/build/html/usage/compile.html`
- R3 MLX，get_active_memory（数组占用与allocator cache等统计边界）：`https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html`

源码事实与测试结果以13和当前源码为准。上述平台资料不能替代真实Metal/MLX lifetime测试，也不能证明性能非劣。
