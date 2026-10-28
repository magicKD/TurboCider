# Core ML版本与所有权审计：9.0、main、校准修复和三层清理不能混为一谈

<!-- knowledge
id: technique-ane-coreml-versioned-calibration-and-ownership
type: method
stage: ffn
operator: conv, gemm, layout, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: precision-loss, lifetime, measurement, storage-pressure
semantics: contract-only
representation: fp16, q8, w8a8
boundary: producer-consumer, weight-runtime
api: public, private, reference-only
-->

[ANE目录](README.md) · [固定源码/API身份](sources.md#version-sources-20261017) · [验证记录（过程原件已移出）](../MAINTENANCE.md) · [8.3实际校准批次](calibration-order-and-qdq-patterns.md)

本页补上2026-10-17（Asia/Singapore）的版本核查：**已合并源码不等于已发布wheel，已发布版本也不一定同时包含所有修复。** 没有安装/运行9.0或main的CoreML包，没有build native或测新的模型速度；新执行仅为原selected Python函数的CPU/自有文件fixture。

## 1. 当前release身份先核实，不能把main当成9.0

当次GitHub `releases/latest`返回[9.0](https://github.com/apple/coremltools/releases/tag/9.0)，published_at=`2025-11-10T22:22:40Z`；保存的45条release列表无下一页，最新非draft/non-prerelease也为9.0。它只证明这份公开GitHub响应，**不是PyPI查询、本机升级或未来仍为latest的保证**。

API响应UTC日期为2026-10-16、本地已为10月17日；二者不是两轮不同实验。9.0的Git tag commit是2025-11-04美国西岸提交，不能拿这个commit日期替换11月10日的release发布时间。

| 本页标签 | 固定commit | 本次资格 |
| --- | --- | --- |
| 8.3 | `07ffd86005e385d8eddfd22072836cfb457f7350` | 先前实际小图运行环境；本页原helper CPU对照 |
| 9.0 tag | `428d4b2658dfc44194f27f4f36870751be402ff7` | 源码/原helper CPU对照，不是安装运行9.0 |
| 捕获的main | `a207b1a3452eafe02f2d870f486287834af2c82d` | 2026-09-30提交的固定树；不是新release或兼容性推荐 |

新增只读sparse clone约2.6MiB；旧8.3/reference、产品venv与已保存设备样本未修改。来源包含三树各9个selected文件，加main两份上游regression test，共29文件身份；不是全仓库审计。

## 2. 一个版本表比“升级后已修复”更准确

原函数body仅从已核SHA的固定Git blob提取，不执行上游模块初始化；使用既有NumPy2.0.2。每版本14个fixture：8个range、2个concat、4个public forwarding，共42个。原public函数在call doubles上执行，不导出新版本模型，尤其group_size=0只测试转发，不执行真实分组。

| 选定行为 | 8.3 | 9.0 tag | 固定main |
| --- | --- | --- | --- |
| 两个range helper保留跨样本极值 | 否，4/8控制保留 | 否，4/8控制保留 | 是，8/8保留 |
| public入口转发group_size | 否，均用-1 | 是 | 是 |
| concat bridge fixture完整合并 | 否 | 否 | 否 |
| prefix/suffix白名单含`linear` | 否 | 是 | 是 |
| 同一白名单含`matmul` | 否 | 否 | 否 |
| debugger.step默认compute_units | CPU_ONLY | CPU_ONLY | CPU_ONLY |
| MLModel包含本页的`_close`/`__del__`对 | 否 | 否 | 是 |
| NumPy array wrapper显式dealloc内GIL guard | 否 | 否 | 是 |

range控制有wide-last的正例，因此旧版本不是“所有8个case都错误”；真正区分版本的是wide先出现、后续narrow仍应保留旧极值。main两个helper先判断已有项，再累积min/max；8.3/9.0先覆盖后合并。相同输入、相同独立union参考，不因新版本而换测试标准。

concat例仍是`[[a,b],[c,d],[b,c]]`，first-match后break留下重叠group；顺序`[[a,b],[b,c],[c,d]]`能合并。main修好range并未顺带使这个helper具备完整连通分量语义。**没有跑新版本完整concat模型**。

9.0多了`linear`的静态pattern，不等于所有等价matmul都被覆盖；也不证明任意frontend完整转换必然保留这个op。本批只执行source helper、读pattern，8.3实际matmul A8 no-op的设备证据仍单独保存，不回填成9.0/main运行结果。

## 3. PR状态、commit标题与release包含关系分别取证

| 改动 | 保存的一手证据 | 结论边界 |
| --- | --- | --- |
| [#2555](https://github.com/apple/coremltools/pull/2555) | merged=`true`，2025-08-01；patch将group_size传给校准器 | 9.0源码与原函数fixture均体现转发；不能据此认定range同时修好 |
| [#2671](https://github.com/apple/coremltools/pull/2671) | merged=`true`，2026-04-23；commit `d4f74165…`；model.py及owned-path测试 | main加入包/代理的提前清理，9.0没有这对方法；非公共API/非本机native验证 |
| range累计修复 `6840d7b66a78ed2207405e42e3799ebe6bae9730` | selected-file history日期2026-06-12；main函数体与CPU结果可核 | 标题含`#2724`，但PR与files API均404，**PR状态unknown**，不擅写独立verified merged |
| [#2829](https://github.com/apple/coremltools/pull/2829) | merged=`true`，2026-09-30；merge commit正是固定main HEAD；array wrapper及subprocess测试 | 读到GIL保护的source，不等于旧wheel已获修复或本机重现/修复native崩溃 |

404既不证明“没合并”，也不证明“已合并”；本页保留commit内容可核与关联PR不可核的不同状态。PR closed也不等于merged。保存14份primary API payload的hash/status/时间，wiki只公开筛选metadata，不镜像PR长正文或patch源码。

这些修复时间晚于9.0 release，有的又仅在部分文件中体现；最终采用依据必须是**所用源码/包里的具体实现与回归**，不是只按日期猜release包含关系。本页从9.0实际树验证缺少range累计、模型close和GIL-dealloc改动，而非仅以时间推断。

## 4. 临时weights、模型包、native代理与NumPy owner是不同资源

| 资源 | 本页读到/观测到的生命周期 | 不能推导什么 |
| --- | --- | --- |
| converter内临时weights_dir | `_mil_convert`创建`TemporaryDirectory()`，包构建后局部对象退出并隐式清理 | warning不等于目录还在；model._close不负责这个局部对象 |
| 临时`.mlpackage` | 旧版登记atexit；main `_close`按is_temp_package标记提前移除 | logical flags清空不自证实际文件已删；用户持有包不能误删 |
| `MLModel.__proxy__` / compiled model owner | main先将Python代理引用置None，再尝试删除包 | Python引用变None不是已观测native mmap/队列drain或设备同步 |
| NumPy input owner | wrapper持有`py::array`；main显式在dealloc中持GIL做dec_ref | 源码有guard不代表当前安装wheel、实际callback线程或物理ANE已验收 |

这四层可以独立改善，也可分别失败。不能把“包目录回收修好了”写成所有临时清理、native owner、buffer/线程安全都已解决。

## 5. 原converter的6个受控实验：main包清理并未改这段临时weights函数

三个版本的`_mil_convert` selected AST hash相同。每个版本执行两次原函数body，用调用桩替代proto/backend/modelClass，只有TemporaryDirectory和sentinel文件操作是真实的；所有目录都显式限定到本轮新output，不改TMPDIR或共享cache。

| 本地warning策略 | 每版本结果 | 实际文件证据 |
| --- | --- | --- |
| 捕获ResourceWarning | 1条隐式清理warning | 临时weights目录已删除，fake package仍在 |
| `-W error`语义的局部error filter | 1条unraisable ResourceWarning，调用继续返回 | 同样weights已删、package仍在 |

source函数这条路径在固定main也未显式`cleanup()`，但这是**原函数在受限桩上的实验**，不是安装main后的完整convert运行。它说明weights的warning与模型包的提前回收是不同控制面。

上轮8.3的40条warning只有finalizer栈，没有保存创建栈。本轮在该source路径独立复现同类warning，**没有唯一归因所有旧warning**，也没有将其视为实际漏删、ANE故障或新版本长期内存问题。warning可在文件已经删掉之后产生；相反，原包的atexit持有问题可能没有这种warning。

## 6. 原main `_close`：七个owned-path/失败控制，不是“无异常就清理成功”

只提取[main模型方法](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/models/model.py#L586)，保留class结构；代理用原创Python哨兵，删除适配器只接受本进程刚创建的**确切sentinel子目录**。没有删除用户模型/缓存或广泛扫描TMPDIR。

| fixture | 返回后文件 | 含义 |
| --- | --- | --- |
| 临时owned包 | 已删 | Python代理哨兵先释放，再rmtree |
| 同对象重复close | 已删，rmtree仅一次 | 局部幂等性控制 |
| user_owned标记为false-temp | 保留 | 不将非临时包当垃圾 |
| missing flags | 保留 | 析构容错，不表示拥有此文件 |
| 注入ignore_errors文件操作失败 | **保留，但package_path=None/is_temp=false** | 逻辑所有权清空不是物理删除证明 |
| 注入shutdown时_os=None | **保留，逻辑flags已清** | 该方法未碰文件；不据此断言整个atexit流程最终泄漏 |
| 注入_close异常后调用原__del__ | **保留，异常被吞** | no-throw destructor不是成功结果 |

最终主试只释放/删除了自建的六个临时weights哨兵目录与两个包哨兵目录，其余小型sentinel留在raw output用于审计；没有材料性用户数据被清理。这些文件观察与Python代理哨兵事件都不等于native CoreML对象或设备释放。

`_close`在源码中明确为内部接口，8.3/9.0又没有本页这对方法，不作为跨版本的公开部署建议。错误控制不是实际CoreML/权限故障复现，而是证明不能只依靠logical flags/exception状态验收清理。

## 7. native NumPy owner：SDK条件、GIL与测试范围

[9.0/main array wrapper](https://github.com/apple/coremltools/blob/428d4b2658dfc44194f27f4f36870751be402ff7/coremlpython/CoreMLPythonArray.mm#L19)在`BUILT_WITH_MACOS26_SDK`分支中增加int8输入类型；8.3无该分支。这是**编译SDK条件下的Python输入admission**，不证明任意9.0 wheel按该SDK构建，不等于内部Q/DQ graph、W8A8 lowering或物理INT8 MAC。

[main dealloc](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremlpython/CoreMLPythonArray.mm#L68)先release出handle，再持GIL dec_ref Python owner，针对CoreML可能从私有queue释放multi-array的情况。9.0 source没有该显式dealloc。

上游两份regression test均已读、未执行：包清理测试只检查自己创建模型的package_path，不改TMPDIR；NumPy owner测试在独立subprocess中使用weakref、PyGILState_Check和CPU_ONLY预测，最多等10秒确认owner释放。它们不是ANE placement/timing测试；本库的CPU桩也不能冒充已经跑了这些native/torch测试。

## 8. 可复现路径与继续问题

原创来源捕获、原函数/owned-file probe、标准库合同、与16个portable tests保存逐版本fixture、源码/AST与primary API身份。

curator不导入NumPy/CoreML，重核27个输入Git blob、14个API payload、全部selected AST及13个owned文件观察。Python3.11与系统Python3.9分别raw复核，生成的source/evidence文件逐字节相同。portable tests不读取外部reference或执行原函数体，只验证本地协议/metadata/小型数学及资格。

最初API抓取因#2724的404中止，后来显式保留unavailable响应；两次helper fixture错误（方法名排序、错误猜测dunder name mangling）在进入删除前或guard处被拒绝，失败receipt和归档代码保留，未当成上游bug。`__proxy__`以双下划线结尾，不能误当成`_MLModel__proxy__`。最终55个case和两个解释器raw审计另存，不覆盖旧失败。

下一步：在另有明确权限的隔离版本环境跑真实9.0/main导出/任务质量与生命周期/native regression；取得物理设备/线程证据，不能从本页helper或source继承；另核实#2724关联PR可用性和未来release包含关系。版本/branch、wheel SDK/native image、NumPy/Python/OS、校准图/数据/owner/线程/失败边界改变时重开。捕获的latest/PR状态按短周期复查，整体研究仍进行中。

随后[LUT批次复查（过程原件已移出）](../MAINTENANCE.md)中这些共享目录缺项已补齐，全库及隔离各1,296项、2 skip、无失败。这是后续工作区结构状态，不覆盖本页当时的失败receipt，也没有重跑或升级本页所限定的原函数/源码实验。
