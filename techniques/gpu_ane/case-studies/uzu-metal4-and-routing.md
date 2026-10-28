# Uzu：真正的 Metal 4 生命周期、GEMM specialization 与 M5 路由

<!-- knowledge
id: technique-gpu-ane-case-studies-uzu-metal4-and-routing
type: case-study
operator: scheduling
backend: metal, coreml, private-ane
device: cpu, gpu, ane
symptom: straggler, memory-bound
semantics: contract-only
boundary: request
api: public, private
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [硬件/API](../hardware-and-tensorops.md) · [第三轮证据](../../MAINTENANCE.md)

本轮固定 [`trymirai/uzu@2c16e96796de84abccf5ade4ef295f464d2b2a87`](https://github.com/trymirai/uzu/tree/2c16e96796de84abccf5ade4ef295f464d2b2a87)。它不同于首轮只读README的`ee5ec57…`。重点是`crates/uzu-engine`新实现，不能把legacy bindings与新backend视为同一套代码；未运行上游或M5。

## 1. 不只是用了 TensorOps 的旧commandBuffer

`MetalCommandBufferEncoding`实际创建MTL4 command allocator/buffer、argument table和compute encoder，维护常量bump allocator及scratch allocation pool。constant allocator初始256KiB是实现配置；表配置31个buffer binding也与本轮SDK的公开slot上限一致，不是GPU内部寄存器数量等硬件规格。

编码、可提交、pending、completed分成不同Rust类型。`end_encoding`要求活跃constant分配已交还；submit后不允许继续使用encoding接口。这种类型分层帮助约束生命周期，但最终仍要看completion和owner是否正确。

实际submit安装commit feedback handler。闭包保有command buffer、constant allocator和allocation pool，反馈到来才把command allocator放回cache；下一次取出后reset。即使调用者不再持有Pending，GPU使用的资源仍被回调闭包保活，而不是在异步API返回时马上回收。

`wait_until_completed`通过channel获取GPU执行时长或错误。这个duration属于GPU反馈时钟域，不可和CPU提交时间戳直接相减；也不包含模型加载/host编码。

## 2. Range-aware hazard 与 sparse residency

encoder记录GPU地址区间上的read/write。已有write遇到read/write，或者已有read遇到write，就插入dispatch→dispatch visibility barrier并清空跟踪；read/read或不相交区间不因同一个逻辑阶段自动同步。

这是应用显式管理依赖，不等于Metal4会替所有alias自动补barrier。区间、owner、allocator generation与真实kernel访问范围必须匹配；当前实现一旦有冲突就做较粗的encoder barrier，仍留有细化空间。

sparse resource操作还有单独queue/event：若发生mapping变更，先signal sparse event，再令compute queue wait。context把residency set关联队列。**映射完成、资源resident、producer完成、consumer可见**仍是不同条件，不能用一个“统一内存”概念代替。

## 3. 能力查询也有非公共边界

所读device helper通过动态selector查询`supportsMXU`、`gpuCoreCount`、placement sparse能力，并使用runtime/混淆字符串。`respondsToSelector`保护不支持的调用，不能把selector变成Apple公共稳定合同。

MXU缺失时回false，core count缺失时回8；后者是保守猜测，不可写作实测设备配置。迁移应将公共GPU family/API可用性、实现特有探针和fallback分开，不能连同混淆策略一起当“标准Metal优化”。本页MXU仍是GPU内矩阵加速，不是ANE。

## 4. GEMM：plan与shader specialization分离

`MatmulShape`包含M/N/K、transpose/leading dimension、weight prologue、bit/group、signed code、activation精度、gather、量化参数layout与output transform。plan选择SIMD-group或MXU、tile和split-K；pipeline key再绑定真正改变shader的specialization。

split-K不只看输出tile少不少：还检查K对齐、activation group、SIMD-group K分片、output bias/RHT约束等；不合法就保留unsplit。`encode_plan`还显式拒绝暂未支持的gather，不能把一个planner生成能力当成完整operator支持。

所读MXU core按operand选择不同数据流：dense、weight量化的staged路径、activation整数化路径。fragment helper真正调用MPP `matmul2d`，以operation helper的cooperative输入/输出类型装填paired fragments；不是简单把原SIMD-group核中的一条指令换掉。

B layout预处理、group scale、asymmetric weight correction、split-K reduction、bias/RHT的位置都可能影响收益与数值。kernel cache的复用不替代checkpoint/量化/LoRA数值资格；新格式也不自动继承老GPUfallback。

## 5. Attention：具体gate比“M5更快”有用

所读grouped路径的资格包括BF16、headD128/256、MXU、无sink/ring/sliding-window、正scale等。当前suffix2–64可走grouped；更长65–1024只为D256 causal prefill调优。否则还有GEMM、fallback、single-pass/two-pass等分支。

KV split的目标来自shape与GPU core数，部分decode区间有实测表；源码也保留跨chip/范围重测TODO。长prefill候选另按partial output和max/sum统计控制scratch，预算256MiB；不把这一个候选上限当整个request峰值。

grouped shader把GQA query/head行组织在一起，沿D分片协作，并可生成split-KV partial。combine用每支max与sum重标定后归并，跳过sum为0的空分区，再在最终输出位置应用gate。不能把各分区已归一化输出简单平均。

## 6. #973：同一个PR中临时规则也会被下一patch撤销

当前HEAD标题引用[PR #973](https://github.com/trymirai/uzu/pull/973)。本轮取得两段patch并核对当前route：第一段曾给短KV临时指定two-pass；第二段去掉grouped的KV最小长度限制，并撤销该临时旁路。

所以不能把两段文字累加成最终规则，也不能沿用旧`KV>=1024`门槛。第二commit标题的“2–3x”仅是作者陈述，本轮没有其完整环境/shape复现，不作本机倍数；PR网页状态未成功取得，不伪称已核查状态。patch、当前源码与HEAD分别保留身份。

深度1的`git show --stat HEAD`曾把整仓库列作变化；它不是可靠PR差异。实际学习使用下载patch和最终文件，不据浅边界统计宣称一个PR改了全部源码。

## 7. 推荐验证

先验证合法layout与实际路由，再做同格式的SIMD-group/MXU对照、split-K/scratch、single/grouped/two-pass attention与不同suffix。M5计性能；本机M4只可验证相应公共API/旧核，不冒充MXU硬件成绩。

另检查callback丢弃Pending时的资源寿命、allocator仅在completion后复用、稀疏mapping event、range alias、error反馈与prepared kernel cache。源码使用Metal4 API不意味着每个模型算子都更快，需回到完整consumer和request。

后续[公共API小客户端](metal4-public-lifecycle-probe.md)已在M4验证相同shader两条提交路径和显式callback保活；不是运行Uzu，也没有验证其私有能力查询、sparse mapping或所有错误路径。另见[Metal4模型pass](metal4-machine-learning-passes.md)，它与本页所读compute encoder不是同一条接口。
