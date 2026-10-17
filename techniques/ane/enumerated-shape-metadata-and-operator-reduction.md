# 枚举shape失败的元数据对照与算子缩减

<!-- knowledge
id: technique-ane-enumerated-shape-metadata-and-operator-reduction
type: method
stage: ffn, pipeline
operator: conv, layout
backend: coreml, cpu-reference
device: cpu, ane
symptom: graph-interruption, measurement, precision-loss
semantics: numerically-equivalent, contract-only
representation: fp16, fp32
boundary: request, producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [前轮完整shape/hint实验](dynamic-shapes-hints-and-failure-boundaries.md) · [固定来源与相反的公开报告](sources.md#enum-reduction-sources-20261017) · 全部输入配方/返回值/失败/字节证书 · [验证（过程原件已移出）](../MAINTENANCE.md)

2026-10-17，继续使用coremltools8.3 `07ffd86005e385d8eddfd22072836cfb457f7350`、M4 Max / macOS26.6.2（25G83）、SDK26.5。新104个独立CPU_ONLY进程全部终止：80个成功、24个首次预测SIGTRAP。所有356份返回数组（1,826,816元素）通过独立完整参考；44个正常feature拒绝另计，不用成功前缀替代失败。

关键进展不是“枚举都坏了”，而是：**同一枚举接口下identity/cast和ReLU可以工作，乘法/自相加等本批表达却失败；完整旧FFN的源MIL程序和整个权重文件不变，只交换input feature的enum/range约束，也足以切换成功/失败。** 这定位了受控的表示边界，还没有定位某条内部BNNS指令或修复converter/runtime。没有性能、物理ANE或全模型资格。

## 可复用的方法：用单因素图缩减定位失败

1. 保留完整失败图、输入和质量 oracle，先只改 shape 元数据，避免同时改变数学和输入域。
2. 再逐步缩减算子、精度或广播表达；每次仍检查完整输出和原错误能否复现。
3. 规范化后的源码身份、序列化次序和编译产物分开记录；issue、相反评论及本机结果不互相覆盖。

适用于排查枚举 shape/算子组合边界；最小成功图不是完整旧图已经修复的证明。

## 1. 公开issue提供线索，也包含相反复现

本轮只读捕获[apple/coremltools #2730](https://github.com/apple/coremltools/issues/2730)及其全部一条评论，HTTP200、无下一页，API原始字节/hash留ignored目录。issue创建于2026-06-01，2026-10-17捕获时仍open。作者报告9.0、二维leading-batch形状的推理/stride异常，并给出乘2示例及自己的root-cause假设。

[2026-06-11的社区评论](https://github.com/apple/coremltools/issues/2730#issuecomment-4682113003)却报告同示例在其macOS26.5.1上通过，要求精确失败模型和系统身份，并提醒报错中的stack与所贴简单图不一致。API的author_association为NONE；这里不把它当维护者确认、已合并修复或已发布regression test。作者假设也不是已证实的根因。

本机是**8.3、四维trailing-S**，没有安装/执行作者的Torch2.11、9.0、Swift工程或TiRex模型。不能把不同版本、轴、图和“错误shape/异常/SIGTRAP”三类症状合并成同一个已复现bug。后面的结论来自自有受控图与原完整FFN，不来自issue标题、open状态或某个评论的自述。

尤其不要混淆两个`-5`：issue中的NSError/Espresso错误码与本机`subprocess.returncode=-5`不是同一字段；后者表示进程被SIGTRAP终止。相同数字不能建立两类故障的同一性，本批正常feature拒绝又是另一条返回NSError的路径。

## 2. 最强对照：完整旧图只改一个input oneof

前轮seed17的enum64与range64自有资产保持只读。本轮加载spec、不加载执行模型，双向CopyFrom替换一个input的multiarray约束，再保存到**新包/新compiled目录**。默认shape仍是`[1,64,1,64]`，IO仍FP16，输出描述不变。

```text
Model
├─ description(2).input(1).type(3).multiArrayType(5)
│  ├─ shape(1)、dataType(2)不变
│  └─ enumeratedShapes(21) ↔ shapeRange(31)
└─ mlProgram(502)整段不变
```

发布四份**自有模型protobuf状态的确定性序列化**无损packet。标准库有界wire reader沿上述明确字段路径，只剥去21/31并重建父message长度；余下整个模型状态逐字节一致，且502字段逐字节一致。它不猜测任意bytes是不是protobuf，也不依赖Core ML/protobuf包来离线验证证书。

两方向的源MIL hash均为`4b6bfb7535285068cf5d15990096049da3f220b94b45341b17a4a73ffefc27f9`，整个权重文件hash均为`51928f5539bc7c8c6fc5a3372bc8bf79be063d68c402d6799dd0337102b404cb`。权重是本库合成参数；不是checkpoint或上游源码。

阶段也要分清：本批原始program502证书不含`FlexibleShapeInformation`这个字符串，compile后的enum和range `model.mil`函数属性却都有它；40个模型中28个动态表示有、12个固定表示没有。通过的ReLU/identity枚举和完整range也含该属性。因此仅在compiled MIL搜到这个词，既不能诊断失败，也不能把源MIL字节证书与compiled MIL文件当成同一对象；本轮没有闭源compiler内部的插入/布局trace。

| 完整旧FFN表示 | 正反序新进程 | 所有共同合法shape的结果 |
| --- | --- | --- |
| 原enum64 | 两次首次预测SIGTRAP | 没有返回数组 |
| enum→range，新副本 | 两次成功 | S64/16/256/64完整输出均通过，S32也合法 |
| 原range64 | 两次成功 | 相同完整消费者通过 |
| range→enum，新副本 | 两次首次预测SIGTRAP | 没有返回数组 |

双向替换后的确定性模型状态**恰好等于另一baseline**。这是受控再生成，不是新的compiler修复。enum→range扩大了允许输入域，原enum并没有因此获得资格；range→enum即使只测试两者共同的合法shape也失败。

这项证据说明，在本批旧完整图/参数/环境中，输入flexibility表示的改变足以改变随后compile/load/predict路径的表现。它**不证明compiled code、内部布局或kernel仍相同**；相反，编译器本来就可能依据feature metadata进行specialization。原MIL/权重身份不能升级为机器代码或firmware身份，也不能单独把缺陷归给某个开源converter函数。

## 3. 图缩减与精度消融：不是只检查一个更容易通过的小图

另建四个明确标为诊断的图：identity、X×2、ReLU(X)、原数学完整64→128 ReLU→64+residual。每个各有固定64、有界range16..256及enum{16,64,256}三种feature；各有三种精度方案：

- `half`：FP16端口与显式FP16 core。
- `mixed`：FP32端口，进入core前cast FP16，输出cast回FP32。
- `float`：FP32端口、FP32参数和core。

converter的FLOAT32设置在这里用于**不自动插入降精度pass**；真正的core类型来自显式输入/参数/cast，再核对序列化conv/relu/add/mul输出类型10或11。不能像[旧精度研究](precision-islands-and-observer-effects.md)所警告的那样，仅凭precision policy的名字称图为真正FP32。identity的mixed版本实际保留两次cast，不是零操作假设。

四图×三精度×三feature共36模型，加上上节四个旧图/干预控制共40。正反case/visit顺序各一轮，每单元新进程；固定模型只喂S64，动态模型喂64→16→256→64，反序按canonical visit编号倒序。每次generation改变，默认点仍一致。

| enum诊断表达 | half | mixed | float |
| --- | --- | --- | --- |
| identity / typed casts | 完整输出通过 | 完整输出通过 | 完整输出通过 |
| ReLU | 完整输出通过 | 完整输出通过 | 完整输出通过 |
| X×2 | 首次预测SIGTRAP | 首次预测SIGTRAP | 首次预测SIGTRAP |
| 完整FFN | 首次预测SIGTRAP | 首次预测SIGTRAP | 首次预测SIGTRAP |

所有固定/range对应图均完成并通过原参考。通过的enum图仍会正常拒绝S32，说明不是通过删除枚举约束获得“成功”。因此“公共MLMultiArray的known strides让所有枚举模型都无法执行”不符合这些本机正例；“只要改成FP32/换端口就能修复”也不符合这些负例。

这些都是声明的MLProgram/端口类型，不是物理累加器精度证明。输入是有限的dyadic合成值，mixed模式的输入在half中恰好可表示；没有因此证明其对任意FP32输入与float模式数学等价。

## 4. 自操作数对照：源码层标量广播并非必要条件

再从已保存的真FP32乘2spec出发，保持feature与dtype不变，只改变一个operator/operand binding。每种fixed/range/enum均保留原乘2对照，并分别测试：

| 新source表达 | 修改 | enum结果，两轮 |
| --- | --- | --- |
| X+X | mul→add，y operand指向X | 首次预测SIGTRAP |
| X×X | y operand指向X | 首次预测SIGTRAP |
| X+2 | mul→add，保留原标量operand | 首次预测SIGTRAP |

固定/range全部返回正确完整数组。自操作数版本还保留一个**未使用的旧scalar const**，但y operand确实不再引用它；独立解析compiled MIL的add/mul表达也确认x/y均为X。不能写成“原图没有任何常量”。

这反驳的是“**serialized/compiled MIL中的标量广播操作数是本批失败必要条件**”。内部优化仍可能重写X+X、选择不同BNNS primitive；本轮没有内部kernel/ISA trace，不能据此提出“所有binary op都坏了”或某条硬件指令的解释。未过滤的文本token扫描会把`string/fp32`等literal也算进去，证据特意标为非kernel/op计数；实际operand结论来自具体表达，非这个计数。

原完整FFN始终保留为失败控制。新ReLU成功、自操作数图或metadata-range成功都不是原完整enum图的内部trace、真实模型修复或可采用加速方案。

## 5. 全部结果与分母，包含正常拒绝及SIGTRAP

| 本批实际观察 | 数量 |
| --- | ---: |
| 主矩阵/旧图干预单元 | 80 |
| binary operand/source控制单元 | 24 |
| 成功进程 / 首次预测SIGTRAP | 80 / 24 |
| 正常四visit返回数组 | 320 |
| 合法range S32返回数组 | 36 |
| 正常off-menu feature拒绝 | 44 |
| 返回数组 / 输出元素 | 356 / 1,826,816 |

合计424次预测API尝试：356个成功返回、44个feature拒绝、24个没有返回的SIGTRAP尝试；不是424个硬件dispatch。所有父级collector完成，不代表子模型全成功。每次load/predict的before/after event及typed input witness保存；失败末事件均为实际首次`predict-begin`，没有值、估计时间或0ms代填。

所有返回数组都对独立完整FP64参考重算，保留原逐元素`.002 + .01*abs(reference)`门槛，0个拒绝；所有非full的dyadic结果还与FP64参考逐值精确一致。全量结果用77个唯一typed、lossless packet去重，f16/f32与形状长度分别核对，不是只检cosine、一个scalar或argmax。52份实际完整weight/bias payload也按FP16/FP32 storage类型独立解码并匹配参考配方。

本轮24个trap中，仅取得7份与owned receipt PID精确匹配的OS crash报告，所采样本有BNNS frame；其余17份unavailable，不能假定相同完整stack。原始地址/二进制/模型、API正文留ignored目录，公开只含自有合成tensor、wire证书、身份、数值和无地址frame摘要。未调整全局日志设置、安全策略或其他进程，也未主动清理既有模型或全局系统缓存。

## 6. 本地代码、验证与仍未解决的问题

原输入/完整数学/typed codec/wire reader、、、portable tests。CPU入口不import Core ML/protobuf、不联网或读取系统crash目录：

默认case真正重放104个已采集单元、356完整数组和两项metadata证书，并执行“追加无关root wire字段必须拒绝”“真实保存值对错一代输入必须拒绝”的负控；后者本批4,096个元素均拒绝。它不重新执行104个原生进程，不只是返回固定的说明性布尔值。

显式执行：prepare/全部主单元监督、公共API typed原生客户端、operand-only补充。只使用已有8.3/NumPy与新owned目录，最少2GiB reserve，无模型下载或包安装。原报告/资产/旧失败未覆写，后续发现不回填为前轮当时已经知道。

这里的新目录指显式输出；Core ML/coremltools自行创建的全新临时包按其正常生命周期回收，不代表所有框架临时文件都位于输出目录。没有主动清理既有模型或全局系统缓存。

随后补强默认CPU重放case，数值证据仍逐字节不变，只更新验证器/测试的source binding。期间另两份隔离快照分别暴露新Metal query索引不一致和新GPU/ANE语音文档的README登记未完成，失败照存；文件hash稳定、索引自洽也不等于全部内容合同通过。最终取源前后与副本1,459文件hash相等、未删页/未改复制内容的完整快照，明确等其生成完毕后再隔离，2,298项（2 skips）及全部六个命令通过。该固定快照不是共享工作树以后仍然通过的保证。

可学习的接入设计是**语义图去重与compiled artifact身份分开**：本例的MIL+weights hash相同，feature约束却不同，而且共同合法点的执行表现也不同。只用前两者作compiled-cache/资格key不足；完整feature ABI、配置和runtime环境需另绑定。这是从受控模型得到的设计反例，未声称已在某个现有项目复现错误cache命中。若把range作为候选替代，caller仍应显式维护原菜单并重新验收完整消费者，而非把原enum标成已修复。

能力探针宜放在独立owned进程；SIGTRAP不由NSError或Python `try/except`正常返回。确认进程终止后才能重新接收整个请求，用原输入完整回退，不能拼接未完成值。本例无MLState，不能据此授予真实KV/训练状态的回滚或恢复资格；本轮没有修改产品fallback策略。

继续问题包括：二维leading-batch与四维trailing-S的匹配比较、rank/layout/输出接口及算子组合的进一步缩减、跨converter/OS版本的精确模型对照，以及实际首选ANE的代表性消费者。**本轮只做一个seed17的受控诊断、两种执行顺序和CPU_ONLY**；不继承旧报告的其他seed/ANE/hint资格，也不声称已找全根因、已修复公开issue、取得生产速度或完成整体ANE目标。
