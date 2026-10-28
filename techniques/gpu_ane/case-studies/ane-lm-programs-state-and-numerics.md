# ANE-LM：模型级CPU/ANE分工、状态与数值合同

<!-- knowledge
id: technique-gpu-ane-case-studies-ane-lm-programs-state-and-numerics
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [本轮摘要](../../MAINTENANCE.md) · [来源/行范围](../sources.md#ane-lm-state) · 数学见证 · 计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17。路线图N10此前只是项目入口，本轮追到Qwen3/Qwen3.5实际forward、私有ANE包装、CPU attention/SSM、权重转换和generation消费者。该ref是**CPU＋私有ANE**，不是GPU＋ANE并行，也不是“整个LLM都在ANE”。原生C++、BLAS、私有接口、Torch参考及完整模型均未执行；实跑的是独立标准库数学见证，不能当原实现输出或模型质量测量。

最需先验证的不是速度：Qwen3.5所调用的`ssm_step`在衰减前预测旧state，而固定Transformers参考在衰减后预测。两种数学式从零state第一步可一致、第二步已能不同；仅测单步零初态或ANE投影status会漏掉这层消费者差异。

## 1. 身份与构建边界

固定[`johnmai-dev/ANE-LM@04b1af12aa30b3f280f511d286398469764b4326`](https://github.com/johnmai-dev/ANE-LM/tree/04b1af12aa30b3f280f511d286398469764b4326)，tree `f836d243f1c8146f6ccaad87c9fc91a5ba673214`，commit日期2026-03-04。25份选定文本按原Git blob核验并记录阅读范围，不称全仓库/全部vendor审计。tree无AGENTS；MIT源码/元数据只留ignored `outputs/gpu-ane-ane-lm-round36-20261017`，不分发模型或原源文件。

参考为[`huggingface/transformers@8498ed425fa01bbee7d086efdf1a6634c2962aed`](https://github.com/huggingface/transformers/blob/8498ed425fa01bbee7d086efdf1a6634c2962aed/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L424)，日期2026-10-16；只选读文件头和`torch_recurrent_gated_delta_rule:424–483`，不是完整Transformers审计或Qwen原始checkpoint验证。generated文件未改，root AGENTS link及`.ai/AGENTS.md`正文已读，Apache-2.0许可保留于raw。

ANE-LM的CMake会FetchContent：json为v3.11.3归档URL但无URL_HASH；tokenizers-cpp为shallow `main`，不是完整commit锁。因此固定父repo不等于整个未来binary可重现。本轮不解析/安装这些依赖，不执行cmake、convert、模型下载或private framework初始化；本机项目venv没有torch，也没有为参考数学安装它。

参考获取有一次保留的失败：GitHub Contents API对AGENTS提供的内容与所报Git blob不匹配，本库hash gate拒绝。随后按同一固定ref的canonical Git blob取得symlink文本`.ai/AGENTS.md`，再读取目标正文。是下载器对link身份的处理修订，不是上游文件损坏、原生测试失败或模型故障。

## 2. 实际分工：大投影封在ANE，attention/state仍跨CPU

`utils.cpp`按model_type选择Qwen3或Qwen3.5，再设置cache偏好并load。Qwen3每层融合Q/K/V投影为一次ANE program；CPU做Q/K norm、RoPE、KV写入和GQA，随后ANE做O projection；CPU residual/norm后，再用一个ANE program保留完整gate/up→sigmoid/multiply→down FFN，最后CPU residual。不是将attention score/softmax或整层全部放到ANE。

Qwen3.5对full-attention层保留上述结构，另有Q/output-gate排列、CPU gate。linear-attention层则用ANE融合QKV/Z投影，CPU做小A/B投影、causal conv、SiLU、Q/K归一化、逐value-head SSM和gated norm，再回ANE O projection。代码中的“hybrid”模型结构不代表设备并行；这里有明确前后数据依赖，也没有Metal queue或GPU消费者。

每层正常主干有3次ANE请求：first projection、O projection、fused FFN；头部若ANE启用，再按最多16384个vocab row分chunk。这个数是程序边界推导，不是hardware dispatch trace。融合FFN避免大intermediate回host，但每个主干边界仍有FP32 host→FP16 surface→FP32 host转换，不能从“图内融合”推零交接开销。

`generate.cpp`的prefill也是逐token `forward(token,i)`，不是矩阵批量prefill。`ANE_SPATIAL=32`是本ref构图/IO占位约定：MIL使用`[1,C,1,32]`，host每channel只写/read第0lane，其他lane在surface创建时清零。不能把32解释成实际批了32个token，也不能把header的hardware注释推广为所有ANE/SDK的最小尺寸定律；真实编译port/pitch未探测。

## 3. graph、weight blob与持久marker不是同一个cache资格

私有包装以ObjC runtime C API找`_ANEInMemoryModelDescriptor`等class；MIL用1×1 conv表达投影，融合二/三投影后channel concat，FFN保留完整非线性子图。常量通过带128-byte头部的FP16 blob输入，不是INT8、runtime-weight或LoRA更新图。MIL中写死的buildInfo字符串也不是本机编译器版本测量。

`ane_compile_raw`先创建descriptor/model，向私有对象询问`hexStringIdentifier`作目录/marker身份，缺失则用UUID。host没有自己证明这个opaque identifier包含哪些graph、权重、OS/compiler内容，不能据名字授予完整内容寻址资格。marker只是`.ok`存在位：有marker先尝试load，失败删marker再compile/load；它不会把marker存在直接当成功。

重要的是即使命中marker，所读入口仍先写MIL/weight文件。compile、load、写文件、进程resident kernel与模型数值资格必须分账；全局compile/cache计数也不是某个layer纯净的耗时或本次加载工作量。

另一条预转换cache仅通过`ane_weights`目录存在选择blob模式，所读路径不校验checkpoint hash/转换recipe manifest；blob读取也不构成shape/dtype/短读的全面验证。即使源safetensors内容改变，同目录旧blob不应自动继承质量资格。本轮没有制造真实stale artifact或磁盘故障，不称已复现错误缓存。

LM head即便blob模式仍可回到原BF16指针按row切块重新构造常量，不能把convert描述成“不再读原权重”。Qwen3对tied/untied LM head分别选权重；Qwen3.5所读头部始终用embedding表，需限定到该实现实际支持的模型recipe，不将配置字段存在视为全部untied变体已正确。

## 4. IOSurface调用合同与失败边界

每kernel持有独立输入/输出IOSurface和request；source将surface组织为byte-oriented的一维容量，清零实际alloc size，再按`c×32`索引写FP16。`ane_matvec`锁输入、写lane0、解锁，调用evaluate并检查BOOL；成功后锁输出取lane0转FP32。这里没有公共Metal buffer事件、GPU producer或真实返回layout观察，故不能授予physical zero-copy或GPU/ANE overlap资格。

API调用方式是顺序evaluate/读回；本轮未执行私有方法，不能独立证明其全部完成/超时/失败语义。free路径调用unload后释放surface，但没有观察设备drain。部分初始化失败清理及文件I/O错误处理需要单独fault验证；源码风险与设备故障复现保持区分。

主干ANE不可用或first/O/FFN失败会拒绝load/forward，不提供整模型CPU fallback。LM head则是另一种可恢复边界：compile失败禁用ANE head；执行chunk失败时释放head kernels，再CPU重算**整个head**，覆盖已写的部分logits，后续继续CPU head。不能把此局部回退扩大为全部forward可事务重试。

原因是CPU KV、conv和SSM早已可能更新，而后续O/FFN才失败；外层`stream_generate`在null logits时打印错误并直接返回，没有在这个点恢复所有layer state或保证最终end callback。聊天入口每轮明确reset并重新prefill完整历史。下次重试、取消与并发请求不能只回退pos：Qwen3.5的conv_state/conv_pos/SSM state不由token position参数重建。

KV也是固定2048容量的CPU ring，满时覆盖最旧slot，attention按两段ring顺序读取。较大的RoPE表/模型max position不等于保留同样长的完整上下文；源码的局部窗口行为需要与目标checkpoint attention合同核对，本轮不作长上下文质量结论。

## 5. 递归顺序差异：投影正确不足以证明消费者正确

把单head状态写成`S∈R^(K×V)`，当前已准备好的向量为q/k/v，`α=decay`、`β=beta`。ANE-LM `cpu_ops.cpp::ssm_step`先算`Sᵀk`和delta，再缩放state；Qwen3.5调用者已把A_log取exp并计算`α=exp(−A·softplus(a+dt_bias))`，因此不是遗漏了另一个host衰减步骤。

两条来源在相同局部输入下对应：

```text
ANE-LM顺序： S_A = αS + β k (v − Sᵀk)ᵀ
参考顺序：   S_R = αS + β k (v − αSᵀk)ᵀ
相差：      S_A − S_R = −β(1−α) k kᵀS
输出：      y = S_newᵀq
```

这不是FP16舍入或private ANE accumulator误差，而是源操作顺序在实数域已经不同。参考内部会把Q除以sqrt(K)、可选归一化；上述比较从这些预处理之后的局部输入开始，不把本库的矩阵fixture冒充完整Qwen attention。最小K=V=1、q=k=1时不需要隐藏的维度缩放。

从S=0开始，两步分别v=2、v=0，均α=β=1/2：第一步两路S/y=1；第二步ANE-LM顺序y=0，参考顺序y=1/4。独立有理数脚本另检查162组局部公式，36组有差，其余含零state、α=1或β=0等不区分控制；每个矩阵元素完整符合上式。

**执行身份严格限定：**本库公式实现是独立Python数学重述，没有编译原C++、调用Accelerate、导入Torch或原样执行参考函数。故这是源码公式不一致及其可复算反例，不是原生运行故障、真实checkpoint偏差或已验证修复。后续应先在CPU原函数与同recipe参考上做多步回归，再测整个模型；不直接改上游。

## 6. 半精度：权重软件helper与arm64输入cast分开

权重blob构造和convert都会走自有BF16→F32→software F16 helper；它对mantissa截断，对NaN也走exp溢出分支变inf。实时arm64 activation输入则选`__fp16` cast，非arm64才走software helper。不能把两条路径统称为完全相同的“FP16转换”，也不能没有编译器实测就给native cast分配通用舍入/NaN资格。

本库用与C++位移写法独立的frexp/数学重述，对65536个BF16 pattern产生完整half-code表，并与Python标准库IEEE half转换比较：65280个有限输入中1024个half code不同，均为两边有限的舍入差；254个NaN在source公式中全变inf；2个inf另列。有限BF16的普通可表示正常值不能由此称为普遍有误，差异集中在半精度subnormal/舍入边界。

另一个**非BF16**的F32例子`0x3f801001`：截断公式为half `0x3c00`，IEEE reference为`0x3c01`。两份完整65536-entry binary表及162组有理数记录在独立新进程全部字节复现；它们都是本库数学输出，不是上游helper/设备的实测值，更不是模型误差预算。NaN输入诊断也不表示真实权重含NaN。

## 7. 怎样迁移这份实现的经验

可以借鉴的是大投影合并、完整FFN保留在子图、LM head按row分块、失败后重算全部局部输出，以及CPU状态/ANE纯函数图分离。不能机械继承32宽、16384-row、2048 ring、opaque cache key、转换策略或private版本要求。

采用前至少补：固定全部依赖/权重/转换recipe；原始CPU多步递归与半精度回归；真实compiled port及完整值；每个失败点后的状态回收；原始GPU强基线；完整prefill/decode/质量、resident/峰值、冷/热和能耗。INT8、M5 TensorOps、GPU并行不在本ref已证实能力中。

本轮空间仍未恢复原10GiB大动作储备，只进行少量文本获取、文档和标准库数学/隔离检查：raw≤16MiB、下载≤1MiB、curated JSON≤2MiB、CPU snapshot≤64MiB、空闲≥2GiB。没有GPU/ANE提交、native build、依赖安装、模型或系统/参考库改动，不清用户文件/cache。20项新测试及隔离范围见[验证（过程原件已移出）](../../MAINTENANCE.md)，整体研究保持active。
