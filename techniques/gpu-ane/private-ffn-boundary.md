# 私有 FFN 分担：完整 GPU 分母、精度岛与交接成本

[总目录](../README.md) · [切分代数](partitioning.md) · [ANE图搜索](../ane/graph-rewrite-search.md) · [全部本机结果](../review/qwen-private-ffn-20261016.md)

适用：已有较强 GPU FFN、另一路设备能处理合法channel子集，且并发时间足以隐藏慢分支。这里是实验方法，不是新生产runtime。数学切分可等价；实际BF16→FP16、累加顺序与join变化属于需明确数值预算的精度配方，非bitwise/无损承诺。

## 1. 冻结完整消费者，再选择分担比例

本轮真实 Qwen layer0为 `[1,1024,4096]` BF16输入，intermediate width12288。完整GPU分母保留所有gate/up/down和原始BF16权重；ANE只接配对的前F0个gate/up及对应down列，GPU执行剩余width，最终恢复完整BF16输出。没有少token/step、重复toy行或把一个小matmul称整个模型。

`channel_slices` 的[核心参考](../labs/silu_domain.py)使gate、up和down列同源；上游 fused `gate_up` 的up起点是原始F，不是F0。当前base FFN没有在本实验中新增bias/residual；真实模型有bias、LoRA或不同modulation时必须按原合同处理，不能直接套本图。

## 2. 稳定 SiLU 是图表达，不是全局精度资格

令 `e=exp(-abs(g))`，使用 `select(g>=0, g, g*e)/(1+e)`。指数参数非正，在实数/F64参考中避免对大负g计算exp(-g)溢出；[CPU domain示例](../labs/silu_domain.py)不调用设备。后端的FP16 exp、select、除法和乘法仍可能改变误差，必须独立实测，不能因公式稳定就授予硬件资格。

独立[Qwen FP64 witness](../labs/ane/qwen_witness.py)只读原始BF16源的数值，不导入MLX/ANE；选4个实际row、4个输出column，但每个row使用完整K/F。实际GPU baseline全输出对保存的BF16 replica必须exact，候选还须同时过完整source、独立full sample、完整prefix BF16 reference和独立prefix sample。门槛固定relative L2≤0.03、cosine≥0.999，所有first/warm/timed调用均检查。

## 3. 三个arm和一个完整边界

| Arm | 工作 | 用途 |
| --- | --- | --- |
| full GPU | 原始完整BF16 fused FFN | 必须赢过的分母 |
| serial | GPU→host FP16；private prefix；完整GPU remainder；FP32 join→BF16 | 相同图和copy路径的串行控制 |
| hybrid | 相同输入转换后，一个owned worker执行private prefix，主线程执行GPU remainder，最后join | 比较并发能否隐藏慢分支 |

边界为**prepared真实BF16 GPU输入 → 完整BF16 GPU输出ready，显式MLX drain**。GPU cast/readback、Model输入copy/同步/输出FP32副本、restore/upload、FP32 join和BF16恢复都计入；共同输入准备、常量权重prepare/compile/load和数值验证另列。输入值不变，但每group创建新资源，避免把同一lazy结果缓存误当重复执行。

这是明确copy-based bridge，**不是IOSurface共享或零拷贝**。每call逻辑上有8MiB输入FP16和16MiB输出FP32跨接口，内部还有wrapper复制；这些是表示字节数，不是DRAM PMU流量。Py调用的重叠区间只证明host API schedule，不能代替物理ANE placement/overlap。

## 4. 为什么慢分支也可能值得分担

完整边界近似为 `cast/readback + max(private_call_concurrent, gpu_remainder_concurrent) + restore/join`，各项应从同一运行读取，不拼接不同实验。

本机2048-channel长确认中，hybrid的phase中位约1.07ms输入转换、11.44ms private call、17.68ms GPU remainder、0.55ms join；完整约19.42ms，full GPU约21.75ms。中位数不可严格相加，这只是关键路径解释。serial约31.36ms，因此serial→hybrid约1.61×并不是最终分数；full GPU→hybrid约1.12×才对应本边界的收益。

512channels没有覆盖交接税，1024只有约3.4%探索性收益，2048在固定小grid中最好；并不说明更多分担一定更快。两设备争抢统一内存、不同GEMM形状、launch/restore和新输入分布都可能改变最优点。

## 5. 冷启动、驻留与完整请求仍能否决它

本轮全部arm共存的GPU逻辑权重约576MiB（含full baseline、remainder和参考prefix），其中相对full-only多约288MiB；2048channels另有48MiB private逻辑权重。没有独立测每arm的系统/设备峰值或内存受限机器，不能当低内存产品配方。

同一次长确认的native emit/compile/load约682ms，warm每次FFN节省约2.33ms。若机械地把这笔新增setup全部计给候选、其他都不变，需约293次调用才能抵消；这只是派生算式，不是实测请求break-even。系统cache未清，其他层权重是不同常量，不能把32层都当同一图复用。预编译、load、按层bank、LoRA失效和全请求内存/质量都须重新验证。

## 实现参考与复现

核心学习代码：[domain/合法切片](../labs/silu_domain.py)、[独立数学witness](../labs/ane/qwen_witness.py)、[只读source/gate/统计verifier](../scripts/curate_qwen_private_ffn.py)、[离线负控](../tests/test_qwen_private_ffn.py)。显式[upstream integration](../integrations/README.md)需要已审核的ANEForge副本和现有私有模型/捕获，不是核心或默认查询依赖。

```sh
python3 -B techniques/labs/run.py silu-domain
```

本机[完整报告](../review/qwen-private-ffn-20261016.md)保存三档discovery、冻结2048后的step19/10确认、24-group确认与frontend失败；不同process时钟不合并。它不是完整Transformer block、生成请求、LoRA或产品默认资格。

下一步：真实GPU/ANE input/output backing、实际stride/alias与owner见证，减少copy后重测；更广层/prompt/seed/LoRA和完整请求；compile/load复用与内存/长时回归。不能只把 `zero_copy` 字段改成true，也不能相乘toy ANE和这里的收益。

2026-10-16后续[合成actual-backing实验](shared-surface.md)已验证request surface与Metal映射及完整GPU消费者；它是接真实FFN前的一个前提，不回填本页copy-bridge资格。真实二维MIL、MLX adoption和full-source精度/成本仍需重新实测。

## 6. 后续实际shared完整FFN：新证据，不给旧桥接换标签

2026-10-16后续已在[真实五臂实验](../review/qwen-shared-ffn-20261016.md)实连：实际2D MIL port是`[1,1,1024,4096]` FP16，GPU从原BF16 input pack，request output实际surface导入MLX，作为compiled join的runtime参数。完整GPU/copy serial/copy hybrid/shared serial/shared hybrid同source配对，560次调用全部过full/prefix/独立FP64固定门槛。

后两run另有264个完整实际tail/private部分的独立CPU FP32加法＋BF16 RNE位级见证，64项旧实际point输出拒绝观察；不能把位级join正确当整个mixed-precision模型无损。新[纯CPU合同](../labs/real_shared_ffn.py)、[data verifier](../scripts/curate_real_shared_ffn.py)与[离线反例](../tests/test_real_shared_ffn.py)随本目录分发；设备入口属于显式[integration](../integrations/README.md)。

长确认完整GPU约21.6ms、shared-hybrid约18.8–18.9ms，paired约1.146–1.149×；同client copy→shared只有约1.1%观察且首轮三point方向相反。copy控制使用FP16 staging及同一private client，不等于本页旧e5rt/NumPy bridge；不把两批差值全归因于copy。

setup仍能否决：load-only只省掉客户端compile selector，不是内部无编译；实测load650.99ms与32.11ms都保留。新clone/compile与只读复用各自admission，未执行的存储拒绝不是硬件负结果；[文件clone](../sources/official/clonefile.md)也不是设备zero-copy。完整请求/LoRA、更多层/输入、内存/长时/失败取消和physical placement/overlap仍未获得资格。

```sh
python3 -B techniques/labs/run.py real-shared-join
```

## 7. 完整ordinary block：执行、数值和交付是不同状态

[独立seed/prompt的后续记录](../review/qwen-block-context-20261016.md)在实际ordinary GPU observer内比较原始compiled block callable、staged GPU、copy hybrid、shared serial和shared hybrid。attention/norm producer、完整FFN、modulation/residual consumer均在计时边界内；staged GPU对原始输出exact，作为分段本身的ablation。它是借用原参数的完整block重放，不是完整请求或未改变内部FFN的直接capture。

两份记录完成330次完整调用和198个exact actual-part join，但最后的原始tensor/无损archive导出都被存储reserve拒绝，因此协议仍failed、comparisons为null，只保留绝对耗时诊断，不授予新倍率。另一份load返回false且NSError为nil，不能凭error code0判定成功，也不能由TMPDIR上下文差异推断因果。

[纯CPU合同](../labs/block_context.py)、[只读curator](../scripts/curate_block_context.py)与[离线回归](../tests/test_block_context.py)分开核对execution/numeric/artifact/promotion轴。小矩阵C++ oracle和合成archive roundtrip测试不替代真实模型raw持久化或请求质量验证。

```sh
python3 -B techniques/labs/run.py block-evidence-axes
```
