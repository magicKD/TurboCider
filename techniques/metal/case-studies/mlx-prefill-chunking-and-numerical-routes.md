# 原始MLX-LM prefill分块：step size改变算术路线，不只是内存和吞吐参数

<!-- knowledge
id: technique-metal-case-studies-mlx-prefill-chunking-and-numerical-routes
type: case-study
stage: prefill, decode, pipeline
operator: attention, gemm, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, measurement
semantics: approximate, contract-only
representation: bf16, fp32
boundary: weight-runtime, producer-consumer, request, cpu-gpu
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [GQA/score精度边界](mlx-sdpa-gqa-fallback-and-score-precision.md) · [KV来源与lookahead](mlx-prompt-cache-provenance-and-lookahead.md) · [固定来源](../sources.md#mlx-prefill-schedule-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十二轮。直接调用未修改的`mlx-lm@93e6d4a6447b3080861985947f0820e76fcb957d`原始`generate_step`，不是手写分块forward冒充该API。复用同一官方BF16 Qwen完整24层/290参数；MLX0.32.3、M4 Max。四个既有文章的7/32-token前缀，step1/4/5/8/9/2048，BF16/显式FP32权重提升控制×两重复。

关键结果：分块会改变query/GQA准入及其他shape相关舍入。7-token时step1/4明显更接近原数学参考，32-token时默认31+1已很好，切成step5/8反而更偏。**没有“越小越正确”的通用规则，也没有可采用的精度修复/提速资格。** FP32全部288个raw向量过原门槛，BF16对应288个全拒绝。

## 1. 原API先预留最后一个prompt token

读[`generate_step`](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/generate.py)：只要剩余prompt长度>1，就处理`min(prefill_step_size, remaining-1)`；每块eval原KV状态、调用进度callback和`mx.clear_cache()`。最后一个prompt token进入生成用的`_step`，只取最后logits，计算原logprobs，再采样。

因此step size是上限，不是实际提交query数。prefill期间`_model_call`返回值被丢弃、只eval KV；不能把每次Python模型调用都按实际执行了完整词表head来计工作量。没有pipeline trace时不宣称每个底层kernel的具体数量。`clear_cache`调用也没有重置这些KV对象，本实验有效offset一直递增。

| prompt长度 / step | 实际prompt调用query长度，含最后1 | 本例D64/GQA7的源码准入 |
| --- | --- | --- |
| 7 / 2048、8、9 | 6+1 | 6是fallback，1是vector候选 |
| 7 / 4 | 4+2+1 | 全vector候选 |
| 7 / 5 | 5+1+1 | 5是fallback |
| 32 / 2048 | 31+1 | full候选+vector候选 |
| 32 / 8 | 8+8+8+7+1 | 前四块fallback，末1是vector候选 |
| 32 / 9 | 9+9+9+4+1 | full候选×3、vector候选×2 |
| 任一 / 1 | 每次1 | 全vector候选，但模型调用更多 |

这是[MLX0.32.3完整选定guard](mlx-sdpa-gqa-fallback-and-score-precision.md)在D64、inference、Q14/KV2下的预测：query≤8还需query×7≤32。不是所有模型/训练/sinks的完整selector，不是intrinsic硬件极限。原生pass-through observer确实取得全部24层的Q/KV形状，共39936次Python attention调用；**形状见证+源码准入仍不是逐PSO/ISA/counter取证**。

## 2. 为什么使用固定真实续文，而不是各自greedy跑开

若不同分块先采出不同token，后续比较会同时混入输入变化。这里sampler明确返回既有文章的固定后续token，保持每一路实际prefix一致。原API仍真实计算完整logprobs、更新KV、保留lookahead；只是采样策略被显式设为teacher forcing，不称greedy质量测试。

4文章×2长度×6step×4进程=192组observed/unobserved配对，实际调用384个原generator。每个observed产生3个完整raw行，两个variant各返回2个完整API logprobs，总计1344个151936维向量：576raw、768logprobs。两个返回token为何有3个raw见[下一页](mlx-prompt-cache-provenance-and-lookahead.md)。原始独立24层FP64-reduction/FP32-stage oracle重新算24个独特prefix，不用MLX CPU forward或另一原生引擎作gold。

样本是此前8文章中的固定前4篇、7/32上下文、每处3个真实target；是后验机制研究，不是新独立held-out任务。24个数学目标、重复进程和192配对不能冒充192个独立文章。未使用chat template/BOS/新生成token替换原文目标，也不声称完整WikiText PPL。

## 3. 完整输出结果：短prompt改善，不是全局单调

下表为BF16、每个cell对4文章等权平均，**仅rep17统计**；rep29只作独立反序重复，672个完整向量身份组全部逐字节相同。KL为原参考→实际的完整151936支持，不是top-k近似；后续index1/2对应追加相同真实token后的prefix。

| 长度 / step | prompt末行平均KL | 再消费1个真实token | 再消费2个真实token |
| --- | --- | --- | --- |
| 7 / 1 | 0.00155910 | 0.00105043 | 0.00127141 |
| 7 / 4 | 0.00193561 | 0.00303404 | 0.00287348 |
| 7 / 5 | 0.03777230 | 0.01698925 | 0.00947520 |
| 7 / 8、9、2048 | 0.03962408 | 0.02671408 | 0.01034368 |
| 32 / 1 | 0.00040855 | 0.00094518 | 0.00073661 |
| 32 / 4 | 0.00049446 | 0.00031625 | 0.00031251 |
| 32 / 5 | 0.00175417 | 0.00046889 | 0.00116514 |
| 32 / 8 | 0.00194458 | 0.00104302 | 0.00081992 |
| 32 / 9 | 0.00082423 | 0.00030611 | 0.00054104 |
| 32 / 2048 | 0.00039906 | 0.00040564 | 0.00030110 |

7-token的8/9/2048调用序列相同，所有raw、原API logprobs和所选完整KV也都精确相同，这是正控。32-token时不能照搬短prompt结论：step8反复走fallback；step9虽走full，也不保证每个文章/后续位置优于默认。原证据保留72个完整summary cell及每文章结果，不只展示改善的case。

FP32的288个raw全过`.005+.005*abs(ref)`且relative-L2≤.005；最大KL约4.89e-10。BF16的288个raw全拒绝同一门槛，不因平均KL下降、top1相同或重复性而晋升。FP32提升只改变原BF16权重的表示，不换成另一个官方FP32模型或重新量化。

原文target NLL也没有与KL同方向保证：L7、index1，默认平均NLL相对参考约**-0.36317**，step1约-0.03881；默认离参考更远，却在这4个真实target上损失更低。不能把“更像数学参考”直接叫“任务更好”，也不能从微型样本反向宣布低精度更优。

## 4. 控制与归因边界

observed只增加原函数的pass-through attention wrapper、返回原logits的identity processor和完整CPU读回；unobserved不装这两者。每组最终48个有效K/V的native全量hash相同，返回的两份完整logprobs逐位相同，函数finally恢复。doc0另外实际保存全部有效KV bytes并由CPU逐段重验；其他文章只有native完整KV hash，不能写成全部都进行了独立raw回放。

这些控制支持观察数值保值，不代表无同步/无allocator/时序扰动：原API有独立generation stream；v2在释放/切换variant前显式同步该stream。`mx.synchronize()`默认只同步默认stream，不能因为调用了同名函数就假设已等待generation stream，见[固定stream源码](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/python/src/stream.cpp)。未记录正式速度、没有qualified倍率。

上一轮同实际QKV重放已支持低dtype score图的主要机制；本轮分块还改变QKV投影M、norm/RoPE和求和顺序。doc0某些step在**第0层K/V**就与默认不同，而K/V在本层attention之前已写入，故不能把所有分块差异都归给attention或LM head。依赖边界详见[KV页](mlx-prompt-cache-provenance-and-lookahead.md)。

## 5. 可执行学习与复查

CPU合同、tests仅用标准库，不导入MLX/模型。在显式原件上以已有NumPy独立计算，不执行GPU；原API客户端要显式新目录、固定资产、supervisor和串行GPU准入。发布器只作无损JSON格式压缩，raw/portable SHA分开；不发布真实词表向量/KV或模型权重。

复查条件：source/wheel/native SHA、SoC/OS、模型/权重、dtype、GQA/mask、prompt长度、step、缓存位置/格式、sampler/processor/stream或消费者改变。下一步需更长context、真实greedy/serving及包含source cache-eval/clear-cache/同步成本的可信性能控制；没有为更容易通过预算而改门槛，也不把step1/4设为生产默认。
