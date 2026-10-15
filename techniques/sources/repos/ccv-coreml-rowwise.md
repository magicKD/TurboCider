# ccv：图内K分块、surface缓存与output backing

[来源目录](../README.md) · [综合图合同](../../ane/coreml-graphs.md) · [本地实验](../../labs/ane/tiling_bench.py)

来源：`liuliu/ccv@66dc16a3a3b688adfa6dc1d19eeeb3071beeb701`；`lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm`，阅读1–210、630–705、1030–1245行。文件SHA-256为`626a83d608a80b5428a62ab74b8c5506726718ca78a25dd74bc803e0183884b4`，工作树干净，2026-10-15。具体范围和许可也登记在[清单](../../references/sources.json)。

## 源码事实：阈值是候选，不是硬件定律

`select_matmul_k_split`在M≥512才启用K拆分，按M选择3072、2560或2688的direct-K门槛，再以整除条件选择1、2、4块。不能将这些值改写为所有ANE的通用SRAM容量或性能上限。

`generate_dynamic_i8_two_input_matmul_scaled_mil`在一个图中切weight和activation对应的K区间，各自dequantize，然后matmul并相加。4块使用平衡求和。这个表达是INT8接口→dequantize→浮点MIL计算，源码本身不能证明硬件INT8 MAC。

缓存额度区分activation、weight、output：512 MiB、128 MiB、1 GiB；scratch另有16 KiB对齐。它们是此实现的资源政策，不是硬件缓存容量；移植时还要检查stride、generation和并发使用寿命。

## 消费者合同：请求了backing不等于确认采用

`ccv_nnc_mfa_ane_rowwise_coreml_evaluate`从pixel buffer创建MLMultiArray，条件允许时设置output backings，再获取prediction实际返回的multiarray。已读路径检查返回pointer非空和总字节数；未看到它核对返回pointer等于预定surface backing，也未看到不同时将实际结果复制回scratch。

因此存在一个需要实测的条件风险：若某OS/图忽略output backing，后续直接消费预定scratch可能读到旧数据。**本轮未运行该上游bridge，不能把这个条件分析称为已复现的ccv bug。** 也不能只凭count相等证明physical stride相等。

本地实验读取实际返回数组并显式restore，承认Python桥有copy；不宣称零拷贝。真正消除restore之前，必须验证backing身份、pitch、owner、完成事件和拒绝/回退路径。

## 本地吸收与证据边界

[tiling_graphs.py](../../labs/ane/tiling_graphs.py)是原创公共MIL实验：配对fused gate/up、图内K/N分块、平衡求和、完整down；不是上述INT8 emitter的复制或逐行移植。FP16实验不能替上游W8A8或共享surface取得资格。

错误对照：把SiLU移到每个K partial再相加会改变函数；把每个K tile变成一次host prediction会改变调度粒度。后者需要另测，不能把图内tiling收益直接迁移给它。

重新探索条件：SoC/OS/编译器、矩阵shape、输入格式、tile、实际backing采用和完整consumer边界变化。上游实现本轮状态：`source-inspected-selected-ranges / upstream-not-run`。

选定lib源码按根COPYING的BSD-3-Clause分类；doc/samples/site另为CC-BY-4.0，内含第三方另有条款。本页为原创概括，没有分发上游源码片段。

[固定版本源码](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm)。
