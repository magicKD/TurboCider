# Apple ANE Transformers：迁移布局，不替换模型数学

<!-- knowledge
id: source-upstream-ane-transformers-ffn
type: source
-->

[来源目录](../README.md) · [图合同](../../ane/coreml-graphs.md) · [阶段地图](../../models/stage-map.md)

来源：`apple-aiml-research/ml-ane-transformers@590eb35913622cdef217ebc5329047aab389ac98`，`ane_transformers/reference/ffn.py`全文；SHA-256 `f0d7d1bb7189e3607c069918f656054c875563ffb6c729fd6ac27df32509d7a8`，工作树干净，2026-10-15。

此文件的FFN是1×1 `Conv2d`→ReLU→可选dropout→1×1 Conv2d；`ResidualFFN`和PreNorm变体调整残差/LayerNorm所在位置。它不是Qwen Image的配对gate/up SwiGLU，也不是证明所有二维matmul都会被ANE编译器同样lower。

可迁移的是：用明确的channel/spatial维表达线性算子，保留producer/consumer可复用布局，比较转换次数与完整边界；不可迁移的是：为了套上旧图而换激活、删分支、换norm位置，或把训练dropout当作无差别推理步骤。

本地`layout=channels`样本使用`[1,1,C,M]`的4D matmul，不是本文件的Conv2d逐行移植，也不是本库frozen图`[1,C,1,M]`的同一ABI。只有shape标签一致而实际stride不一致仍会错；实验显式转置/恢复并计时。

本轮状态：`source-inspected-one-file / upstream-not-run`。后续仍需真实1×1 Conv与matmul的等权重消融，不把channel-major小样本通过泛化为所有Transformer布局。

许可是Apple sample-code条款，见LICENSE.md与ACKNOWLEDGEMENTS，不标成MIT；本页仅原创说明，没有复制原实现。

[固定版本文件](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py) · [Apple研究文章](https://machinelearning.apple.com/research/neural-engine-transformers)。
