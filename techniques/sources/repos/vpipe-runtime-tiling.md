# vpipe：图内K/N tiling与runtime权重布局

<!-- knowledge
id: source-upstream-vpipe-emitter
type: source
-->

[来源目录](../README.md) · [既有vpipe导读](../../references/vpipe.md) · 本地图实验

固定来源：`tgo-app-dev/vpipe@c2989a76841888e4d8d208fad4f476c487710962`，工作树干净，2026-10-15。`apple-silicon/coreml/ane-emitter.h`全文、`.cc`的115–150、235–365、415–448行；逐文件SHA和阅读范围见[同一来源清单](../README.md)。

## 可迁移机制

`AneGraphSpec`明确区分Matmul、SwiGluFfn、GeluFfn、bk/bn、runtime weights、checkpoint的`[out,in]`布局以及bias。`AneEmitter`的投影构造按N tile收集列，再按K tile完成线性归约，最后concat。

runtime模式从完整矩阵输入切tile；常量模式的tile写入各自blob slab。合法性检查对常量要求tile整除，runtime允许较窄的末tile。这是上游实现合同，不是MIL或硬件对tail的普遍限制。

SwiGLU的gate/up分别投影，bias在各自完整投影归约后才加；SiLU与乘法之后才执行down。不能将非线性移进K partial；down bias也不能每个partial重复加。runtime布局避免checkpoint权重每次大转置的收益，要与预测端的布局代价一起测；本页不继承上游注释中的速度数字。

## 不能一起照搬的格式假设

header明确说明直接写`.mlmodelc`所依赖的MIL、83-byte header、weight entry/header与64-byte对齐是对编译产物的推断，不是Apple公开格式合同。`self_test()`用数值而不只用load成功检查它，失败应保留GPU退路。

本地实验采用coremltools→公共compile_model，不复制bundle writer。这样只研究图内tiling与布局机制；它不证明上游手写bundle在当前OS可用，也不包含上游worker的共享IO能力。

## 该怎样评价一次移植

同一source数组、相同gate/up顺序与down消费者，分别测frozen/runtime、monolithic/tiled、serial/concurrent；compile/load/first/warm与输入准备/restore/join分开。性能探索缓存不能继承不同checkpoint或LoRA的数值资格。

本轮上游执行：`not-run`。原创MIL实验的结果在[研究工作台](../../MAINTENANCE.md)，不能写成“vpipe在此机达到某TOPS”。新的OS、不同layout/tile或原生shared IO是重开条件。

许可：根LICENSE为Apache-2.0，Copyright 2026 T-Go LLC；第三方另见THIRD_PARTY_LICENSES.md。本页不分发其代码资产。

[固定版本header](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.h) · [固定版本实现](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc)。
