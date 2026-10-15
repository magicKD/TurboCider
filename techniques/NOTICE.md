# 来源、署名与独立实现边界

[总目录](README.md) · [设计样本](designs/README.md) · [机器可读吸收账本](references/absorbed-designs.json)

本目录原创文档、Python/Metal/Objective-C++ 教学代码遵循本目录的 [MIT license](LICENSE)（保留仓库原版权声明）。它们是用于解释机制和拒绝反例的独立实现，不是把外部项目的完整 runtime、模型代码或权重打包进来；不宣称格式兼容或第三方 bit-exact 复现。

思想和工程经验的明确出处：Splash/incoai（owned command payload、pack/consumer 协同）；带 ANE 扩展的 Splash snapshot（channel FFN、双 bank、rows calibration 与 handoff）；vpipe/tgo-app-dev（阶段内存协作、固定驻留子集和 typed refill）；Sana/NVlabs 的 sol-engine snapshot（capability/effect 接缝与 schedule）；KernelWiki/mit-han-lab（可搜索技术卡与证据等级组织）。另吸收mac_local_ai的消费者驱动ABI/精度岛、mac_transformer的拓扑与deadline planner、flux2-engine的persistent hybrid分工，以及mflux的FLUX.2单块图结构；[专项账本](references/gpu-ane-sources.json)记录实际读取文件SHA与工作树状态，新增demo仍为独立重写而非直接拷贝上述运行时。源码检查版本与原文件 SHA 保留在 [sources.json](references/sources.json)，**仅用于溯源，不是离线运行前置条件**。

Public Core ML 和 Metal 示例使用 Apple 公共 SDK/API。计算设备策略不等于实际 ANE placement，INT8 特征/存储不等于 INT8 硬件算术；SDK 与运行时要求由脚本和编译器明确检查。

若以后直接复制第三方代码/修改其具体实现，必须另行保留对应文件的版权、许可证和 NOTICE，并记录修改范围。不要因现在采用独立教学实现而推断原项目代码也可无条件改用 MIT；外部项目及权重仍受各自原许可约束。
