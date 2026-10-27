# 新模型加速与交付清单

<!-- knowledge
id: technique-templates-new-model-checklist
type: guide
-->

[总目录](../README.md) · [实施顺序](../guides/new-model.md)

逐项附证据，只有实际满足才打勾；不适用也要写原因。

## 模型与基线

- [ ] checkpoint/组件/tokenizer/adapter来源和精确encoding，所有原文件只读。
- [ ] topology、gate/up配对、H/F/heads、实际rows/phase、mask/position、scheduler、causalVAE事实表。
- [ ] 原始GPU完整请求成功；same-noise/conditioning layer→step→latent→media对照。
- [ ] 优化GPU基线，compiled/attention/LoRA/cache有效选项完全记录。
- [ ] 请求operation/ref/audio等不支持项显式拒绝，不裁剪/少steps冒充优化。

## GPU/ANE正确性与所有权

- [ ] 合法partition代数、logical→physicallayout、bias/residual一次、原始pitch保留。
- [ ] 完整LoRA gate/up在激活前、down消费完整correctedhidden、base/A/B/base隔离。
- [ ] shape/bucket/tail/padding/stride/alias/finite/headroom与encoding拒绝门。
- [ ] input/weight/metadata/outputs的sourcegeneration/owner保活至物理完成。
- [ ] actualproducer-ready/ANE-done/consumer-done，无queue依赖环。
- [ ] boundedW/A/IOslots/pipelines，futureidentity不匹配丢弃，无第三bank隐性膨胀。
- [ ] latechunk/readererror/GPucallback失败不发布半成品，整operationGPU重算与真实label。
- [ ] cancel/retry、timeout/quarantine/latecompletion、teardown与borrowedsource释放顺序。

## 性能与内存

- [ ] alone/concurrent/完整block/request各自scope；exposedwait不代替branchalone。
- [ ] fixedsamples/coldwarm/cache/ABBA与load规则；实际全部attempt保留。
- [ ] 首步/后续、不同reference/真实LoRA、不同比例和最小bucket分别验证。
- [ ] completequality的独立候选试验；costmodel不代替真实运行。
- [ ] 全phase/trunk/activation/history/typedslot/CoreML/媒体/临时磁盘预算。
- [ ] 独立load→exit进程树采样、gaps/unknown、compression/swap与各计数不重叠相加。
- [ ] 原目标小RAM硬件或清楚限定的模拟；budgethint不标hardcap。

## 质量与产品

- [ ] 多prompt/seed/holdout，数值与感知/语义门均按预设；finite始终严格。
- [ ] 视频完整帧/FPS/PTS、motion/闪烁/身份、音频与必要AV/lipsync；codec与模型误差分层。
- [ ] 新组合单独验证，不把cache×ANE×sparse历史收益相乘。
- [ ] modelmodule/Session/registry、source/memory/actualreceipt复用既有框架。
- [ ] device/OS/shape/artifact/adapter精确门控，unknownGPUfallback，用户profile不能绕硬件。
- [ ] Public默认/Private研究与experimentalmarkers发行隔离检查。
- [ ] CLI/API/App实际请求、保存/导入/历史复用/切换/取消/卸载和实际loadedbinary验证。
- [ ] 便携文档、命令、小型证据、失败结论、复测触发条件；无权重/媒体/凭据/个人路径入库。

## 最后问自己

陌生工程师能否只用这些文档定位实现、理解合法性、选测试、复现对照并知道什么时候不启用？如果仍必须找到某条历史聊天才知道关键guard，就把那条经验补到手册。
