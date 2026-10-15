# 无依赖辅助脚本与CPU参考

[总目录](../README.md) · [现有设备工具](../profiling/tools.md)

这里维护首版离线辅助工具；新增逐主题参考实现与真实设备实验统一在 [实验室](../labs/README.md)，不和这些纯CPU工具混称。

Python3.9+标准库；本轮验证使用本机Python。默认脚本不联网、不下载模型、不调用GPU、不改系统参数、不删除用户数据；`check_standalone.py --isolate --device ...`是显式选择的小型设备验证例外。只读输入，结果打印stdout；若要保存报告，由调用方明确选择新产物路径。测试临时fixture使用系统临时目录并自行回收。

## check_wiki.py

```sh
python3 techniques/scripts/check_wiki.py
```

默认检查techniques内Markdown链接、入链、标题、个人路径/明显key与local scope来源字面符号。产品/兄弟仓库链接和integration scope来源单独计数，不是自包含核心要求；`--include-integrations`显式追加原完整来源核查，缺失的optional checkout warning、缺失的required product anchor失败。

这不是语义审稿、互联网URL存活、Markdownanchor或编译正确性校验。不能用绿色链接检查证明手册全面、性能真实或模型质量合格；覆盖/内容仍需 [人工审计](../review/coverage.md)。

## check_standalone.py

```sh
python3 -B techniques/scripts/check_standalone.py --isolate
```

静态核查9项已吸收设计、6个模型阶段、必需本地代码/测试、imports和native quoted includes。随后只复制techniques到新临时目录，忽略继承的PYTHON路径，实际运行全部unit tests、wiki/catalog/checker和所有CPU cases；不复制产品源码、兄弟仓库、模型或依赖环境。通过不能代替语义审稿，变更检测有反例测试。

可用具备依赖的Python显式选择真实设备验证，各项串行：

```sh
python -B techniques/scripts/check_standalone.py --isolate \
  --device metal-gemm --device coreml-frozen --device coreml-runtime \
  --device coreml-hybrid --device shared-int8 \
  --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/new-isolated-device-report.json
```

设备选项分别要求本机Apple SDK、NumPy/CoreML，hybrid另需MLX。overlay是可选已安装包目录，不安装或下载；沙箱不能访问设备时需明确授权，不伪造CPU通过。新报告永不覆盖，临时副本和其构建产物自动回收。完整产品fixtures在[integrations](../integrations/README.md)，不纳入核心隔离套件。

## compare_runs.py

```sh
python3 techniques/scripts/compare_runs.py techniques/examples/paired-runs.json
```

输入是刻意简化的normalizedcampaignJSON，不自动解析现有任意benchmarkreport：显式scope、共同hardware/software/workload/weights/adapter/cache/precision身份；每variant固定build，每个cold/warm pair各一baseline/candidate。原始执行顺序保留为A/B。

输出warm中位数、nearest-rankp90、worst、pooled中位倍率与paired倍率中位数，cold数量单列。重复/缺配对、混build、NaN/非正duration、sample试图覆盖共同identity拒绝；任何error/cancelled/qualityfail或unknown均保留样本数量并不输出speedup。所有输出`qualification_passed=false`：它不验证调用方的质量标签、hash或缓存事实，不自动认定显著性/发行资格。

将真实report转换成此schema时应人工核对原始输入和actualreceipts，不能只搬有利热样本；metadata无法证明的事实写清unknown而不是伪造pass。

## interval_peak.py

```sh
python3 techniques/scripts/interval_peak.py techniques/examples/memory-plan.json
```

输入有序serializedphases与claims：`name/allocation/bytes/first/last`。区间两端包含，跨phase存活按区间计；同allocationalias只计一次，bytes必须是完整backing并一致。可选target/reserve输出declaredslack；始终`hard_memory_bound=false`。

仅声明模型，不读系统、不知道driver/CoreML未知分配、不支持并发请求时间线、不计算自动pinnedcount、不替代nativeadmission。不同owner复用同一个allocationID会低估，因此generation变化要新ID。

## history_index.py

显式传sessions目录和项目cwdcomponent（可重复），不要扫整个个人history后发布原文：

```sh
python3 techniques/scripts/history_index.py "$TC_SESSIONS" \
  --project TurboCider --project splash --project mac_transformer
```

默认只有sessionmetadata、topiccounts与命中行号，无消息正文；不导出toolpayload/system/developer/analysischannel，常见环境/goal和reviewer段落排除。cwd组件匹配只能筛候选，不能证明内容属于该项目或结论有效；session起始时间也不是每条事件时间。

`--excerpts --limit 3`仅供明确选择的私有本地审读：best-effort脱敏仍可能包含敏感文本，**不得提交或外发**。本wiki只保留经当前代码/文档复核的概括，不含rawsession或私有对话。知识来源层次见 [历史经验](../references/history-lessons.md)。

`--match 'W8A8|headroom' --tail-excerpts`可只保留最后N段相关assistantprose；默认仍不导出正文。正则只筛whitelist消息，不搜索toolarguments；旧contextsummary可能混入，因此结论仍需source/report交叉核对。

## split_ffn_reference.py

```sh
python3 techniques/examples/split_ffn_reference.py
```

合成CPUfloatoracle，验证不均匀channelcuts、row split、gate/upLoRA在SiLU前、down-A-ranks相加后一次B、bias/residual一次。不是设备实现、precision模拟或性能benchmark；不把它塞生产推理。

## 测试

```sh
python3 -m unittest discover -s techniques/tests -v
```

涵盖脚本拒绝路径/数值、CPU拆分、history默认不导出正文和真实wiki链接。测试规模和本轮结果见 [当前自包含验收](../review/self-contained-validation.md)；[首版记录](../review/validation.md)保留历史含义。
