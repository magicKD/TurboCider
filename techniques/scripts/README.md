# 无依赖辅助脚本与CPU参考

[总目录](../README.md) · [现有设备工具](../profiling/tools.md)

这里维护首版离线辅助工具；新增逐主题参考实现与真实设备实验统一在 [实验室](../labs/README.md)，不和这些纯CPU工具混称。

Python3.9+标准库；本轮验证使用本机Python。默认脚本不联网、不下载模型、不调用GPU、不改系统参数、不删除用户数据；`check_standalone.py --isolate --device ...`是显式选择的小型设备验证例外，`fetch_references.py --download`显式允许新建参考源码checkout。只读输入，结果打印stdout；若要保存报告，由调用方明确选择新产物路径。测试临时fixture使用系统临时目录并自行回收。

## query.py / get_page.py

```sh
python3 -B techniques/scripts/query.py LoRA --type technique --section gpu-ane --limit 3
python3 -B techniques/scripts/get_page.py gpu-ane/lora.md --follow-sources --include-code --max-bytes 12000
```

直接读取现有`labs/catalog.json`与`references/sources.json`，不依赖尚未建立的词表或第二份章节清单。当前覆盖catalog章节与`scope=local`来源；不扫描整个工作区、不读取产品源码或兄弟checkout，也不执行被查询的代码。章节ID从相对路径生成，例如`technique-gpu-ane-lora`；重命名章节会改变ID。

关键词支持中文原文与英文，不做语义别名扩展；多词按OR匹配，再按标题、case/controls/symbols、正文命中加权排序，ID用于稳定排序。`--type technique|source|evidence`与`--section`只过滤文档类别，重复过滤值按AND处理，不代表该页通过了设备/性能资格。当前没有stage/device/backend等精细元数据，未知过滤项拒绝，不猜测标签。

展开只接受登记的ID或相对路径，拒绝绝对路径、越界路径与外部symlink。正文和可选来源/代码共用字节预算；`truncated`、`omitted_pages`、`omitted_code`明确披露裁剪与省略，不默默丢失来源。JSON结构和元数据不计入正文预算。默认输出JSON，查询另有`--compact`。

章节的`source_ids`与`evidence_ids`都引用既有来源登记，不依赖另一份手工页面表。`--follow-evidence`（别名`--include-negative-results`）显式展开登记的本地实验/失败报告；来源卡与本机证据是不同kind，不能把source-inspected晋升为measured。

较大的证据JSON可用返回的`next_byte_offset`继续：`--offset-bytes N`只作用于主资产，来源/代码仍从头按共同预算展开。分页保留UTF-8字符边界；不接受负offset、越文件尾或字符中间位置，不因报告过大而默默丢弃后面的失败/确认样本。

## 来源索引、实验监督与证据归一化

`generate_source_index.py`从同一登记生成[sources目录](../sources/README.md)，默认只打印Markdown，`--check`拒绝索引漂移；不会fetch上游。

`run_experiment.py --timeout ... --output-dir ... -- command args`是显式执行边界，不由检索调用。只监督自己新建的进程组，超时保留stdout/stderr与失败receipt；不输出环境变量、不触发隐式安装，也不以进程停止冒充设备drain。

`curate_ffn_campaign.py`只读取调用方明确指定的报告，保留失败和全部warmup/样本，输出可移植JSON。failed numerical arm没有speedup；配对重复、缺失、错point或非法wall拒绝。四对样本的seeded bootstrap区间只作探索性不确定性，不授予硬件/请求资格。

`host_trace.py report.json --trial N`只把一个已记录的trial转为host-only Chrome Trace事件JSON，不补造GPU/ANE时钟、不启动viewer或实验。已有[真实Qwen案例](../review/qwen-ffn-tiling-20261015.md)展示为什么需要把完整GPU分母、同图serial/hybrid诊断与计划数据分开。

`curate_gpu_campaign.py --root ... --fixture ... --run group:run-name`是独立只读data verifier，不导入候选benchmark模块；复核固定门槛、pair、原始source budget标记、source snapshots与ratio。格式内mechanism收益与原始模型组件收益分别标注；后者budget失败时拒绝暴露model ratio。`--diagnostic`与`--failed-launch`保留所选摘录/精度控制和frontend失败，不把programming error变成硬件不支持。结果见[GPU affine研究](../review/gpu-affine-20261016.md)。

## fetch_references.py

```sh
python3 -B techniques/scripts/fetch_references.py --project R01 --project R08 \
  --reference-root "$TC_REFERENCE_ROOT"
```

[候选清单](../candidates/projects.json)保留roadmap项目ID；列入清单不等于已审读、兼容或复现。默认只生成计划/只读检查，缺失目录不会创建。进度写stderr，stdout始终是单个JSON receipt。

只有显式增加`--download`才联网浅克隆缺失项目，可选sparse checkout；不运行上游脚本、不安装依赖。已有目录从不fetch/reset/clean，即使同时传入`--download`也只检查；检查禁用可选Git写锁和fsmonitor命令。失败或超时保留部分checkout供人工检查，不自动删除。`--output`可显式保存新receipt，已有文件拒绝覆盖；原始receipt应保存在已忽略的`outputs/`。

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
