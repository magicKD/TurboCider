# 查询合同：主题、别名与版本复查不授予资格

[总目录](../README.md) · [自动主题索引](../queries/README.md) · [固定KernelWiki来源](../sources/repos/kernelwiki-retrieval.md) · [工具说明](../scripts/README.md)

本目录不是第二张页面清单。页面/代码/source/evidence仍在`labs/catalog.json`和`references/sources.json`登记；前者的`facets`覆盖全部已登记技术卡。`vocabulary.json`只定义受控值和单跳别名，`version-claims.json`只绑定已有页面ID与历史内容身份。查询只读这些本地资产，不访问产品/兄弟源码、安装依赖、联网、下载模型、探测机器或运行实验。

## 1. 标签只描述页面主题

支持`stage / operator / backend / device / symptom / semantics / representation / boundary / api`，以及原有`section / type`。具体取值只在[vocabulary.json](vocabulary.json)维护；从同一登记生成[主题索引](../queries/README.md)，不在多处复制取值表。

```sh
python3 -B techniques/scripts/query.py '前馈网络' --stage dit --backend private-ane --limit 3
python3 -B techniques/scripts/query.py --stage vae --symptom memory-bound --limit 5
python3 -B techniques/scripts/query.py --backend coreml --api public --limit 5
```

同维度重复值与跨维度均为AND，关键词组之间仍为OR并排名。别名归一后重复filter拒绝；未知dimension/value报错，不猜测最接近的标签。技术卡显式填写所有facet，空数组表示未分类，不是通用适用。source/evidence可以未分类，不从关联章节继承标签；输出`metadata_coverage`明确披露缺口。

一页可同时讨论public/private、数值等价/近似，`--api public`只表示它谈公共接口，**不保证整页没有私有路径**；`--semantics numerically-equivalent`也不为其中所有候选授予等价资格。`representation`是涉及的表示，不是两种格式相互兼容；`device`是涉及的物理资源，不是实际placement。先展开实际实现、来源与失败证据，再做选择。

## 2. 单跳别名不抹平语义

中文“前馈网络”和FFN、Core ML和coreml可以辅助查询。最长phrase先匹配，同一canonical组只取最高variant分数，不因多写几个同义词重复加分；原query、归一filter和keyword expansions均返回。中文词语可出现在连续短句中，不需要伪称完整自然语言理解。

NAX不映射ANE，FP8不映射Q8，GGUF容器不映射Q4/Q8；SoC基础/Pro/Max不互相映射或折成GPU性能profile。词表验证拒绝canonical碰撞、alias链/歧义和这些明确危险映射；它不能自动证明任意新别名的语义，新增别名仍须人工复核。精确环境字符串不做alias、semver范围或硬件能力推断。

## 3. 版本复查是告警，不是compatible/qualified

当前登记容易误用的历史结论：MLX0.32.0 same-dtype Copy alias、Qwen warm shared FFN组件范围、完整block导出失败不得晋升，以及BGRA媒体交接的CPU/GPU组件边界。每条有固定statement、已有page IDs、source/evidence SHA256、明确observed context、复核日期和有限review window。不是所有页面都已取得版本声明；空列表不表示无需复查。

```sh
python3 -B techniques/scripts/check_knowledge.py --as-of 2026-10-16 --strict
python3 -B techniques/scripts/query.py 'Copy' --context '{"mlx":"0.33.0"}' --as-of 2026-10-16 --limit 3
python3 -B techniques/scripts/get_page.py gpu-ane/private-ffn-boundary.md \
  --follow-evidence --max-bytes 24000 --as-of 2026-10-16
```

`--context`是调用方提供的精确历史环境JSON，不是自动读取当前机器；可用context字段由本地合同定义，值必须为有界字符串。`--as-of`可固定日期以便复测，省略时用本地日期；“0.33.0”仅为刻意改变的合成测试值，不宣称该版本当前存在或最新。

| 状态 | 含义 |
| --- | --- |
| `needs-review` | 绑定内容变化、给定环境不同、复核日期在未来，或达到复查窗口 |
| `context-incomplete` | 尚未发现上述变化，但缺少精确环境字段或提供了声明未覆盖的字段 |
| `recorded-context-only` | 仅所有给定记录字段相同且当前未触发复查；不是兼容、正确性或性能资格 |

状态附带具体changed/missing/uncovered字段、content binding与日期原因；历史页面不因告警消失，也不覆盖旧receipt。`--strict`遇到needs-review返回1，格式错误返回2；缺上下文保留incomplete而不是伪造全机器失败。techniques-only隔离检查包含strict版本复查，不需要硬件环境，仍检查schema/引用/hash并明确披露不完整上下文；达到日期窗口后需人工复核，不应自动刷新日期让CI通过。查询和展开重新核对绑定的本地内容，index加载后换源也不能静默沿用旧hash状态。所有路径均保持`retrieval_grants_qualification=false`。

## 4. 校验与维护

```sh
python3 -B techniques/scripts/check_knowledge.py --as-of 2026-10-16 --strict
python3 -B techniques/scripts/generate_query_index.py --check
python3 -B -m unittest discover -s techniques/tests -p 'test_retrieval_contracts.py'
python3 -B techniques/scripts/check_standalone.py --isolate
```

新增章节时在既有catalog中填facet，不编辑生成索引；新增source/evidence按自身内容标注，不借关联关系提升状态。更新结论前读原始来源/receipt，按需要重跑独立验证，再复核日期/内容绑定；不能机械替换hash消除告警。未知OS/SDK/MLX/SoC必须建立自己的profile和完整消费者/请求证据。

本轮不新增加速或设备实验，真实block raw持久化、完整request/画质、更多层/LoRA、内存/冷/长时和其他roadmap研究仍未完成。检索改进不能替代这些工作。
