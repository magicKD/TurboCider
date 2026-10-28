# 知识库使用与维护

<!-- knowledge
id: technique-guides-knowledge-base
type: guide
-->

[总目录](../README.md) · [工程指南](README.md) · [技术卡](../templates/technique-card.md) · [维护规则](../MAINTENANCE.md)

这里保存可迁移的优化认识，不承担向读者分发原始实验证据或验证环境的职责。方法可以来自源码阅读；案例可以是失败案例；没有本机实验不妨碍收录，但不能虚构结果。

## 内容分层

| 类型 | 回答的问题 |
| --- | --- |
| `method` | 为什么有效，什么时候值得尝试，代价和失效条件是什么？ |
| `implementation` | 数据流、布局、selector 和接入位置怎样设计？ |
| `case-study` | 一个具体问题如何形成假设、比较方案并得到有限结论？ |
| `guide` | 如何排查、迁移或开展一次优化？ |
| `navigation` | 下一步读哪里？ |
| `source` | 固定版本源码、论文和官方文档在哪里，值得读什么？ |

原始数组、JSON、采集程序、测试回执和逐轮日志不在本目录。案例中的历史数字只用于理解取舍，不代表当前机器验证或产品推荐。

## 阅读与设计

任务事实 → 找机制 → 读实现/反例 → 写可证伪假设 → 在任务授权范围内实验 → 提炼认识。

优先回答被省略的计算/流量、必须维持的语义与所有权、新增搬运/同步/控制成本。具体工作流见[优化流程](optimization-workflow.md)，方法使用[技术卡](../templates/technique-card.md)。阅读不触发实验。

## Markdown 原生检索

页面标题、链接和短注释就是检索输入，没有 catalog 或实验数据依赖：

```text
<!-- knowledge
id: technique-example
type: method
stage: prefill, decode
operator: gemm
backend: metal
-->
```

保留已有 ID；重命名只调整文件路径。必要时用 `aliases:`保留旧 ID。其余可选标签是 `device/symptom/semantics/representation/boundary/api`。空标签表示未分类，不表示通用适用或验证通过。新增标签应简明一致，避免把 NAX=ANE、容器=量化格式等不同概念合并。

```sh
python3 -B techniques/scripts/query.py DeepCache --intent learn --limit 3 --compact
python3 -B techniques/scripts/query.py small-M --intent implement --limit 3 --compact
python3 -B techniques/scripts/query.py '共享' --section gpu_ane --page-type method --compact
python3 -B techniques/scripts/get_page.py metal/gemm-and-quantization.md --follow-sources
```

来源检索用 `--intent sources`。工具可向 stdout 输出结构化内容供 AI 使用，但不生成或依赖知识库里的 JSON 文件。旧 `evidence` 意图只兼容为来源阅读，不提供实验包。

## 新增或更新一条知识

先查同机制页面 → 补一句话机制与前提 → 给实现入口/伪代码 → 写代价、反例和迁移 → 链接固定来源 → 更新导航。

保留结论的范围，不保留整个实验目录；原因尚未证实就标假设。来源链接和学习材料不等于当前实测或上线资格。

维护检查只验证链接、元数据、检索和无数据文件规则，不运行模型，也不复算历史实验：

```sh
python3 -B techniques/scripts/check_wiki.py
python3 -B techniques/scripts/generate_query_index.py --check
python3 -B -m unittest discover -s tests/repository -p 'test_technique_knowledge.py'
```
